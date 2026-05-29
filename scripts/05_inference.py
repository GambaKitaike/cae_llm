"""
05_inference.py - CAE LLM 推論スクリプト（CLI + Streamlit UI）

学習済みLoRAアダプターを読み込み、CAE特化チャットを提供する。

使用例:
    # CLI モード（対話型）
    python scripts/05_inference.py \
        --adapter_path outputs/sft/final_adapter \
        --mode cli

    # Streamlit UI モード
    streamlit run scripts/05_inference.py -- \
        --adapter_path outputs/sft/final_adapter \
        --mode ui

    # 単発クエリ（バッチ処理等に便利）
    python scripts/05_inference.py \
        --adapter_path outputs/sft/final_adapter \
        --mode single \
        --query "接触解析でペナルティ剛性を適切に設定する方法を教えてください"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model_for_inference
from src.utils import build_prompt, DEFAULT_SYSTEM_MESSAGE

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 生成ユーティリティ
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    question: str,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """
    質問文から回答テキストを生成する。

    Returns:
        生成された回答テキスト（プロンプト部分を除く）
    """
    prompt = build_prompt(user=question, system_message=system_message, add_answer=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CLI モード
# ---------------------------------------------------------------------------

def run_cli(model, tokenizer, gen_kwargs: dict) -> None:
    """対話型 CLI モードで推論を実行する。"""
    print("\n" + "=" * 60)
    print("CAE LLM チャットアシスタント (CLI モード)")
    print("終了するには 'quit' または 'exit' と入力してください")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("あなた: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if question.lower() in {"quit", "exit", "q", "終了"}:
            print("終了します。")
            break

        if not question:
            continue

        print("アシスタント: ", end="", flush=True)
        response = generate_response(model, tokenizer, question, **gen_kwargs)
        print(response)
        print()


# ---------------------------------------------------------------------------
# Streamlit UI モード
# ---------------------------------------------------------------------------

def run_streamlit_ui(adapter_path: str, gen_kwargs: dict) -> None:
    """Streamlit UI モードで推論を実行する。"""
    try:
        import streamlit as st
    except ImportError:
        logger.error("streamlit がインストールされていません。pip install streamlit を実行してください。")
        sys.exit(1)

    st.set_page_config(
        page_title="CAE LLM アシスタント",
        page_icon="⚙️",
        layout="wide",
    )

    st.title("⚙️ CAE LLM チャットアシスタント")
    st.caption("CAE（有限要素解析・流体解析・構造解析）専門アシスタント")

    # サイドバー: 生成パラメータ設定
    with st.sidebar:
        st.header("生成設定")
        max_new_tokens = st.slider("最大生成トークン数", 64, 1024, gen_kwargs.get("max_new_tokens", 512), step=64)
        temperature = st.slider("Temperature", 0.1, 1.5, gen_kwargs.get("temperature", 0.7), step=0.05)
        top_p = st.slider("Top-p", 0.5, 1.0, gen_kwargs.get("top_p", 0.9), step=0.05)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 1.5, gen_kwargs.get("repetition_penalty", 1.1), step=0.05)

        st.divider()
        st.markdown(f"**アダプターパス**\n\n`{adapter_path}`")

        if st.button("会話をリセット"):
            st.session_state.messages = []
            st.rerun()

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        with st.spinner("モデルを読み込み中..."):
            base_model_name = _get_base_model_name(adapter_path)
            st.session_state.model, st.session_state.tokenizer = load_model_for_inference(
                model_name=base_model_name,
                adapter_path=adapter_path,
            )
        st.success("モデルの読み込み完了")

    # 過去のメッセージを表示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ユーザー入力
    if prompt := st.chat_input("CAEに関する質問を入力してください..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                current_gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                }
                response = generate_response(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    prompt,
                    **current_gen_kwargs,
                )
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def _get_base_model_name(adapter_path: str) -> str:
    """
    adapter_path の adapter_config.json から base_model_name_or_path を読む。
    失敗した場合はデフォルトを返す。
    """
    import json
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path", "llm-jp/llm-jp-3-1.8b")
    return "llm-jp/llm-jp-3-1.8b"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAE LLM 推論スクリプト（CLI / Streamlit UI / 単発クエリ）"
    )
    parser.add_argument(
        "--adapter_path", type=str, default="outputs/sft/final_adapter",
        help="LoRAアダプターのパス (default: outputs/sft/final_adapter)"
    )
    parser.add_argument(
        "--base_model", type=str, default=None,
        help="ベースモデル名 (指定しない場合はadapter_config.jsonから自動取得)"
    )
    parser.add_argument(
        "--mode", choices=["cli", "ui", "single"], default="cli",
        help="実行モード: cli=対話型, ui=Streamlit, single=単発クエリ (default: cli)"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="単発クエリ（--mode single の場合に使用）"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="最大生成トークン数 (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="生成温度 (default: 0.7)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p サンプリング (default: 0.9)"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.1,
        help="繰り返しペナルティ (default: 1.1)"
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", default=True,
        help="4-bit量子化で推論する (default: True)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }

    if args.mode == "ui":
        # Streamlit は直接 run するとモデルロードを内部で行う
        run_streamlit_ui(adapter_path=args.adapter_path, gen_kwargs=gen_kwargs)
    else:
        # CLI / single: アダプターを読み込んで推論
        base_model = args.base_model or _get_base_model_name(args.adapter_path)
        logger.info("Loading model: %s + %s", base_model, args.adapter_path)
        model, tokenizer = load_model_for_inference(
            model_name=base_model,
            adapter_path=args.adapter_path,
            load_in_4bit=args.load_in_4bit,
        )

        if args.mode == "single":
            if not args.query:
                logger.error("--mode single には --query が必要です。")
                sys.exit(1)
            response = generate_response(model, tokenizer, args.query, **gen_kwargs)
            print(response)
        else:
            run_cli(model, tokenizer, gen_kwargs)
