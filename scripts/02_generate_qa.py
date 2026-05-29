"""
02_generate_qa.py - CAEドキュメントから合成Q&Aを生成するスクリプト

前処理済みJSONL（data/processed/）を読み込み、
OpenAI APIまたはローカルLLMを使ってCAE特化のQ&Aペアを生成する。

出力: data/synthetic/sft_train.jsonl, sft_eval.jsonl
      各行: {"input": "質問", "output": "回答"}

使用例 (OpenAI API):
    python scripts/02_generate_qa.py \
        --input_dir data/processed \
        --output_dir data/synthetic \
        --backend openai \
        --model gpt-4o-mini \
        --pairs_per_chunk 3

使用例 (ローカルLLM / vLLM):
    python scripts/02_generate_qa.py \
        --input_dir data/processed \
        --output_dir data/synthetic \
        --backend local \
        --base_url http://localhost:8000/v1 \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --pairs_per_chunk 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# プロンプト定義
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
あなたはCAE（Computer-Aided Engineering）の専門家です。
与えられたCAE技術文書テキストを参照し、
エンジニアが実務で直面しそうな質問とその模範的な回答を生成してください。

以下のルールに従ってください:
- 質問は具体的で技術的なものにすること（例: ソルバー設定、境界条件、材料定義、収束問題、後処理）
- 回答は技術的に正確で、文書の内容を根拠に説明すること
- JSON配列形式で出力すること: [{"input": "...", "output": "..."}, ...]
- 入力テキストに記載のない情報を回答で作り上げないこと
- 質問・回答ともに日本語で記述すること
"""

USER_TEMPLATE = """\
以下のCAE技術文書テキストから、{n_pairs}個の質問回答ペアをJSON配列形式で生成してください。

--- テキスト ---
{chunk_text}
--- 終わり ---

JSON配列のみを出力してください（前後に説明文は不要です）。
"""


# ---------------------------------------------------------------------------
# APIクライアントファクトリ
# ---------------------------------------------------------------------------

def build_openai_client(base_url: str | None = None):
    """OpenAI互換クライアントを構築する。"""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def generate_qa_pairs(
    client,
    model: str,
    chunk_text: str,
    n_pairs: int = 3,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> list[dict]:
    """
    チャンクテキストから Q&A ペアリストを生成する。

    Returns:
        [{"input": "...", "output": "..."}, ...]  （失敗時は空リスト）
    """
    user_msg = USER_TEMPLATE.format(chunk_text=chunk_text, n_pairs=n_pairs)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()
            pairs = _parse_json_response(raw)
            valid = _validate_pairs(pairs)
            return valid
        except Exception as e:
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    return []


# ---------------------------------------------------------------------------
# パース・バリデーション
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> list[dict]:
    """LLMのレスポンスからJSON配列を抽出してパースする。"""
    # コードブロックの除去
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.strip("`").strip()

    # 先頭の [ から末尾の ] までを抽出
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        logger.debug("No JSON array found in response: %s", raw[:200])
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.debug("JSON parse error: %s\nRaw: %s", e, raw[:200])
        return []


def _validate_pairs(pairs: list) -> list[dict]:
    """input/output キーを持つ有効なペアのみを返す。"""
    valid = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if inp and out and len(inp) >= 10 and len(out) >= 20:
            valid.append({"input": inp, "output": out})
    return valid


# ---------------------------------------------------------------------------
# JSONL読み込み・書き込み
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def generate_from_directory(
    input_dir: Path,
    output_dir: Path,
    client,
    model: str,
    pairs_per_chunk: int,
    eval_ratio: float,
    seed: int,
    max_chunks: int | None,
    temperature: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # すべてのJSONLファイルからチャンク収集
    all_chunks: list[str] = []
    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        records = read_jsonl(jsonl_path)
        for r in records:
            text = r.get("text", "")
            if text:
                all_chunks.append(text)
    logger.info("Total chunks to process: %d", len(all_chunks))

    if max_chunks and len(all_chunks) > max_chunks:
        random.seed(seed)
        random.shuffle(all_chunks)
        all_chunks = all_chunks[:max_chunks]
        logger.info("Limiting to %d chunks", max_chunks)

    all_pairs: list[dict] = []
    for chunk in tqdm(all_chunks, desc="Generating Q&A"):
        pairs = generate_qa_pairs(
            client=client,
            model=model,
            chunk_text=chunk,
            n_pairs=pairs_per_chunk,
            temperature=temperature,
        )
        all_pairs.extend(pairs)

    logger.info("Generated %d Q&A pairs total", len(all_pairs))

    if not all_pairs:
        logger.error("No Q&A pairs generated. Check API key and input data.")
        return

    random.seed(seed)
    random.shuffle(all_pairs)
    split_idx = max(1, int(len(all_pairs) * (1 - eval_ratio)))

    train_path = output_dir / "sft_train.jsonl"
    eval_path = output_dir / "sft_eval.jsonl"
    write_jsonl(train_path, all_pairs[:split_idx])
    write_jsonl(eval_path, all_pairs[split_idx:])

    logger.info("Saved train: %s (%d records)", train_path, split_idx)
    logger.info("Saved eval : %s (%d records)", eval_path, len(all_pairs) - split_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAEドキュメントから合成Q&Aペアを生成する"
    )
    parser.add_argument(
        "--input_dir", type=Path, default=Path("data/processed"),
        help="前処理済みJSONLのディレクトリ (default: data/processed)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/synthetic"),
        help="出力JSONLのディレクトリ (default: data/synthetic)"
    )
    parser.add_argument(
        "--backend", choices=["openai", "local"], default="openai",
        help="APIバックエンド: openai (OpenAI API) / local (vLLM等) (default: openai)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="使用するモデル名 (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="ローカルLLMのAPIエンドポイント (localバックエンド時に使用)"
    )
    parser.add_argument(
        "--pairs_per_chunk", type=int, default=3,
        help="チャンクあたりに生成するQ&Aペア数 (default: 3)"
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=0.05,
        help="評価データの割合 (default: 0.05)"
    )
    parser.add_argument(
        "--max_chunks", type=int, default=None,
        help="処理するチャンク数の上限（Noneで全件）"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="生成温度 (default: 0.7)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="ランダムシード (default: 42)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    base_url = args.base_url if args.backend == "local" else None
    client = build_openai_client(base_url=base_url)

    generate_from_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        client=client,
        model=args.model,
        pairs_per_chunk=args.pairs_per_chunk,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_chunks=args.max_chunks,
        temperature=args.temperature,
    )
