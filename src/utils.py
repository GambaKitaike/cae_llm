"""
トークナイザー・プロンプトテンプレート・設定ロードユーティリティ。
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 設定ロード
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    """YAMLファイルから設定を読み込む。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from: %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# プロンプトテンプレート
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_MESSAGE = (
    "あなたはCAE（有限要素解析・流体解析・構造解析）の専門アシスタントです。"
    "エンジニアの質問に対して、技術的に正確で分かりやすい回答を日本語で提供してください。"
)

# llm-jp-3 の Instruction format
PROMPT_TEMPLATE = (
    "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user} [/INST] {assistant}</s>"
)

PROMPT_TEMPLATE_NO_ANSWER = (
    "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user} [/INST]"
)


def build_prompt(
    user: str,
    assistant: str = "",
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    add_answer: bool = True,
) -> str:
    """
    llm-jp-3 instruction 形式のプロンプトを組み立てる。

    Args:
        user: ユーザーの質問文。
        assistant: アシスタントの回答（学習時は必須、推論時は空でOK）。
        system_message: システムプロンプト。
        add_answer: False の場合、回答部分を含まない推論用プロンプトを返す。
    """
    if add_answer:
        return PROMPT_TEMPLATE.format(
            system_message=system_message,
            user=user,
            assistant=assistant,
        )
    return PROMPT_TEMPLATE_NO_ANSWER.format(
        system_message=system_message,
        user=user,
    )


def build_prompt_from_row(
    row: dict,
    user_key: str = "input",
    assistant_key: str = "output",
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    add_answer: bool = True,
) -> str:
    """JSONL行辞書からプロンプトを組み立てる。"""
    user = row.get(user_key, "")
    assistant = row.get(assistant_key, "") if add_answer else ""
    return build_prompt(user=user, assistant=assistant, system_message=system_message, add_answer=add_answer)


# ---------------------------------------------------------------------------
# VRAM 情報表示
# ---------------------------------------------------------------------------

def print_gpu_info() -> None:
    """利用可能なGPUとVRAM情報をログ出力する。"""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                logger.info(
                    "GPU %d: %s | Total: %.1f GB | Reserved: %.1f GB | Allocated: %.1f GB",
                    i,
                    torch.cuda.get_device_name(i),
                    total,
                    reserved,
                    allocated,
                )
        else:
            logger.warning("CUDA is not available. Training will run on CPU (very slow).")
    except ImportError:
        logger.warning("torch not installed.")


