"""
CPT用・SFT用 PyTorch Dataset / HuggingFace Dataset ユーティリティ。

CPTDataset : ラベルなしテキストを next-token-prediction 形式で提供
SFTDataset : instruction/response ペアを build_prompt でフォーマットして提供
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.utils import build_prompt_from_row

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------

def _read_jsonl(path: str | Path) -> list[dict]:
    """JSONLファイルを読み込んでリストで返す。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _tokenize_and_truncate(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    """テキストをトークナイズしてmax_lengthでtruncateする。"""
    return tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )


# ---------------------------------------------------------------------------
# CPT Dataset（継続事前学習用）
# ---------------------------------------------------------------------------

class CPTDataset(Dataset):
    """
    テキストコーパスを next-token-prediction 用に提供する Dataset。

    JSONL形式: {"text": "..."}
    labels は input_ids と同一（CLM損失計算用）。
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
        text_column: str = "text",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

        records = _read_jsonl(data_path)
        self.texts = [r[text_column] for r in records if text_column in r]
        logger.info("CPTDataset: loaded %d samples from %s", len(self.texts), data_path)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        enc = _tokenize_and_truncate(text, self.tokenizer, self.max_length)
        input_ids = enc["input_ids"]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# SFT Dataset（指示チューニング用）
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Instruction/Response ペアを llm-jp-3 フォーマットで提供する Dataset。

    JSONL形式: {"input": "質問文", "output": "回答文"}

    mask_instruction=True（デフォルト）の場合、[/INST] より前のトークンの
    labels を -100 にマスクし、回答部分のみ loss が計算される。
    04_train_sft.py の MaskedLabelCollator と組み合わせて使う。
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
        user_key: str = "input",
        assistant_key: str = "output",
        system_message: str | None = None,
        mask_instruction: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.user_key = user_key
        self.assistant_key = assistant_key
        self.system_message = system_message
        self.mask_instruction = mask_instruction

        self.records = _read_jsonl(data_path)
        logger.info("SFTDataset: loaded %d samples from %s", len(self.records), data_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        kwargs = dict(
            user_key=self.user_key,
            assistant_key=self.assistant_key,
            add_answer=True,
        )
        if self.system_message:
            kwargs["system_message"] = self.system_message

        full_text = build_prompt_from_row(row, **kwargs)
        enc = _tokenize_and_truncate(full_text, self.tokenizer, self.max_length)
        input_ids = enc["input_ids"]
        labels = list(input_ids)

        if self.mask_instruction:
            # "[/INST]" より前をマスク (-100) して回答部分のみ損失を計算
            response_token = self.tokenizer.encode(" [/INST]", add_special_tokens=False)
            mask_end = _find_subsequence(input_ids, response_token)
            if mask_end != -1:
                for i in range(mask_end + len(response_token)):
                    labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _find_subsequence(seq: list[int], subseq: list[int]) -> int:
    """seq 内で subseq が最後に出現する開始インデックスを返す。見つからなければ -1。"""
    n, m = len(seq), len(subseq)
    for i in range(n - m, -1, -1):
        if seq[i:i + m] == subseq:
            return i
    return -1


def load_hf_dataset_for_cpt(
    data_path: str | Path,
    text_column: str = "text",
    eval_split_ratio: float = 0.05,
    seed: int = 42,
):
    """JSONL → HuggingFace Dataset（CPT用）を返す。"""
    from datasets import Dataset as HFDataset

    records = _read_jsonl(data_path)
    random.seed(seed)
    random.shuffle(records)

    texts = [{"text": r[text_column]} for r in records if text_column in r]
    split_idx = max(1, int(len(texts) * (1 - eval_split_ratio)))
    train_data = HFDataset.from_list(texts[:split_idx])
    eval_data = HFDataset.from_list(texts[split_idx:])
    logger.info("HF CPT dataset: train=%d, eval=%d", len(train_data), len(eval_data))
    return train_data, eval_data
