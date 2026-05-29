"""
04_train_sft.py - Phase 2: 指示チューニング（Supervised Fine-Tuning）スクリプト

TRL 1.5.0 対応版。
- SFTConfig（TrainingArguments の上位互換）を使用
- DataCollatorForCompletionOnlyLM が TRL 1.5.0 で削除されたため、
  SFTDataset（ラベルマスク済み）＋独自 PaddingCollator で代替

使用例:
    python scripts/04_train_sft.py --config configs/sft_config.yaml
    python scripts/04_train_sft.py --config configs/sft_config.yaml --report_to wandb
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model_for_training
from src.dataset import SFTDataset
from src.utils import load_config, print_gpu_info, DEFAULT_SYSTEM_MESSAGE

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as HFDataset
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ラベルマスク済みデータ用パディング Collator
# (TRL 1.5.0 で DataCollatorForCompletionOnlyLM が削除されたための代替)
# ---------------------------------------------------------------------------

@dataclass
class MaskedLabelCollator:
    """
    SFTDataset が出力する {input_ids, attention_mask, labels} をパディングする。
    labels の -100 はそのまま保持し、回答部分のみ loss が計算される。
    """
    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # HF Dataset 経由で list になった要素をテンソルに戻す
        def to_tensor(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)

        input_ids = pad_sequence(
            [to_tensor(f["input_ids"]) for f in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [to_tensor(f["attention_mask"]) for f in features],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [to_tensor(f["labels"]) for f in features],
            batch_first=True,
            padding_value=self.label_pad_id,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# PyTorch Dataset → HuggingFace Dataset 変換
# ---------------------------------------------------------------------------

def _to_hf_dataset(ds) -> HFDataset:
    """
    SFTDataset (PyTorch) → HuggingFace Dataset に変換する。

    TRL 1.5.0 の SFTTrainer は column_names 属性を要求するため、
    HF Dataset 形式に変換してから渡す。テンソルはリスト化して格納。
    """
    records = []
    for i in range(len(ds)):
        item = ds[i]
        records.append({
            "input_ids": item["input_ids"].tolist(),
            "attention_mask": item["attention_mask"].tolist(),
            "labels": item["labels"].tolist(),
        })
    return HFDataset.from_list(records)


# ---------------------------------------------------------------------------
# 学習設定構築（SFTConfig: TRL 1.5.0）
# ---------------------------------------------------------------------------

def build_sft_config(config: dict, report_to: str | None = None) -> SFTConfig:
    tc = config["training"]
    dc = config["data"]

    if report_to is None:
        report_to = tc.get("report_to", "none")

    warmup_ratio = tc.get("warmup_ratio", 0.05)
    warmup_steps = tc.get("warmup_steps", max(10, int(warmup_ratio * 1000)))

    return SFTConfig(
        output_dir=tc["output_dir"],
        num_train_epochs=tc.get("num_train_epochs", 3),
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=tc.get("gradient_checkpointing", True),
        learning_rate=float(tc.get("learning_rate", 1e-4)),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=tc.get("weight_decay", 0.01),
        max_grad_norm=tc.get("max_grad_norm", 1.0),
        logging_steps=tc.get("logging_steps", 20),
        save_steps=tc.get("save_steps", 200),
        save_total_limit=tc.get("save_total_limit", 3),
        eval_strategy="steps",
        eval_steps=tc.get("save_steps", 200),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=tc.get("bf16", True),
        fp16=tc.get("fp16", False),
        tf32=tc.get("tf32", False),
        dataloader_num_workers=tc.get("dataloader_num_workers", 4),
        dataloader_pin_memory=tc.get("dataloader_pin_memory", True),
        remove_unused_columns=False,
        report_to=report_to,
        # SFTConfig 固有パラメータ
        max_length=dc.get("max_seq_length", 1024),
        dataset_text_field="text",    # MaskedLabelCollator 使用時は無視される
        packing=False,
    )


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def train(config: dict, dry_run: bool = False, report_to: str | None = None) -> None:
    print_gpu_info()

    dc = config["data"]
    pc = config.get("prompt", {})

    train_file = dc.get("train_file", "data/synthetic/sft_train.jsonl")
    eval_file = dc.get("eval_file", "data/synthetic/sft_eval.jsonl")
    max_seq_length = dc.get("max_seq_length", 1024)
    eval_split_ratio = dc.get("eval_split_ratio", 0.05)
    system_message = pc.get("system_message", DEFAULT_SYSTEM_MESSAGE)

    train_path = Path(train_file)
    eval_path = Path(eval_file)

    if not train_path.exists():
        logger.error("Train file not found: %s", train_path)
        logger.error("Run 02_generate_qa.py first.")
        sys.exit(1)

    if dry_run:
        logger.info("[DRY RUN] Config loaded. Skipping training.")
        return

    # モデル・トークナイザー読み込み
    model, tokenizer = load_model_for_training(config)

    # SFTDataset: instruction 部分を -100 でマスクしたラベルを生成
    train_ds = SFTDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        system_message=system_message,
        mask_instruction=True,
    )

    if eval_path.exists():
        eval_ds = SFTDataset(
            data_path=eval_path,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            system_message=system_message,
            mask_instruction=True,
        )
    else:
        # ファイルがない場合は train から分割
        split = int(len(train_ds) * (1 - eval_split_ratio))
        eval_ds = torch.utils.data.Subset(train_ds, range(split, len(train_ds)))
        train_ds = torch.utils.data.Subset(train_ds, range(split))

    logger.info("Train samples: %d, Eval samples: %d", len(train_ds), len(eval_ds))

    # PyTorch Dataset → HuggingFace Dataset に変換
    # （TRL 1.5.0 は column_names 属性を要求するため）
    train_ds = _to_hf_dataset(train_ds)
    eval_ds = _to_hf_dataset(eval_ds)

    # ラベルマスクを保持するパディング collator
    data_collator = MaskedLabelCollator(pad_token_id=tokenizer.pad_token_id)

    sft_config = build_sft_config(config, report_to=report_to)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    output_dir = Path(config["training"]["output_dir"])
    adapter_save_path = output_dir / "final_adapter"
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    logger.info("Adapter saved to: %s", adapter_save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: CAE特化指示チューニング（SFT）")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/sft_config.yaml"),
    )
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, dry_run=args.dry_run, report_to=args.report_to)
