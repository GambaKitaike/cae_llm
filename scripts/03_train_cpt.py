"""
03_train_cpt.py - Phase 1: 継続事前学習（Continued Pre-Training）スクリプト

llm-jp-3-1.8b を 4-bit QLoRA で読み込み、CAEドメインテキストで
next-token-prediction を実施してドメイン適応を行う。

使用例:
    python scripts/03_train_cpt.py --config configs/cpt_config.yaml

    # W&B トラッキング有効
    python scripts/03_train_cpt.py \
        --config configs/cpt_config.yaml \
        --report_to wandb

    # GPU確認のみ（学習なし）
    python scripts/03_train_cpt.py --config configs/cpt_config.yaml --dry_run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# src/ を Python パスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model_for_training
from src.dataset import load_hf_dataset_for_cpt
from src.utils import load_config, print_gpu_info

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 学習設定構築
# ---------------------------------------------------------------------------

def build_training_args(config: dict, report_to: str | None = None) -> TrainingArguments:
    tc = config["training"]
    output_dir = tc["output_dir"]

    if report_to is None:
        report_to = tc.get("report_to", "none")

    # warmup_ratio → warmup_steps に変換（transformers 5.x で warmup_ratio 削除）
    num_epochs = tc.get("num_train_epochs", 3)
    grad_accum = tc.get("gradient_accumulation_steps", 16)
    # steps_per_epoch は概算（データ数不明のためデフォルト500 stepsを想定）
    warmup_ratio = tc.get("warmup_ratio", 0.05)
    # 学習全体の step 数が不明なためデフォルト 100 steps で warmup
    warmup_steps = tc.get("warmup_steps", max(10, int(warmup_ratio * 2000)))

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=tc.get("gradient_checkpointing", True),
        learning_rate=float(tc.get("learning_rate", 2e-4)),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=tc.get("weight_decay", 0.01),
        max_grad_norm=tc.get("max_grad_norm", 1.0),
        logging_steps=tc.get("logging_steps", 50),
        save_steps=tc.get("save_steps", 500),
        save_total_limit=tc.get("save_total_limit", 3),
        eval_strategy="steps",
        eval_steps=tc.get("save_steps", 500),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=tc.get("bf16", True),
        fp16=tc.get("fp16", False),
        tf32=tc.get("tf32", False),
        dataloader_num_workers=tc.get("dataloader_num_workers", 4),
        dataloader_pin_memory=tc.get("dataloader_pin_memory", True),
        remove_unused_columns=tc.get("remove_unused_columns", False),
        report_to=report_to,
    )


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def train(config: dict, dry_run: bool = False, report_to: str | None = None) -> None:
    print_gpu_info()

    dc = config["data"]
    train_file = dc.get("train_file", "data/processed/cpt_train.jsonl")
    eval_file = dc.get("eval_file", "data/processed/cpt_eval.jsonl")
    max_seq_length = dc.get("max_seq_length", 512)
    text_column = dc.get("text_column", "text")
    eval_split_ratio = dc.get("eval_split_ratio", 0.05)
    max_train_samples = dc.get("max_train_samples", None)
    max_eval_samples = dc.get("max_eval_samples", None)

    train_path = Path(train_file)
    eval_path = Path(eval_file)

    if not train_path.exists():
        logger.error("Train file not found: %s", train_path)
        logger.error("Run 01_preprocess.py first.")
        sys.exit(1)

    # データセット読み込み
    if eval_path.exists():
        train_ds, _ = load_hf_dataset_for_cpt(train_path, text_column, eval_split_ratio=0.0)
        eval_ds, _ = load_hf_dataset_for_cpt(eval_path, text_column, eval_split_ratio=0.0)
    else:
        logger.info("No separate eval file found. Splitting from train data.")
        train_ds, eval_ds = load_hf_dataset_for_cpt(
            train_path, text_column, eval_split_ratio=eval_split_ratio
        )

    if max_train_samples and max_train_samples < len(train_ds):
        train_ds = train_ds.select(range(max_train_samples))
        logger.info("Limiting train to %d samples", max_train_samples)

    if max_eval_samples and max_eval_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(max_eval_samples))
        logger.info("Limiting eval to %d samples", max_eval_samples)

    logger.info("Train samples: %d, Eval samples: %d", len(train_ds), len(eval_ds))

    if dry_run:
        logger.info("[DRY RUN] Dataset loaded successfully. Skipping training.")
        return

    # モデル・トークナイザー読み込み
    model, tokenizer = load_model_for_training(config)

    # テキストをトークナイズ
    def tokenize_fn(batch):
        return tokenizer(
            batch[text_column],
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=[text_column])
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=[text_column])

    # DataCollator（MLM=False で CLM）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    training_args = build_training_args(config, report_to=report_to)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting CPT training...")
    trainer.train()

    # アダプター保存
    output_dir = Path(config["training"]["output_dir"])
    adapter_save_path = output_dir / "final_adapter"
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    logger.info("Adapter saved to: %s", adapter_save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: CAEドメイン継続事前学習"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cpt_config.yaml"),
        help="設定ファイルのパス (default: configs/cpt_config.yaml)"
    )
    parser.add_argument(
        "--report_to", type=str, default=None,
        help="実験トラッキング先: wandb / none (config.yamlを上書き)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="データ読み込みのみ実行し、学習はスキップする"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, dry_run=args.dry_run, report_to=args.report_to)
