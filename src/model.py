"""
QLoRAモデルロード・LoRAアダプター設定ユーティリティ。

llm-jp-3-1.8b を 4-bit QLoRA で読み込み、PEFT LoRA アダプターを
アタッチして返す。CPT / SFT 両フェーズで共通利用する。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def build_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """4-bit量子化設定を構築する。"""
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    compute_dtype = dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
    target_modules: Optional[list[str]] = None,
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """LoRA設定を構築する。"""
    if target_modules is None:
        # llm-jp-3 (Mistral/LLaMA 系アーキテクチャ) のデフォルト対象
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type=TaskType[task_type],
    )


def load_base_model(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
    trust_remote_code: bool = True,
    gradient_checkpointing: bool = True,
) -> AutoModelForCausalLM:
    """BnB量子化でベースモデルを読み込む。"""
    logger.info("Loading base model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """トークナイザーを読み込む。padding_sideをleftに設定。"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_peft_model(
    base_model: AutoModelForCausalLM,
    lora_config: LoraConfig,
) -> AutoModelForCausalLM:
    """LoRAアダプターをベースモデルにアタッチする。"""
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_for_training(config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    設定 dict からモデルとトークナイザーを構築して返す。

    CPT/SFT スクリプトから呼ぶエントリポイント。
    config には model.* / lora.* / training.gradient_checkpointing が必要。
    """
    mc = config["model"]
    lc = config["lora"]
    use_gc = config.get("training", {}).get("gradient_checkpointing", True)

    bnb_config = build_bnb_config(
        load_in_4bit=mc.get("load_in_4bit", True),
        bnb_4bit_quant_type=mc.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=mc.get("bnb_4bit_compute_dtype", "bfloat16"),
        bnb_4bit_use_double_quant=mc.get("bnb_4bit_use_double_quant", True),
    )
    lora_config = build_lora_config(
        r=lc.get("r", 16),
        lora_alpha=lc.get("lora_alpha", 32),
        lora_dropout=lc.get("lora_dropout", 0.05),
        bias=lc.get("bias", "none"),
        target_modules=lc.get("target_modules", None),
        task_type=lc.get("task_type", "CAUSAL_LM"),
    )
    base_model = load_base_model(
        model_name=mc["name"],
        bnb_config=bnb_config,
        trust_remote_code=mc.get("trust_remote_code", True),
        gradient_checkpointing=use_gc,
    )

    # CPT フェーズのアダプターがある場合はロードしてから新しい LoRA を追加
    adapter_path = mc.get("adapter_path")
    if adapter_path and Path(adapter_path).exists():
        logger.info("Loading CPT adapter from: %s", adapter_path)
        base_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    else:
        base_model = build_peft_model(base_model, lora_config)

    tokenizer = load_tokenizer(mc["name"], trust_remote_code=mc.get("trust_remote_code", True))
    return base_model, tokenizer


def load_model_for_inference(
    model_name: str,
    adapter_path: str,
    load_in_4bit: bool = True,
    trust_remote_code: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """推論用モデルロード（LoRAマージ済み）。"""
    bnb_config = build_bnb_config(load_in_4bit=load_in_4bit)
    tokenizer = load_tokenizer(model_name, trust_remote_code)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer
