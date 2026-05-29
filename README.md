# cae-llm: CAE特化生成モデル

llm-jp-3-1.8b ベースの CAE（Computer-Aided Engineering）専用チャットアシスタント。  
QLoRA を使い 8GB VRAM 環境で **CPT（継続事前学習）→ SFT（指示チューニング）** の 2 段階学習を実施する。

## 対応タスク

- 有限要素解析（FEA: OptiStruct, Abaqus, ANSYS, LS-DYNA 等）に関する Q&A
- 流体解析（CFD: OpenFOAM, Fluent 等）パラメータ相談
- 構造・熱・振動解析の手法説明
- 境界条件・材料定義・ソルバー設定のアドバイス

## 学習結果サマリー

| フェーズ | データ | 学習時間 | 最終 loss |
|---|---|---|---|
| CPT（継続事前学習） | Altair OptiStruct ドキュメント（pkl 変換）45k サンプル | 約 8 時間 | 0.21 |
| SFT（指示チューニング） | 合成 Q&A JSONL | 約 50 分 | 0.06 |

- ベースモデル: `llm-jp/llm-jp-3-1.8b`
- GPU: NVIDIA RTX 5060 (8 GB VRAM)
- LoRA rank: 16 / QLoRA 4-bit NF4 / bfloat16 + TF32

## 必要環境

- Python 3.10+
- CUDA 対応 GPU（VRAM 8 GB 以上推奨）
- CUDA 11.8 / 12.1
- 主要ライブラリ: `transformers>=5.0`, `trl>=1.5.0`, `peft`, `bitsandbytes`

## セットアップ

```bash
pip install -r requirements.txt
```

OpenAI API を使う場合（合成 Q&A 生成）:

```bash
cp .env.example .env
# OPENAI_API_KEY=sk-... を記入
```

### Windows 環境での注意（TRL UTF-8 パッチ）

Windows 環境では TRL ライブラリ内の `.jinja` テンプレート読み込みがデフォルト CP932 エンコーディングで失敗することがあります。  
その場合は以下のワンライナーで TRL ソースを修正してください。

```python
import pathlib, site, re
for sp in site.getsitepackages():
    p = pathlib.Path(sp) / "trl" / "chat_template_utils.py"
    if p.exists():
        p.write_text(re.sub(r"\.read_text\(\)", ".read_text(encoding='utf-8')", p.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"Patched: {p}")
```

## 学習フロー

### 0. pkl データの変換（RAG プロジェクトのデータを使う場合）

LangChain Document として収集済みの `.pkl` ファイルをそのまま変換できます。

```bash
python scripts/00_convert_pkl.py \
  --pkl_path ../rag_project/data/crawled_documents.pkl \
  --output_dir data/processed
```

`data/processed/cpt_train.jsonl`, `cpt_eval.jsonl`, `qa_source.jsonl` が生成されます。  
この場合は手順 1 をスキップして手順 2 から開始できます。

### 1. データ前処理（raw テキスト・HTML から変換する場合）

`data/raw/` に `.txt` / `.md` / `.html` / `.htm` ファイルを置いて実行します。

```bash
python scripts/01_preprocess.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --chunk_size 512 \
  --chunk_overlap 64
```

HTML ファイルはナビゲーション・スクリプト等を自動除去し、本文テキストのみ抽出します。  
Altair OptiStruct などの技術ドキュメント HTML をそのまま投入できます。

### 2. 合成 Q&A 生成（SFT 用データ）

```bash
python scripts/02_generate_qa.py \
  --input_file data/processed/qa_source.jsonl \
  --output_dir data/synthetic \
  --model gpt-4o-mini \
  --pairs_per_chunk 3
```

### 3. Phase 1: 継続事前学習（CPT）

ドメインテキストで next-token-prediction を実施しドメイン適応する。  
データが 10 MB 以上ある場合に推奨。スキップして Phase 2 のみでも動作する。

```bash
python scripts/03_train_cpt.py --config configs/cpt_config.yaml
```

アダプターは `outputs/cpt/final_adapter/` に保存されます。

### 4. Phase 2: 指示チューニング（SFT）

```bash
python scripts/04_train_sft.py --config configs/sft_config.yaml
```

`sft_config.yaml` の `model.adapter_path` に CPT アダプターのパスが設定されており、  
CPT → SFT の 2 段階学習が自動的に行われます。  
アダプターは `outputs/sft/final_adapter/` に保存されます。

### 5. 推論・デモ

```bash
# CLI 対話モード
python scripts/05_inference.py \
  --adapter_path outputs/sft/final_adapter \
  --mode cli

# Streamlit UI
streamlit run scripts/05_inference.py -- \
  --adapter_path outputs/sft/final_adapter \
  --mode ui

# 単発クエリ（スクリプト組み込みや動作確認に便利）
python scripts/05_inference.py \
  --adapter_path outputs/sft/final_adapter \
  --mode single \
  --query "等方性材料の弾性係数と降伏応力の定義方法を教えてください"
```

## プロジェクト構成

```
cae_llm/
├── data/
│   ├── raw/            # 入力 CAE テキスト（.txt / .md / .html）
│   ├── processed/      # チャンク済み JSONL（CPT 用）
│   └── synthetic/      # 合成 Q&A JSONL（SFT 用）
├── src/
│   ├── dataset.py      # CPTDataset / SFTDataset / HF Dataset ヘルパー
│   ├── model.py        # QLoRA モデルロード・LoRA アダプター設定
│   └── utils.py        # 設定ロード・プロンプトテンプレート・GPU 情報
├── scripts/
│   ├── 00_convert_pkl.py   # LangChain Document pkl → JSONL 変換
│   ├── 01_preprocess.py    # テキスト・HTML 前処理
│   ├── 02_generate_qa.py   # 合成 Q&A 生成（OpenAI API）
│   ├── 03_train_cpt.py     # Phase 1: 継続事前学習
│   ├── 04_train_sft.py     # Phase 2: 指示チューニング
│   └── 05_inference.py     # CLI / Streamlit UI 推論
├── configs/
│   ├── cpt_config.yaml     # CPT 学習設定
│   └── sft_config.yaml     # SFT 学習設定
├── outputs/            # チェックポイント・アダプター保存先（.gitignore 対象）
├── .env.example
├── requirements.txt
└── README.md
```

## VRAM 最適化設定

| 設定 | 値 | 備考 |
|---|---|---|
| 量子化 | 4-bit NF4 (QLoRA) | `bitsandbytes` |
| LoRA rank | 16 | alpha=32 |
| バッチサイズ | 1 | batch=2 は 8GB で VRAM 溢れ |
| gradient_accumulation | 8 | 実効バッチ = 8 |
| gradient_checkpointing | 有効 | VRAM 節約（速度は若干低下） |
| 混合精度 | bfloat16 + TF32 | RTX 5060 (Blackwell) 最適化 |
| シーケンス長 | 512 (CPT) / 1024 (SFT) | CPT は 512 で十分 |
| max_train_samples | 45,000 (CPT) | 約 8 時間の学習量 |
| eval_steps | 1,000 | eval 時間を 15 分 → 90 秒に短縮 |

## ライブラリバージョン互換性メモ

| ライブラリ | 変更点と対応 |
|---|---|
| `transformers >= 5.x` | `Trainer` / `SFTTrainer` の `tokenizer=` → `processing_class=` に変更 |
| `transformers >= 5.x` | `warmup_ratio` 削除 → `warmup_steps` を明示的に指定 |
| `trl >= 1.5.0` | `DataCollatorForCompletionOnlyLM` 削除 → `MaskedLabelCollator`（独自実装）で代替 |
| `trl >= 1.5.0` | `SFTTrainer` が `column_names` 属性を要求 → PyTorch Dataset を HF Dataset に変換 |

## ライセンス

ベースモデル: [llm-jp/llm-jp-3-1.8b](https://huggingface.co/llm-jp/llm-jp-3-1.8b) のライセンスに従う。
