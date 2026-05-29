"""
00_convert_pkl.py - LangChain Document pickle → 学習用 JSONL 変換スクリプト

RAGプロジェクト等で収集した LangChain Document の .pkl ファイルを
CPT学習用 JSONL（data/processed/）と
合成Q&A生成用のチャンク JSONL（data/processed/qa_source.jsonl）に変換する。

前提: pickle の各要素が langchain_core.documents.base.Document であること
     （page_content: str, metadata: dict を持つ）

使用例:
    python scripts/00_convert_pkl.py \
        --pkl_path ../rag_project/data/crawled_documents.pkl \
        --output_dir data/processed

    # 別のpklも追加でマージする場合
    python scripts/00_convert_pkl.py \
        --pkl_path path/to/a.pkl path/to/b.pkl \
        --output_dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import re
import sys
import unicodedata
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# テキストクリーニング（01_preprocess.py と共通ロジック）
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Unicode正規化・連続空白・制御文字の除去を行う。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_valid_chunk(text: str, min_chars: int = 50) -> bool:
    """有効なチャンクかどうかを判定する。"""
    stripped = re.sub(r"[\s\W]", "", text)
    return len(stripped) >= min_chars


def split_into_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """段落優先のオーバーラップ付きチャンク分割。"""
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                current = current[-chunk_overlap:] + "\n\n" + para if chunk_overlap > 0 else para
            else:
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    chunks.append(para[i: i + chunk_size])
                current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if is_valid_chunk(c)]


# ---------------------------------------------------------------------------
# pkl 読み込み
# ---------------------------------------------------------------------------

def load_pkl(path: Path) -> list:
    """pickle ファイルを読み込む。"""
    logger.info("Loading pickle: %s", path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of Documents, got {type(data)}")
    logger.info("  %d documents loaded", len(data))
    return data


def extract_text(doc) -> tuple[str, str]:
    """
    LangChain Document から (テキスト, ソースURL) を抽出する。
    page_content 属性がない場合は __str__ にフォールバック。
    """
    if hasattr(doc, "page_content"):
        text = doc.page_content or ""
    elif isinstance(doc, dict):
        text = doc.get("page_content", doc.get("text", ""))
    else:
        text = str(doc)

    source = ""
    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        source = doc.metadata.get("source", doc.metadata.get("source_url", ""))

    return text, source


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def convert(
    pkl_paths: list[Path],
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    eval_ratio: float,
    seed: int,
    min_doc_chars: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    total_docs = 0
    skipped_docs = 0

    for pkl_path in pkl_paths:
        docs = load_pkl(pkl_path)
        for doc in tqdm(docs, desc=f"Chunking {pkl_path.name}"):
            text, source = extract_text(doc)
            if not text:
                skipped_docs += 1
                continue

            cleaned = normalize_text(text)

            # 短すぎるドキュメントはスキップ
            if len(re.sub(r"\s", "", cleaned)) < min_doc_chars:
                skipped_docs += 1
                continue

            chunks = split_into_chunks(cleaned, chunk_size, chunk_overlap)
            for chunk in chunks:
                all_chunks.append({"text": chunk, "source": source})
            total_docs += 1

    logger.info("Processed docs  : %d", total_docs)
    logger.info("Skipped docs    : %d", skipped_docs)
    logger.info("Total chunks    : %d", len(all_chunks))
    if all_chunks:
        avg_len = sum(len(c["text"]) for c in all_chunks) // len(all_chunks)
        logger.info("Avg chunk length: %d chars", avg_len)

    if not all_chunks:
        logger.error("No chunks generated. Check input files.")
        sys.exit(1)

    random.seed(seed)
    random.shuffle(all_chunks)

    split_idx = max(1, int(len(all_chunks) * (1 - eval_ratio)))
    train_chunks = all_chunks[:split_idx]
    eval_chunks = all_chunks[split_idx:]

    # CPT 用 JSONL（text列のみ）
    train_cpt = output_dir / "cpt_train.jsonl"
    eval_cpt = output_dir / "cpt_eval.jsonl"
    _write_jsonl(train_cpt, [{"text": c["text"]} for c in train_chunks])
    _write_jsonl(eval_cpt, [{"text": c["text"]} for c in eval_chunks])
    logger.info("CPT train -> %s (%d records)", train_cpt, len(train_chunks))
    logger.info("CPT eval  -> %s (%d records)", eval_cpt, len(eval_chunks))

    # Q&A生成のソース用 JSONL（source情報も保持）
    qa_source = output_dir / "qa_source.jsonl"
    _write_jsonl(qa_source, all_chunks)
    logger.info("QA source -> %s (%d records)", qa_source, len(all_chunks))
    logger.info("")
    logger.info("Next steps:")
    logger.info("  CPT  : python scripts/03_train_cpt.py --config configs/cpt_config.yaml")
    logger.info("  Q&A  : python scripts/02_generate_qa.py --input_dir data/processed")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangChain Document の pkl ファイルを学習用 JSONL に変換する"
    )
    parser.add_argument(
        "--pkl_path", type=Path, nargs="+", required=True,
        help="変換する .pkl ファイルのパス（複数指定可）"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/processed"),
        help="出力先ディレクトリ (default: data/processed)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512,
        help="チャンクあたりの最大文字数 (default: 512)"
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=64,
        help="チャンク間のオーバーラップ文字数 (default: 64)"
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=0.05,
        help="評価データの割合 (default: 0.05)"
    )
    parser.add_argument(
        "--min_doc_chars", type=int, default=100,
        help="ドキュメントとして採用する最小文字数（記号除く）(default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="ランダムシード (default: 42)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        pkl_paths=args.pkl_path,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        min_doc_chars=args.min_doc_chars,
    )
