"""
01_preprocess.py - CAEテキスト前処理スクリプト

data/raw/ 以下のテキスト・HTMLファイルを読み込み、
オーバーラップ付きチャンク分割でJSONLに変換する。

対応形式: .txt / .md / .rst / .html / .htm

Altair OptiStruct などのHTMLドキュメントを data/raw/ に置くことで、
そのままCAEドメイン学習データとして活用できる。

出力: data/processed/cpt_train.jsonl, cpt_eval.jsonl

使用例:
    python scripts/01_preprocess.py \
        --input_dir data/raw \
        --output_dir data/processed \
        --chunk_size 512 \
        --chunk_overlap 64 \
        --eval_ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# テキストクリーニング
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Unicode正規化・連続空白・制御文字の除去を行う。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_valid_chunk(text: str, min_chars: int = 50) -> bool:
    """有効なチャンクかどうかを判定する（短すぎる・記号のみを除外）。"""
    stripped = re.sub(r"[\s\W]", "", text)
    return len(stripped) >= min_chars


# ---------------------------------------------------------------------------
# チャンク分割（文字数ベース、オーバーラップ付き）
# ---------------------------------------------------------------------------

def split_into_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """
    テキストをオーバーラップ付きで chunk_size 文字ずつ分割する。

    改行優先で分割して文が途中で切れるのを防ぐ。
    """
    # 段落・文単位で分割してから結合
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # 段落を追加してもチャンクサイズ内に収まるか確認
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # 現在のバッファを保存してリセット
            if current:
                chunks.append(current)
                # オーバーラップ: 前のチャンク末尾を次の先頭に引き継ぐ
                current = current[-chunk_overlap:] + "\n\n" + para if chunk_overlap > 0 else para
            else:
                # 段落そのものが chunk_size を超える場合は文字数で強制分割
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    chunks.append(para[i: i + chunk_size])
                current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if is_valid_chunk(c)]


# ---------------------------------------------------------------------------
# ファイル読み込み
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".html", ".htm"}

# OptiStruct ドキュメント特有の除去対象セレクタ
_HTML_NOISE_SELECTORS = [
    "script", "style", "nav", "header", "footer",
    "noscript", "iframe", "aside",
    # Altair HelpCenter のナビゲーション要素
    "[class*='nav']", "[class*='menu']", "[class*='breadcrumb']",
    "[class*='toc']", "[id*='toc']", "[id*='nav']",
    "[class*='footer']", "[class*='header']",
]

# セクション見出しタグ（テキスト抽出時に改行を挿入）
_BLOCK_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "li", "td", "th", "dt", "dd",
    "blockquote", "pre", "code",
    "div", "section", "article",
}


def extract_text_from_html(raw_html: str) -> str:
    """
    HTMLからナビゲーション・スクリプト等を除去してプレーンテキストを抽出する。

    Altair OptiStruct などの技術ドキュメントHTMLに最適化されており、
    見出し・本文・テーブルのテキストを構造を保ちつつ抽出する。
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 がインストールされていません。pip install beautifulsoup4 lxml を実行してください。")
        return ""

    soup = BeautifulSoup(raw_html, "lxml")

    # ノイズ要素を除去
    for selector in _HTML_NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # <br> を改行に変換
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # ブロック要素の前後に改行を挿入して段落を保持
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text(separator=" ")

    # 連続する空白・改行を整理
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_file(path: Path) -> str:
    """拡張子に応じてテキストまたはHTMLを読み込み、プレーンテキストを返す。"""
    raw = _read_raw(path)
    if not raw:
        return ""
    if path.suffix.lower() in {".html", ".htm"}:
        return extract_text_from_html(raw)
    return raw


def _read_raw(path: Path) -> str:
    """バイトを読んでデコードする。"""
    for encoding in ("utf-8", "utf-8-sig", "cp932", "shift_jis"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    logger.warning("Could not decode: %s — skipping", path)
    return ""


def collect_files(input_dir: Path) -> list[Path]:
    """input_dir 以下の対応拡張子ファイルを再帰収集する。"""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(files)


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def process_directory(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    eval_ratio: float,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = collect_files(input_dir)
    if not files:
        logger.warning("No supported files found in: %s", input_dir)
        return

    all_chunks: list[str] = []
    for path in tqdm(files, desc="Processing files"):
        raw = read_file(path)
        if not raw:
            continue
        cleaned = normalize_text(raw)
        chunks = split_into_chunks(cleaned, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        logger.info("  %s -> %d chunks", path.name, len(chunks))

    logger.info("Total chunks: %d", len(all_chunks))

    random.seed(seed)
    random.shuffle(all_chunks)

    split_idx = max(1, int(len(all_chunks) * (1 - eval_ratio)))
    train_chunks = all_chunks[:split_idx]
    eval_chunks = all_chunks[split_idx:]

    train_path = output_dir / "cpt_train.jsonl"
    eval_path = output_dir / "cpt_eval.jsonl"

    _write_jsonl(train_path, [{"text": c} for c in train_chunks])
    _write_jsonl(eval_path, [{"text": c} for c in eval_chunks])

    logger.info("Saved train: %s (%d records)", train_path, len(train_chunks))
    logger.info("Saved eval : %s (%d records)", eval_path, len(eval_chunks))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAEテキストファイルをチャンク分割してJSONLに変換する"
    )
    parser.add_argument(
        "--input_dir", type=Path, default=Path("data/raw"),
        help="入力テキストファイルのディレクトリ (default: data/raw)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/processed"),
        help="出力JSONLのディレクトリ (default: data/processed)"
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
        "--seed", type=int, default=42,
        help="ランダムシード (default: 42)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
