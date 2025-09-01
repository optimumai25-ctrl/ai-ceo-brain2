# file_parser.py
import os
from pathlib import Path
from docx import Document
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

DOCUMENTS_PATH = "downloaded_files"
PARSED_PATH = "parsed_data"

Path(PARSED_PATH).mkdir(parents=True, exist_ok=True)

def parse_docx(path: Path) -> str:
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_pdf(path: Path) -> str:
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def parse_xlsx(path: Path) -> str:
    text = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        text.append(f"### Sheet: {sheet} ###\n{df.to_string(index=False)}")
    return "\n\n".join(text)

def save_text(text: str, out_path: Path) -> None:
    out_path.write_text(text, encoding="utf-8")

def process_file(file_path: Path):
    ext = file_path.suffix.lower()
    try:
        if ext == ".docx":
            return parse_docx(file_path)
        elif ext == ".pdf":
            return parse_pdf(file_path)
        elif ext == ".xlsx":
            return parse_xlsx(file_path)
        else:
            return None
    except Exception as e:
        print(f"❌ Failed to parse {file_path.name}: {e}")
        return None

def main():
    root = Path(DOCUMENTS_PATH)
    if not root.exists():
        print(f"📁 '{root}' not found. Create it or run OneDrive sync.")
        return

    print(f"📁 Scanning folder: {root.resolve()}")
    for full_path in tqdm(list(root.rglob("*")), desc="Parsing"):
        if full_path.is_file() and full_path.suffix.lower() in {".pdf", ".docx", ".xlsx"}:
            rel = full_path.relative_to(root)
            print(f"📄 Processing: {rel}")
            text = process_file(full_path)
            if text:
                out_path = Path(PARSED_PATH) / f"{rel.stem}.txt"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_text(text, out_path)
                print(f"✅ Saved to {out_path}")
            else:
                print(f"⚠️ Skipped: {rel}")

if __name__ == "__main__":
    main()
