# file_parser.py

import os
from pathlib import Path
from docx import Document
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

DOCUMENTS_PATH = "downloaded_files"
PARSED_PATH = "parsed_data"

os.makedirs(PARSED_PATH, exist_ok=True)

def parse_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def parse_xlsx(path):
    text = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        text.append(f"### Sheet: {sheet} ###\n{df.to_string(index=False)}")
    return "\n\n".join(text)

def save_text(text, out_path):
    Path(out_path).write_text(text, encoding="utf-8")

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
    print(f"📁 Scanning folder: {DOCUMENTS_PATH}")
    for root, _, files in os.walk(DOCUMENTS_PATH):
        for file in files:
            full_path = Path(root) / file
            rel_path = full_path.relative_to(DOCUMENTS_PATH)
            print(f"📄 Processing: {rel_path}")
            text = process_file(full_path)
            if text:
                out_path = Path(PARSED_PATH) / f"{rel_path.stem}.txt"
                os.makedirs(out_path.parent, exist_ok=True)
                save_text(text, out_path)
                print(f"✅ Saved to {out_path}")
            else:
                print(f"⚠️ Skipped: {rel_path}")

if __name__ == "__main__":
    main()
