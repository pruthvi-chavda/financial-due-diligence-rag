from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from src.config import DOCLING_DO_OCR,DOCLING_DO_TABLE_STRUCTURE,DOCLING_FORCE_FULL_PAGE_OCR,PROCESSED_DATA_DIR,RAW_DATA_DIR,chunk_overlap,chunk_size

import html
import json
import re
import uuid
from pathlib import Path

from langchain_ollama import ChatOllama
from src.config import get_model_name
import time 
from langchain_groq import ChatGroq

CHUNKS_DATA_DIR = PROCESSED_DATA_DIR.parent / "chunks"


def pdf_to_markdown(pdf_path: Path) -> tuple[str, list[str]]:
    out_dir = PROCESSED_DATA_DIR / pdf_path.stem
    uncleaned_file = out_dir / "text_uncleaned.md"

    if uncleaned_file.exists():
        print(f"Cache found. Loading parsed markdown from {out_dir}")
        full_markdown = uncleaned_file.read_text(encoding="utf-8")
        tables_md = [
            p.read_text(encoding="utf-8")
            for p in sorted(out_dir.glob("table_*.md"))
        ]
        return full_markdown, tables_md

    pipeline_options = PdfPipelineOptions(
        do_ocr=DOCLING_DO_OCR,
        do_table_structure=DOCLING_DO_TABLE_STRUCTURE,
        ocr_options=RapidOcrOptions(force_full_page_ocr=DOCLING_FORCE_FULL_PAGE_OCR),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(str(pdf_path))
    doc = result.document

    full_markdown = doc.export_to_markdown()
    tables_md = [table.export_to_markdown(doc=doc) for table in doc.tables]
    return full_markdown, tables_md


def export_uncleaned_md(full_markdown: str, tables_md: list[str], out_dir: Path) -> None:
    if (out_dir / "text_uncleaned.md").exists():
        print("Uncleaned cache already exists. Skipping export.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "text_uncleaned.md").write_text(full_markdown, encoding="utf-8")

    for i, table in enumerate(tables_md, start=1):
        (out_dir / f"table_{i}.md").write_text(table, encoding="utf-8")


def clean_markdown(full_markdown: str) -> str:
    text = html.unescape(full_markdown)
    text = re.sub(r"<!--\s*image\s*-->", "", text)
    text = re.sub(r"(?:^\|.*\|$\n?)+", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunking(text: str, source_file: str) -> list[Document]:
    headers_to_split_on = [("#", "Header 1"),("##", "Header 2"),("###", "Header 3")]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_splits = markdown_splitter.split_text(text)

    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\n\n", "\n", ". ", " ", ""])
    chunks = recursive_splitter.split_documents(header_splits)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "source_file": source_file,
            "chunk_type": "text",
            "chunk_index": i,
            "chunk_id": str(uuid.uuid4()),
        })

    return chunks


def export_chunks(chunks: list[Document], pdf_stem: str) -> Path:
    CHUNKS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CHUNKS_DATA_DIR / f"{pdf_stem}_chunks.json"

    serialized = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    out_file.write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_file



def summarize_tables(tables_md: list[str], source_file: str) -> list[Document]:
    fast_model = get_model_name(role="fast", limit_reached=False)
    llm = ChatGroq(model=fast_model, temperature=0)

    summary_prompt = """Summarize this financial table in 2-3 clear sentences.
    Include the main metrics, time periods, units/currency, and key trends so a user searching in plain English can find it.
    Do not add any preamble.
    
    Table:
    {table_md}"""

    table_docs: list[Document] = []

    for i, table in enumerate(tables_md, start=1):
        print(f"Summarizing table {i}/{len(tables_md)}...")
        table_markdown = table.strip()
        if not table_markdown:
            continue
        compact_table = re.sub(r" {2,}", " ", table_markdown)
        summary = llm.invoke(summary_prompt.format(table_md=compact_table)).content.strip()
        time.sleep(5)

        doc = Document(
            page_content=f"Table Summary: {summary}\n\n{table_markdown}",
            metadata={
                "source_file": source_file,
                "chunk_type": "table",
                "table_index": i,
                "chunk_id": str(uuid.uuid4()),
            },
        )
        table_docs.append(doc)

    return table_docs

def ingest_pdf(pdf_path: Path) -> list[Document]:
    out_dir = PROCESSED_DATA_DIR / pdf_path.stem

    full_md, tables = pdf_to_markdown(pdf_path)
    export_uncleaned_md(full_md, tables, out_dir)

    cleaned_md = clean_markdown(full_md)
    (out_dir / "text_cleaned.md").write_text(cleaned_md, encoding="utf-8")

    text_chunks = chunking(cleaned_md, pdf_path.name)
    table_chunks = summarize_tables(tables, pdf_path.name)

    all_docs = text_chunks + table_chunks
    export_chunks(all_docs, pdf_path.stem)
    return all_docs


if __name__ == "__main__":
    tcs_pdf = RAW_DATA_DIR / "tcs_fy26_financials.pdf"
    all_docs = ingest_pdf(tcs_pdf)
    print(f"Total chunks (text + tables): {len(all_docs)}")