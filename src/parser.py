import io
import re
from typing import Union
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize extracted text by stripping non-standard whitespace and control chars."""
    if not text:
        return ""
    # Replace multiple spaces/newlines with single space/newline
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(source: Union[str, Path, io.BytesIO, bytes]) -> str:
    """Extract readable text from a PDF source using pypdf."""
    from pypdf import PdfReader

    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    reader = PdfReader(stream)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return clean_text("\n".join(text_parts))


def extract_text_from_docx(source: Union[str, Path, io.BytesIO, bytes]) -> str:
    """Extract readable text from a DOCX source using python-docx."""
    import docx

    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    doc = docx.Document(stream)
    text_parts = [para.text for para in doc.paragraphs if para.text]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text:
                    text_parts.append(cell.text)
    return clean_text("\n".join(text_parts))


def extract_text(source: Union[str, Path, io.BytesIO, bytes], filename: str = "") -> str:
    """
    Universal extractor supporting .txt, .pdf, .docx, or raw strings.
    """
    if isinstance(source, str) and not filename:
        path = Path(source)
        if path.is_file():
            filename = path.name
            with open(path, "rb") as f:
                content = f.read()
            return extract_text(content, filename)
        else:
            # Assume raw text string
            return clean_text(source)

    fname = filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(source)
    elif fname.endswith(".docx"):
        return extract_text_from_docx(source)
    else:
        # Default text decode
        if isinstance(source, (bytes, bytearray)):
            try:
                return clean_text(source.decode("utf-8"))
            except UnicodeDecodeError:
                return clean_text(source.decode("latin-1", errors="ignore"))
        elif isinstance(source, io.BytesIO):
            return clean_text(source.getvalue().decode("utf-8", errors="ignore"))
        return clean_text(str(source))
