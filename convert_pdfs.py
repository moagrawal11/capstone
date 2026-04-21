import os
from pypdf import PdfReader

# Mapping: source PDF -> target txt path
PDF_MAPPING = {
    "sample-data/robot_catalog.pdf": "dataset/products/robot_catalog.txt",
    "sample-data/axiom_faq.pdf": "dataset/faq/axiom_faq.txt",
    "sample-data/troubleshooting_guide.pdf": "dataset/support/troubleshooting_guide.txt",
    "sample-data/warranty_policy.pdf": "dataset/policies/warranty_and_support_policy.txt",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def convert_pdf_to_txt(pdf_path: str, txt_path: str) -> None:
    abs_pdf = os.path.join(BASE_DIR, pdf_path)
    abs_txt = os.path.join(BASE_DIR, txt_path)

    os.makedirs(os.path.dirname(abs_txt), exist_ok=True)

    reader = PdfReader(abs_pdf)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    full_text = "\n\n".join(text_parts)

    with open(abs_txt, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"  [OK] {pdf_path} -> {txt_path} ({len(full_text):,} chars, {len(reader.pages)} pages)")


def main() -> None:
    """Convert all sample PDFs to text files."""
    print("Converting sample PDFs to dataset .txt files...")
    print()

    for pdf_path, txt_path in PDF_MAPPING.items():
        convert_pdf_to_txt(pdf_path, txt_path)

    print()
    print(f"Done! Converted {len(PDF_MAPPING)} files.")


if __name__ == "__main__":
    main()
