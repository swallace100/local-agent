"""Builds small PDF and Word files for tests and fixtures, with no extra libraries.

Run it to rewrite the committed fixtures:
    python tests/make_docs.py
"""

import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent


def make_pdf(pages: list[list[str]]) -> bytes:
    """Return a PDF with one page per list of lines, in plain Helvetica."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    page_numbers = []
    for lines in pages:
        text = " T* ".join(
            "({}) Tj".format(line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")) for line in lines
        )
        stream = f"BT /F1 12 Tf 14 TL 72 720 Td {text} ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>"
        )
        page_numbers.append(len(objects))
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>"

    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    output += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(output)


def make_docx(paragraphs: list[str]) -> bytes:
    """Return a Word document with one paragraph per string."""
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>" for text in paragraphs)
    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>",
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>",
        "word/document.xml": f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{namespace}"><w:body>{body}</w:body></w:document>',
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


CONTRACT_PAGES = [
    [
        "Kestrel maintenance contract",
        "This agreement covers the Kestrel weather stations at all four sites.",
        "The contract renews on 14 March each year unless cancelled 60 days before.",
        "The annual fee is 18,400 euros, invoiced in January.",
    ],
    [
        "Contacts and response times",
        "The vendor contact is Ilse Brandt, reachable on extension 3391.",
        "Faults reported before noon are attended the same working day.",
    ],
]
ONBOARDING_PARAGRAPHS = [
    "Osprey team onboarding",
    "New members of the Osprey team receive a laptop of model TX-880 on their first day.",
    "Badge requests go to room 4.17, and take two working days.",
    "The team's weekly planning meeting is on Wednesday at 09:30.",
]

def parts_catalog() -> str:
    """A long list of near-identical lines, where only the exact part number tells them apart."""
    items = ["gasket", "bearing", "valve seat", "drive belt", "filter cartridge", "pressure sensor", "coupling", "seal kit"]
    lines = ["# Spare parts catalog", "", "Where each spare part is stored in the workshop.", ""]
    for index in range(300):
        number = 1000 + (index * 7919) % 9000
        shelf = f"{'ABCDEFGH'[(index * 5) % 8]}{1 + (index * 3) % 24}"
        lines.append(f"- Part KX-{number}: {items[index % len(items)]}, stored on shelf {shelf}, reorder at {2 + index % 9} left.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    knowledge = ROOT / "evals" / "fixtures" / "knowledge"
    (knowledge / "parts-catalog.md").write_text(parts_catalog(), encoding="utf-8", newline="\n")
    web = ROOT / "tools" / "web-search" / "fixtures"
    web.mkdir(exist_ok=True)
    for folder in (knowledge, web):
        (folder / "kestrel-contract.pdf").write_bytes(make_pdf(CONTRACT_PAGES))
        (folder / "osprey-onboarding.docx").write_bytes(make_docx(ONBOARDING_PARAGRAPHS))
    print("Wrote the PDF and Word fixtures.")
