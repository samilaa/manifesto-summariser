from pathlib import Path
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from data.data_utils import from_pdf_to_string, chunk_text, save_data

def main():
    # Get the manifesto text
    pdf_path = Path(__file__).resolve().parents[1] / "data/manifestos/keskusta/keskusta-eduskuntavaaliohjelma-2023.pdf"
    manifesto_text = from_pdf_to_string(pdf_path)

    # Write to intermediate text file
    text_path = Path(__file__).resolve().parents[1] / "data/manifestos/keskusta/keskusta-eduskuntavaaliohjelma-2023txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(manifesto_text)

    # Process the text into chunks
    chunks = chunk_text(
        str(text_path),
        n_words=100,
        party_name="Keskusta"
    )

    # Save the chunked output
    output_path = Path(__file__).resolve().parents[1] / "data/processed/keskusta_chunks.json"
    save_data(chunks, str(output_path))

main()
