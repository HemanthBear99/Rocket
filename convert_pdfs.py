"""
PDF to Text Converter for RLV Specification Documents
"""
import PyPDF2
import os

def extract_pdf_text(pdf_path, output_path):
    """Extract text from PDF and save to file."""
    print(f"Processing: {pdf_path}")
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"  Total pages: {len(reader.pages)}")
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"\n\n--- PAGE {i+1} ---\n\n"
                text += page_text if page_text else "[No extractable text on this page]"
            
        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(text)
        
        print(f"  Saved to: {output_path}")
        print(f"  Characters extracted: {len(text)}")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

# Convert all PDFs
pdfs = [
    ("d:/T-client/Developer_Implementation (2).pdf", "d:/T-client/Developer_Implementation.txt"),
    ("d:/T-client/RLV__Developer (3).pdf", "d:/T-client/RLV_Developer_3.txt"),
    ("d:/T-client/RLV__Developer (10).pdf", "d:/T-client/RLV_Developer_10.txt"),
]

print("="*60)
print("Converting PDFs to Text")
print("="*60)

for pdf_path, txt_path in pdfs:
    if os.path.exists(pdf_path):
        extract_pdf_text(pdf_path, txt_path)
    else:
        print(f"File not found: {pdf_path}")

print("\nDone!")
