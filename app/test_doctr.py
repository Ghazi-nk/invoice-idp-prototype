#!/usr/bin/env python3
"""
Test für die doctr OCR-Engine mit und ohne Bounding Box.
"""

import os
import sys
from pathlib import Path

# Get absolute path to the project root
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.document_digitalization.doctr_pdf2txt import doctr_pdf_to_text
from app.document_digitalization.pre_processing import standardize_ocr_output

def main():
    """Test für die doctr OCR-Engine."""
    sample_pdf = str(Path(project_root) / "resources" / "Inovices" / "BRE-01.pdf")
    
    if not os.path.exists(sample_pdf):
        print(f"Fehler: PDF nicht gefunden: {sample_pdf}")
        print(f"Prüfe Pfad: {os.path.dirname(sample_pdf)}")
        try:
            files = os.listdir(os.path.dirname(sample_pdf))
            print(f"Verfügbare Dateien: {files}")
        except Exception as e:
            print(f"Konnte Verzeichnis nicht lesen: {e}")
        return 1
    
    print(f"Verwende PDF: {sample_pdf}")
    
    # Test ohne Bounding Box
    print("\n=== Test doctr_pdf_to_text ohne Bounding Box ===")
    try:
        result = doctr_pdf_to_text(sample_pdf, include_bbox=False)
        print(f"Erfolg! {len(result)} Seiten extrahiert")
        
        if result and len(result) > 0:
            for i, page in enumerate(result):
                preview = page[:100] + "..." if len(page) > 100 else page
                print(f"Seite {i+1}: {preview}")
    except Exception as e:
        print(f"Fehler beim Test ohne Bounding Box: {e}")
        import traceback
        traceback.print_exc()
    
    # Test mit Bounding Box
    print("\n=== Test doctr_pdf_to_text mit Bounding Box ===")
    try:
        result_bbox = doctr_pdf_to_text(sample_pdf, include_bbox=True)
        print(f"Erfolg! {len(result_bbox)} Seiten extrahiert")
        
        if result_bbox and len(result_bbox) > 0:
            for i, page in enumerate(result_bbox):
                print(f"Seite {i+1}: {len(page)} Elemente")
                if page and len(page) > 0:
                    first_item = page[0]
                    print(f"Erstes Element: {first_item}")
                    
                    # Test standardize_ocr_output
                    print("\n=== Test standardize_ocr_output für Seite {i+1} ===")
                    standardized = standardize_ocr_output(page, format_type="formatted_string")
                    print(f"Standardisierung erfolgreich: {bool(standardized)}")
                    if isinstance(standardized, str):
                        preview = standardized[:100] + "..." if len(standardized) > 100 else standardized
                        print(f"Standardisiert: {preview}")
    except Exception as e:
        print(f"Fehler beim Test mit Bounding Box: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 