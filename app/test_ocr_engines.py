#!/usr/bin/env python3
"""
Einfache Testdatei, um die OCR-Engines zu überprüfen.
"""

import os
import sys
from pathlib import Path

# Get absolute path to the project root
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.document_processing import ocr_pdf, get_available_engines
from app.document_digitalization.pre_processing import standardize_ocr_output

def test_engine(engine_name, pdf_path):
    """Test eine OCR-Engine mit und ohne Bounding Boxes."""
    print(f"\n== Teste OCR-Engine: {engine_name} ==")
    
    # Ohne Bounding Boxes
    print(f"- Ohne Bounding Boxes:")
    try:
        result = ocr_pdf(pdf_path, engine=engine_name, include_bbox=False)
        print(f"  Status: Erfolgreich - {len(result)} Seiten extrahiert")
        if result and len(result) > 0:
            sample = result[0]
            preview = sample[:100] + "..." if isinstance(sample, str) and len(sample) > 100 else sample
            print(f"  Beispiel: {preview}")
    except Exception as e:
        print(f"  Status: FEHLER - {str(e)}")
    
    # Mit Bounding Boxes
    print(f"- Mit Bounding Boxes:")
    try:
        result = ocr_pdf(pdf_path, engine=engine_name, include_bbox=True)
        print(f"  Status: Erfolgreich - {len(result)} Seiten extrahiert")
        if result and len(result) > 0:
            sample_page = result[0]
            if isinstance(sample_page, list) and len(sample_page) > 0:
                if isinstance(sample_page[0], dict):
                    print(f"  Format: Liste von Wörterbüchern mit {len(sample_page)} Elementen")
                    sample_item = sample_page[0]
                    print(f"  Beispiel: text='{sample_item.get('text', '')[:30]}', bbox={sample_item.get('bbox', [])}")
                elif isinstance(sample_page[0], list) and len(sample_page[0]) > 0:
                    print(f"  Format: Verschachtelte Liste mit {len(sample_page)} Elementen")
                    sample_item = sample_page[0][0] if len(sample_page[0]) > 0 else {}
                    if isinstance(sample_item, dict):
                        print(f"  Beispiel: text='{sample_item.get('text', '')[:30]}', bbox={sample_item.get('bbox', [])}")
            else:
                print(f"  Unerwartetes Format: {type(sample_page)}")
    except Exception as e:
        print(f"  Status: FEHLER - {str(e)}")
    
    # Test standardize_ocr_output
    print(f"- Standardisierungstest:")
    try:
        result = ocr_pdf(pdf_path, engine=engine_name, include_bbox=True)
        if result and len(result) > 0:
            sample_page = result[0]
            standardized = standardize_ocr_output(sample_page, format_type="formatted_string")
            print(f"  Status: Erfolgreich - Konvertierung zu standardisiertem Format")
            preview = standardized[:100] + "..." if isinstance(standardized, str) and len(standardized) > 100 else standardized
            print(f"  Beispiel: {preview}")
    except Exception as e:
        print(f"  Status: FEHLER - {str(e)}")

def main():
    """Hauptfunktion für den OCR-Engine-Test."""
    # Finde eine Beispiel-PDF-Datei
    sample_pdf = str(Path(project_root) / "resources" / "samples" / "BRE-03.pdf")
    
    if not os.path.exists(sample_pdf):
        print(f"Fehler: Beispiel-PDF nicht gefunden: {sample_pdf}")
        return 1
    
    print(f"Verwende Beispiel-PDF: {sample_pdf}")
    
    # Teste alle verfügbaren OCR-Engines
    engines = get_available_engines()
    print(f"Verfügbare OCR-Engines: {', '.join(engines)}")
    
    for engine in engines:
        test_engine(engine, sample_pdf)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 