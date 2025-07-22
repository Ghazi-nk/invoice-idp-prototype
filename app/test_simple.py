#!/usr/bin/env python3
"""
Vereinfachter Test für eine einzelne OCR-Engine.
"""

import os
import sys
from pathlib import Path
import logging

# Konfiguriere Logging für detaillierte Ausgaben
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get absolute path to the project root
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Print für Debug-Zwecke
print("Skript gestartet")

from app.document_processing import ocr_pdf
from app.document_digitalization.pre_processing import standardize_ocr_output

def main():
    """Teste die PaddleOCR-Engine."""
    engine = "paddleocr"
    sample_pdf = str(Path(project_root) / "resources" / "samples" / "BRE-03.pdf")
    
    if not os.path.exists(sample_pdf):
        print(f"Fehler: PDF nicht gefunden: {sample_pdf}")
        return 1
    
    print(f"Verwende PDF: {sample_pdf}")
    
    # Teste ohne Bounding Boxes
    print("\n=== Test ohne Bounding Boxes ===")
    try:
        result = ocr_pdf(sample_pdf, engine=engine, include_bbox=False)
        print(f"Erfolg! Anzahl der Seiten: {len(result) if result else 0}")
        
        if result and len(result) > 0:
            print(f"Typ der ersten Seite: {type(result[0])}")
            if isinstance(result[0], str):
                preview = result[0][:200] + "..." if len(result[0]) > 200 else result[0]
                print(f"Vorschau: {preview}")
    except Exception as e:
        print(f"Fehler beim Test ohne Bounding Boxes: {e}")
        import traceback
        traceback.print_exc()
    
    # Teste mit Bounding Boxes
    print("\n=== Test mit Bounding Boxes ===")
    try:
        result_bbox = ocr_pdf(sample_pdf, engine=engine, include_bbox=True)
        print(f"Erfolg! Anzahl der Seiten: {len(result_bbox) if result_bbox else 0}")
        
        if result_bbox and len(result_bbox) > 0:
            first_page = result_bbox[0]
            print(f"Typ der ersten Seite: {type(first_page)}")
            
            if isinstance(first_page, list):
                print(f"Anzahl der Elemente auf der ersten Seite: {len(first_page)}")
                
                if first_page and isinstance(first_page[0], dict):
                    first_item = first_page[0]
                    print(f"Beispiel-Item: {first_item}")
                elif first_page and isinstance(first_page[0], list) and len(first_page[0]) > 0:
                    first_item = first_page[0][0]
                    print(f"Beispiel-Item (verschachtelt): {first_item}")
            
            # Teste standardize_ocr_output
            print("\n=== Test standardize_ocr_output ===")
            standardized = standardize_ocr_output(first_page, format_type="formatted_string")
            print(f"Standardisierung erfolgreich: {bool(standardized)}")
            if isinstance(standardized, str):
                preview = standardized[:200] + "..." if len(standardized) > 200 else standardized
                print(f"Standardisiert: {preview}")
    except Exception as e:
        print(f"Fehler beim Test mit Bounding Boxes: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    print("Hauptfunktion wird aufgerufen")
    sys.exit(main()) 