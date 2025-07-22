#!/usr/bin/env python3
"""
Test-Skript für alle OCR-Engines
"""
import os
import sys
from pathlib import Path

# Pfad zum Projektverzeichnis hinzufügen
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.document_processing import get_available_engines, ocr_pdf


def test_engine(engine_name, pdf_path):
    """
    Test einer OCR-Engine mit und ohne Bounding-Box-Ausgabe
    """
    print(f"\n--- TEST: {engine_name.upper()} ---")
    
    # Test: Normale Ausgabe (ohne Bounding-Box)
    print("\n1. Normale Textausgabe (ohne Bounding-Box):")
    result = ocr_pdf(pdf_path, engine=engine_name, include_bbox=False)
    
    if result:
        print(f"  ✓ Erfolg! {len(result)} Seiten extrahiert")
        sample = result[0][:100].replace("\n", " ") + "..." if result[0] else ""
        print(f"  Beispiel: {sample}")
        print(f"  Typ: {type(result[0])}")
        
        # Prüfe, ob die Ausgabe Koordinaten enthält
        if isinstance(result[0], str):
            has_coords = "[y=" in result[0]
            print(f"  Enthält Koordinaten: {'JA (FEHLER!)' if has_coords else 'NEIN (KORREKT)'}")
    else:
        print("  ✗ Fehler: Keine Ausgabe erhalten")
    
    # Test: Ausgabe mit Bounding-Box
    print("\n2. Strukturierte Ausgabe (mit Bounding-Box):")
    result_bbox = ocr_pdf(pdf_path, engine=engine_name, include_bbox=True)
    
    if result_bbox:
        print(f"  ✓ Erfolg! {len(result_bbox)} Seiten extrahiert")
        
        if isinstance(result_bbox[0], list):
            count = len(result_bbox[0])
            print(f"  {count} Elemente auf Seite 1")
            
            if count > 0:
                first_elem = result_bbox[0][0]
                if isinstance(first_elem, dict) and "bbox" in first_elem and "text" in first_elem:
                    print(f"  Struktur korrekt: {{'text': '...', 'bbox': [...]}}")
                    print(f"  Beispiel: text='{first_elem['text'][:20]}...', bbox={first_elem['bbox']}")
                else:
                    print(f"  ✗ Ungültiges Format: {type(first_elem)}")
        else:
            print(f"  ✗ Ungültiges Format: {type(result_bbox[0])}")
    else:
        print("  ✗ Fehler: Keine Ausgabe erhalten")


def main():
    """Hauptfunktion"""
    # Beispiel-PDF suchen
    sample_pdf = str(Path(project_root) / "resources" / "samples" / "BRE-03.pdf")
    if not os.path.exists(sample_pdf):
        print(f"Fehler: Beispiel-PDF nicht gefunden: {sample_pdf}")
        return 1
    
    print(f"Verwende PDF: {sample_pdf}")
    
    # Verfügbare Engines testen
    engines = get_available_engines()
    print(f"Verfügbare OCR-Engines: {', '.join(engines)}")
    
    for engine in engines:
        test_engine(engine, sample_pdf)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 