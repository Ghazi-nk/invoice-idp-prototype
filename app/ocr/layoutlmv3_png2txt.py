"""
LayoutLMv3 OCR-Engine für layout-bewusste Dokumentenverarbeitung.

Dieses Modul implementiert die Integration von Microsoft LayoutLMv3,
einem Transformer-Modell für multimodale Dokumentenverarbeitung.
LayoutLMv3 kombiniert Text-, Layout- und Bildinformationen für hochpräzise
Texterkennung und Strukturverständnis.

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""
import os
import re
from typing import List, Dict, Any, Union, cast

import numpy as np
from PIL import Image
from transformers import LayoutLMv3Processor
import logging

from app.logging_config import ocr_logger

# Initialize processor once
_PROCESSOR = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large")


def layoutlm_image_to_text(image_path: str) -> str:
    """
    Extrahiert Text aus einem Bild mittels LayoutLMv3.
    
    Diese Funktion nutzt Microsoft LayoutLMv3 für multimodale
    Dokumentenverarbeitung, die Text-, Layout- und Bildinformationen
    kombiniert. Sie bietet hochpräzise Texterkennung auch bei
    komplexen Dokumentstrukturen.
    
    Args:
        image_path (str): Pfad zur Bilddatei (PNG, JPEG, etc.)
    
    Returns:
        str: Extrahierter Text als zusammenhängender String
        
    Raises:
        FileNotFoundError: Wenn Bilddatei nicht gefunden wird
        Exception: Bei Verarbeitungsfehlern des LayoutLMv3-Modells
        
    Note:
        - Verwendet microsoft/layoutlmv3-large Pretrained Model
        
    """
    try:
        # 1. Bild laden und mit LayoutLM verarbeiten
        if not os.path.isfile(image_path):
            ocr_logger.error(f"Bild nicht gefunden: {image_path}")
            raise FileNotFoundError(f"Datei nicht gefunden: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        inputs = _PROCESSOR(images=image, return_tensors="pt", truncation=True, max_length=512)
        
        # 2. Tokens extrahieren
        tokens = []
        processor_any = cast(Any, _PROCESSOR)
        tokenizer = getattr(processor_any, "tokenizer", None)
        
        if tokenizer:
            try:
                special_tokens = set(tokenizer.all_special_tokens)
                
                # Casting zur Vermeidung von Linter-Fehlern
                inputs_any = cast(Dict[str, Any], inputs)
                if "input_ids" in inputs_any and "bbox" in inputs_any:
                    ids = inputs_any["input_ids"][0]
                    bboxes = inputs_any["bbox"][0]
                    
                    for idx, tok_id in enumerate(ids):
                        try:
                            tok_str = tokenizer.convert_ids_to_tokens(int(tok_id))
                            if tok_str in special_tokens:
                                continue
                            tokens.append({"text": tok_str, "bbox": bboxes[idx].tolist()})
                        except Exception:
                            # Problematische Tokens überspringen
                            continue
            except Exception as e:
                ocr_logger.warning(f"Fehler bei Tokenverarbeitung: {e}")
        
        # Falls keine Tokens extrahiert werden konnten
        if not tokens:
            ocr_logger.warning(f"Keine Tokens für {image_path} extrahiert")
            return ""
        
        # 3. Tokens nach Zeilen gruppieren
        lines_dict = {}
        line_height = 10  # Zeilenhöhe in Pixeln
        
        for token in tokens:
            y0, y1 = token["bbox"][1], token["bbox"][3]
            y_center = int((y0 + y1) / 2)
            line_key = y_center // line_height
            
            if line_key not in lines_dict:
                lines_dict[line_key] = []
            lines_dict[line_key].append(token)
        
        # Zeilen sortieren
        lines = []
        for _, line_tokens in sorted(lines_dict.items()):
            lines.append(sorted(line_tokens, key=lambda t: t["bbox"][0]))
            
        # 4. Durchschnittlichen Abstand zwischen Tokens berechnen
        all_gaps = []
        for line in lines:
            for i in range(len(line) - 1):
                gap = line[i+1]["bbox"][0] - line[i]["bbox"][2]
                if gap > 0:  # Nur positive Abstände
                    all_gaps.append(gap)
        
        gap_threshold = max(int(np.percentile(all_gaps, 85)), 12) if all_gaps else 20
        
        # 5. Zeilen in Chunks aufteilen basierend auf horizontalen Abständen
        text_chunks = []
        
        for line in lines:
            if not line:
                continue
                
            if len(line) == 1:
                text_chunks.append([line[0]])
                continue
            
            current_chunk = [line[0]]
            for token in line[1:]:
                # Prüfen, ob Abstand zum vorherigen Token den Schwellenwert überschreitet
                if token["bbox"][0] - current_chunk[-1]["bbox"][2] > gap_threshold:
                    # Neuen Chunk beginnen
                    text_chunks.append(current_chunk)
                    current_chunk = [token]
                else:
                    # Aktuellen Chunk fortsetzen
                    current_chunk.append(token)
            
            # Letzten Chunk hinzufügen
            if current_chunk:
                text_chunks.append(current_chunk)
        
        # 6. Token-Chunks in Textobjekte konvertieren
        text_objects = []
        
        for chunk in text_chunks:
            # Token-Texte zu vollständigem Text zusammenfügen
            raw_text = "".join(t["text"] for t in chunk)

            # Text bereinigen
            text = re.sub(r"\s+", " ", raw_text.replace("Ġ", " ")).strip()

            # Leere Chunks überspringen
            if not text:
                continue
                
            # Bounding-Box für alle Tokens im Chunk berechnen
            x_coords = [coord for token in chunk for coord in (token["bbox"][0], token["bbox"][2])]
            y_coords = [coord for token in chunk for coord in (token["bbox"][1], token["bbox"][3])]
            
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            text_objects.append({"text": text, "bbox": bbox})
        
        # Reiner Text ohne Koordinaten zurückgeben
        # Text-Chunks nach vertikaler Position sortieren und zusammenfügen
        sorted_objects = sorted(text_objects, key=lambda obj: (obj["bbox"][1] + obj["bbox"][3]) / 2)
        plain_text = "\n".join(obj["text"] for obj in sorted_objects)
        return plain_text
            
    except Exception as e:
        ocr_logger.exception(f"LayoutLM OCR fehlgeschlagen für '{image_path}':")
        raise