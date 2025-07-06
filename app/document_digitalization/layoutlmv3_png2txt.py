import os
import json
import re
from typing import List, Dict

import numpy as np
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering

from utils.config import TMP_DIR, SAMPLE_PNG_PATH

# 1) Modelle & Processor einmal laden
_PROCESSOR = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large")
_MODEL = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-large")


def layoutlm_image_to_text(image_path: str) -> str:
    """
    Extrahiert Text‐Chunks aus einem Bild mit LayoutLMv3,
    speichert sie als JSON‐Zeilen in TMP_DIR/<basename>.txt
    und liefert das komplette JSON‐Array als String zurück.
    """
    image = _load_image(image_path)
    inputs = _PROCESSOR(images=image, return_tensors="pt", truncation=True, max_length=512)    # nur Forward‐Pass für Tokenization + bboxes

    _MODEL(**{k: inputs[k] for k in ("input_ids", "attention_mask", "bbox", "pixel_values")}) #todo: exception thrown here "index out of range in self None"

    tokens = _flatten_tokens(inputs)
    sorted_lines = _bucket_tokens_by_line(tokens)
    thresh = _compute_dynamic_threshold(sorted_lines)
    chunks = _chunk_all_lines(sorted_lines, thresh)
    chunk_objs = _make_chunk_objects(chunks)

    return _save_and_serialize(chunk_objs, image_path)


def _load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")
    return Image.open(path).convert("RGB")


def _flatten_tokens(inputs) -> List[Dict]:
    """
    Flacht die Token + ihre Bounding‐Boxes ab,
    filtert Spezial‐Tokens und behält das rohe Token‐String mit Ġ‐Präfix.
    """
    toks = []
    special = set(_PROCESSOR.tokenizer.all_special_tokens)
    ids = inputs["input_ids"][0]
    bboxes = inputs["bbox"][0]
    for idx, tok_id in enumerate(ids):
        tok_str = _PROCESSOR.tokenizer.convert_ids_to_tokens(int(tok_id))
        if tok_str in special:
            continue
        toks.append({
            "text": tok_str,               # hier noch inklusive 'Ġ' wenn vorhanden
            "bbox": bboxes[idx].tolist()
        })
    return toks


def _bucket_tokens_by_line(tokens: List[Dict], bucket_height: int = 10) -> List[List[Dict]]:
    """
    Gruppiert Tokens nach y‐Mittelpunkt in Zeilen.
    """
    lines = {}
    for t in tokens:
        y0, y1 = t["bbox"][1], t["bbox"][3]
        key = int(((y0 + y1) / 2) // bucket_height)
        lines.setdefault(key, []).append(t)
    return [
        sorted(line, key=lambda x: x["bbox"][0])
        for _, line in sorted(lines.items())
    ]


def _compute_dynamic_threshold(sorted_lines: List[List[Dict]]) -> int:
    """
    Berechnet das 85. Perzentil aller positiven Lücken
    zwischen benachbarten Tokens, mindestens 12px,
    Fallback 40px.
    """
    all_gaps = []
    for line in sorted_lines:
        for a, b in zip(line, line[1:]):
            gap = b["bbox"][0] - a["bbox"][2]
            if gap > 0:
                all_gaps.append(gap)

    if all_gaps:
        return max(int(np.percentile(all_gaps, 85)), 12)
    return 40


def _chunk_line(tokens: List[Dict], thresh: int) -> List[List[Dict]]:
    """
    Zerlegt eine einzelne Zeile in Chunks nach thresh.
    """
    chunks = []
    current = [tokens[0]]
    for tok in tokens[1:]:
        if tok["bbox"][0] - current[-1]["bbox"][2] > thresh:
            chunks.append(current)
            current = [tok]
        else:
            current.append(tok)
    chunks.append(current)
    return chunks


def _chunk_all_lines(sorted_lines: List[List[Dict]], thresh: int) -> List[List[Dict]]:
    """
    Wendet _chunk_line auf alle Zeilen an.
    """
    chunks = []
    for line in sorted_lines:
        if line:
            chunks.extend(_chunk_line(line, thresh))
    return chunks


def _make_chunk_objects(chunks: List[List[Dict]]) -> List[Dict]:
    """
    Baut aus jedem Chunk das finale Dict mit:
    - 'text': korrekter Text (Ġ→Leerzeichen, dann normalize)
    - 'bbox': Sammel‐Bounding‐Box
    """
    objs = []
    for chunk in chunks:
        # zusammensetzen und Ġ durch Space ersetzen
        raw = "".join(t["text"] for t in chunk)
        text = raw.replace("Ġ", " ")
        # Mehrfache Leerzeichen zusammenfassen
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        xs = [coord for t in chunk for coord in (t["bbox"][0], t["bbox"][2])]
        ys = [coord for t in chunk for coord in (t["bbox"][1], t["bbox"][3])]
        objs.append({
            "text": text,
            "bbox": [min(xs), min(ys), max(xs), max(ys)]
        })
    return objs


def _save_and_serialize(chunks: List[Dict], image_path: str) -> str:
    """
    Speichert jede Zeile als JSON in TMP_DIR und gibt
    das Gesamt‐JSON als String zurück.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(TMP_DIR, f"{base}.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return json.dumps(chunks, ensure_ascii=False)


# --- Test ---
if __name__ == "__main__":
    print(layoutlm_image_to_text(SAMPLE_PNG_PATH))
