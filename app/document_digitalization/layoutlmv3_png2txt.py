import os
import re
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
import logging

_PROCESSOR = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large")
_MODEL = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-large")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("layoutlmv3_png2txt")


def layoutlm_image_to_text(image_path: str) -> str:
    """
    Extracts text chunks from an image using LayoutLMv3 and returns
    a formatted text string with y-coordinates similar to doctr output.
    Logs errors if processing fails.
    """
    try:
        image = _load_image(image_path)
        inputs = _PROCESSOR(images=image, return_tensors="pt", truncation=True, max_length=512)
        _MODEL(**{k: inputs[k] for k in ("input_ids", "attention_mask", "bbox", "pixel_values")})

        tokens = _flatten_tokens(inputs)
        sorted_lines = _bucket_tokens_by_line(tokens)
        thresh = _compute_dynamic_threshold(sorted_lines)
        chunks = _chunk_all_lines(sorted_lines, thresh)
        chunk_objs = _make_chunk_objects(chunks)
        
        # Format lines similar to doctr's output with y-coordinates
        formatted_lines = []
        for chunk in chunk_objs:
            # Extract y-coordinate (middle point of the vertical coordinates)
            x0, y0, x1, y1 = chunk["bbox"]
            y_center = int((y0 + y1) / 2)
            
            # Format text with y-coordinate and quotes like doctr
            formatted_line = f'[y={y_center}] "{chunk["text"]}"'
            formatted_lines.append(formatted_line)
        print(formatted_lines)
        return "\n".join(formatted_lines)
    except Exception as e:
        logger.exception(f"layoutlm_image_to_text failed for '{image_path}':")
        raise


def _load_image(path: str) -> Image.Image:
    """Loads an image from the given path. Logs and raises FileNotFoundError if not found."""
    if not os.path.isfile(path):
        logger.error(f"No such file: {path}")
        raise FileNotFoundError(f"No such file: {path}")
    return Image.open(path).convert("RGB")


def _flatten_tokens(inputs) -> List[Dict]:
    toks = []
    special = set(_PROCESSOR.tokenizer.all_special_tokens)
    ids = inputs["input_ids"][0]
    bboxes = inputs["bbox"][0]
    for idx, tok_id in enumerate(ids):
        tok_str = _PROCESSOR.tokenizer.convert_ids_to_tokens(int(tok_id))
        if tok_str in special:
            continue
        toks.append({"text": tok_str, "bbox": bboxes[idx].tolist()})
    return toks


def _bucket_tokens_by_line(tokens: List[Dict], bucket_height: int = 10) -> List[List[Dict]]:
    lines = {}
    for t in tokens:
        y0, y1 = t["bbox"][1], t["bbox"][3]
        key = int(((y0 + y1) / 2) // bucket_height)
        lines.setdefault(key, []).append(t)
    return [sorted(line, key=lambda x: x["bbox"][0]) for _, line in sorted(lines.items())]


def _compute_dynamic_threshold(sorted_lines: List[List[Dict]]) -> int:
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
    chunks, current = [], [tokens[0]]
    for tok in tokens[1:]:
        if tok["bbox"][0] - current[-1]["bbox"][2] > thresh:
            chunks.append(current)
            current = [tok]
        else:
            current.append(tok)
    chunks.append(current)
    return chunks


def _chunk_all_lines(sorted_lines: List[List[Dict]], thresh: int) -> List[List[Dict]]:
    chunks = []
    for line in sorted_lines:
        if line:
            chunks.extend(_chunk_line(line, thresh))
    return chunks


def _make_chunk_objects(chunks: List[List[Dict]]) -> List[Dict]:
    objs = []
    for chunk in chunks:
        raw = "".join(t["text"] for t in chunk)
        text = re.sub(r"\s+", " ", raw.replace("Ġ", " ")).strip()
        if not text:
            continue
        xs = [coord for t in chunk for coord in (t["bbox"][0], t["bbox"][2])]
        ys = [coord for t in chunk for coord in (t["bbox"][1], t["bbox"][3])]
        objs.append({"text": text, "bbox": [min(xs), min(ys), max(xs), max(ys)]})
    return objs