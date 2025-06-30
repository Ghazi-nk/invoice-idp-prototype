import os
import json
import numpy as np
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering, AutoTokenizer
import pytesseract
from config import TMP_DIR, TESSERACT_CMD

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def layoutlm_image_to_text(image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(TMP_DIR, f"{base}.txt")

    # Initialize processor, model, tokenizer
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {image_path}")

    # Process image
    inputs = processor(images=image, return_tensors="pt")

    # Prepare model inputs
    expected = ['input_ids', 'attention_mask', 'bbox', 'pixel_values']
    model_inputs = {k: inputs[k] for k in expected if k in inputs}

    # Forward pass
    _ = model(**model_inputs)

    # 1) Flatten tokens + bboxes, filter special tokens
    all_tokens = []
    special_tokens = set(tokenizer.all_special_tokens)
    for idx, token_id in enumerate(inputs['input_ids'][0]):
        token_str = tokenizer.convert_ids_to_tokens(int(token_id))
        # Skip special tokens
        if token_str in special_tokens:
            continue
        # Normalize: strip leading 'Ġ' (space marker)
        token_clean = token_str.lstrip('Ġ')
        if not token_clean:
            continue
        bbox = inputs['bbox'][0][idx].tolist()
        all_tokens.append({'text': token_clean, 'bbox': bbox})

    # 2) Group into lines by y-center buckets
    lines = {}
    for t in all_tokens:
        y_center = (t['bbox'][1] + t['bbox'][3]) / 2
        bucket = int(y_center // 10)
        lines.setdefault(bucket, []).append(t)
    sorted_lines = sorted(lines.items(), key=lambda x: x[0])

    # -------------------------------------------------------------
    # Dynamisches Chunking (dokumentweite Gap-Schwelle)
    # -------------------------------------------------------------
    # 1) Sammle alle horizontalen Abstände
    all_gaps = []
    for _, line_tokens in sorted_lines:
        lt = sorted(line_tokens, key=lambda x: x['bbox'][0])
        for i in range(len(lt) - 1):
            gap = lt[i+1]['bbox'][0] - lt[i]['bbox'][2]
            if gap > 0:
                all_gaps.append(gap)

    # 2) Bestimme Schwellenwert
    if all_gaps:
        doc_thresh = max(int(np.percentile(all_gaps, 85)), 12)
    else:
        doc_thresh = 40

    # 3) Chunk-Funktion
    def chunk_line(tokens_sorted, thresh):
        chunks, cur = [], [tokens_sorted[0]]
        for tok in tokens_sorted[1:]:
            gap = tok['bbox'][0] - cur[-1]['bbox'][2]
            if gap > thresh:
                chunks.append(cur)
                cur = [tok]
            else:
                cur.append(tok)
        chunks.append(cur)
        return chunks

    # 4) Wende Chunking an
    chunks = []
    for _, line_tokens in sorted_lines:
        lt = sorted(line_tokens, key=lambda x: x['bbox'][0])
        if lt:
            chunks.extend(chunk_line(lt, doc_thresh))

    # 5) Baue chunk_objects und bereinige Text
    chunk_objects = []
    for ch in chunks:
        texts = [t['text'] for t in ch]
        # Join with single space and strip
        chunk_text = ' '.join(texts).strip()
        if not chunk_text:
            continue
        # Compute aggregate bbox
        xs = [t['bbox'][0] for t in ch] + [t['bbox'][2] for t in ch]
        ys = [t['bbox'][1] for t in ch] + [t['bbox'][3] for t in ch]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        chunk_objects.append({'text': chunk_text, 'bbox': bbox})

    # Save to file
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for obj in chunk_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    return json.dumps(chunk_objects, ensure_ascii=False)


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    layoutlm_image_to_text(path)
