import json
import csv
import re
import unicodedata
import time
from pathlib import Path

from utils.config import INVOICES_DIR, GROUND_TRUTH_DIR as GT_DIR, OLLAMA_MODEL
from utils.pdf_utils import pdf_to_png, extract_text_if_searchable
from document_digitalization.ocr import tesseract_png_to_text, easyocr_png_to_text
from document_digitalization.layoutlmv3_png2txt import layoutlm_image_to_text
from document_digitalization.doctr_pdf2txt import doctr_pdf_to_text

from utils.llm_utils import ask_ollama_for_invoice_fields

# Config
MODEL_SAFE         = OLLAMA_MODEL.replace(":", "_")
OUTPUT_SUMMARY_CSV = f"benchmark_summary_{MODEL_SAFE}.csv"
OUTPUT_DETAIL_CSV  = f"benchmark_details_{MODEL_SAFE}.csv"
USE_SEARCHABLE     = False
PIPELINES          = ["easyocr", "tesseract", "layoutlm", "doctr"]

_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

def canon_text(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.strip().rstrip(".:,").lower()
    s = ''.join(c for c in s if c.isalnum() or c.isspace())
    return ' '.join(s.split())

def canon_money(x: str | float | None) -> float | None:
    if x in (None, "", "null"):
        return None
    if isinstance(x, (int, float)):
        return round(float(x), 2)
    x = str(x).replace("'", "").replace(" ", "").replace("€", "").replace(",", ".")
    try:
        return round(float(x), 2)
    except ValueError:
        return None

def canon_id(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

ID_KEYS    = {"ust-id", "iban"}
MONEY_KEYS = {"total_amount"}

def is_match(field: str, true_val, pred_val) -> bool:
    if true_val in (None, "", "null"):
        return pred_val in (None, "", "null")
    if field in MONEY_KEYS:
        return canon_money(true_val) == canon_money(pred_val)
    if field in ID_KEYS:
        return canon_id(true_val) == canon_id(pred_val)
    return canon_text(true_val) == canon_text(pred_val)

def clean_ocr_text(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"\bA7(\d{2})", r"AT\1", txt)
    txt = txt.replace("'", "")
    txt = txt.replace("Â€", "€")
    txt = re.sub(r"(\d),(\d{2})(\s*€?)", r"\1.\2\3", txt)
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    return txt

def extract_text(pdf_path: str, pipeline: str, use_searchable: bool = False) -> str | None:
    if use_searchable:
        txt = extract_text_if_searchable(pdf_path)
        if txt:
            return clean_ocr_text(txt)
    if pipeline == "doctr":
        txt = doctr_pdf_to_text(pdf_path)
    elif pipeline == "easyocr":
        png = pdf_to_png(pdf_path)
        txt = easyocr_png_to_text(png)
    elif pipeline == "tesseract":
        png = pdf_to_png(pdf_path)
        txt = tesseract_png_to_text(png)
    elif pipeline == "layoutlm":
        png = pdf_to_png(pdf_path)
        txt = layoutlm_image_to_text(png)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    return clean_ocr_text(txt)

def calc_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)

def main():
    inv_paths = sorted(Path(INVOICES_DIR).glob("*.pdf"))
    if not inv_paths:
        raise RuntimeError(f"No PDFs found in {INVOICES_DIR}")

    first_base = inv_paths[0].stem
    with open(Path(GT_DIR)/f"{first_base}.json", encoding="utf-8") as fh:
        fields = list(json.load(fh).keys())

    header_summary = ["invoice", "pipeline", *fields, "accuracy", "precision", "recall", "f1", "duration"]

    # buffering=1 sorgt für Zeilenpufferung
    with open(OUTPUT_SUMMARY_CSV, "w", newline="", encoding="utf-8", buffering=1) as sum_f, \
         open(OUTPUT_DETAIL_CSV,  "w", newline="", encoding="utf-8") as det_f:

        summary_writer = csv.DictWriter(sum_f, fieldnames=header_summary)
        detail_writer  = csv.DictWriter(det_f, fieldnames=["invoice","pipeline","field","expected","predicted","match"])
        summary_writer.writeheader()
        detail_writer.writeheader()

        for pipeline in PIPELINES:
            print(f"\n▶ Running pipeline: {pipeline}")
            for pdf_path in inv_paths:
                start_time = time.perf_counter()

                base = pdf_path.stem
                gt   = json.load(open(Path(GT_DIR)/f"{base}.json", encoding="utf-8"))

                ocr_txt = extract_text(str(pdf_path), pipeline, use_searchable=USE_SEARCHABLE)
                pred    = ask_ollama_for_invoice_fields(ocr_txt)

                row_sum = {"invoice": base, "pipeline": pipeline}
                correct = tp = fp = fn = 0

                for fld in fields:
                    match = is_match(fld, gt.get(fld), pred.get(fld))
                    row_sum[fld] = int(match)
                    if match:
                        correct += 1
                        if gt.get(fld) not in (None, "", "null"):
                            tp += 1
                    else:
                        gt_has   = gt.get(fld)   not in (None, "", "null")
                        pred_has = pred.get(fld) not in (None, "", "null")
                        if gt_has:   fn += 1
                        if pred_has: fp += 1

                    detail_writer.writerow({
                        "invoice": base,
                        "pipeline": pipeline,
                        "field": fld,
                        "expected": gt.get(fld),
                        "predicted": pred.get(fld),
                        "match": int(match)
                    })

                row_sum["accuracy"] = round(correct / len(fields), 3)
                prec, rec, f1 = calc_metrics(tp, fp, fn)
                row_sum.update(precision=prec, recall=rec, f1=f1)
                row_sum["duration"] = round(time.perf_counter() - start_time, 3)

                summary_writer.writerow(row_sum)
                sum_f.flush()  # direkt nach jedem write!

    print(f"\n✓ Done – wrote {OUTPUT_SUMMARY_CSV} & {OUTPUT_DETAIL_CSV}")

if __name__ == "__main__":
    main()
