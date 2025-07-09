import json
import csv
import re
import unicodedata
import time
import os
from pathlib import Path
import multiprocessing

# --- Main pipeline functions to be benchmarked ---
from document_processing import extract_invoice_fields_from_pdf, get_available_engines

# --- Configuration for benchmark directories and files ---
from utils.config import (
    INVOICES_DIR,
    LABELS_DIR,
    OLLAMA_MODEL
)

# --- Configuration for this script ---
# The "Safety Valve": Set the number of parallel processes to run.
# 1 = Safe sequential mode (invoice-centric).
# >1 = Parallel mode. Start with 2-3 and increase if your machine can handle it.
NUM_WORKERS = 2
MODEL_SAFE = OLLAMA_MODEL.replace(":", "_")
OUTPUT_SUMMARY_CSV = f"summary_{MODEL_SAFE}.csv"
OUTPUT_DETAIL_CSV = f"details_{MODEL_SAFE}.csv"


# =============================================================================
# --- Evaluation Helpers ---
# =============================================================================

def canon_text(s: str | None) -> str:
    """Canonicalize a string for comparison."""
    if not s: return ""
    s = unicodedata.normalize("NFKD", str(s)).strip().rstrip(".:,").lower()
    s = ''.join(c for c in s if c.isalnum() or c.isspace())
    return ' '.join(s.split())


def canon_money(x: str | float | None) -> float | None:
    """Canonicalize a money value to a float."""
    if x in (None, "", "null"): return None
    if isinstance(x, (int, float)): return round(float(x), 2)
    x = str(x).replace("'", "").replace(" ", "").replace("€", "").replace(",", ".")
    try:
        return round(float(x), 2)
    except ValueError:
        return None


def canon_id(s: str | None) -> str:
    """Canonicalize an ID string."""
    if not s: return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def is_match(field: str, true_val, pred_val) -> bool:
    """Compares a true and predicted value based on the field type."""
    ID_KEYS = {"ust-id", "iban"}
    MONEY_KEYS = {"total_amount"}
    if true_val in (None, "", "null"): return pred_val in (None, "", "null")
    if field in MONEY_KEYS: return canon_money(true_val) == canon_money(pred_val)
    if field in ID_KEYS: return canon_id(true_val) == canon_id(pred_val)
    return canon_text(true_val) == canon_text(pred_val)


def calc_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Calculates precision, recall, and F1-score."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


# =============================================================================
# --- Worker Function for Parallel Processing ---
# =============================================================================

def process_task(task_args: tuple) -> dict | None:
    """
    Processes a single (pdf_path, engine, label_path) task.
    This function is executed by each worker in the pool.
    """
    pdf_path, engine, label_path = task_args
    base_name = pdf_path.stem

    print(f"  Processing '{base_name}.pdf' with '{engine}'...")
    try:
        start_time = time.perf_counter()
        with open(label_path, encoding="utf-8") as f:
            gt = json.load(f)

        # --- CORE PIPELINE CALL ---
        pred = extract_invoice_fields_from_pdf(
            pdf_path=str(pdf_path),
            engine=engine,
            clean=True
        )
        duration = round(time.perf_counter() - start_time, 3)

        # --- Evaluation ---
        fields = list(gt.keys())
        summary_row = {"invoice": base_name, "pipeline": engine, "duration": duration}
        detail_rows = []
        correct = tp = fp = fn = 0

        for fld in fields:
            match = is_match(fld, gt.get(fld), pred.get(fld))
            summary_row[fld] = int(match)
            if match:
                correct += 1
                if gt.get(fld) not in (None, "", "null"): tp += 1
            else:
                gt_has = gt.get(fld) not in (None, "", "null")
                pred_has = pred.get(fld) not in (None, "", "null")
                if gt_has: fn += 1
                if pred_has: fp += 1

            detail_rows.append({
                "invoice": base_name, "pipeline": engine, "field": fld,
                "expected": gt.get(fld), "predicted": pred.get(fld), "match": int(match)
            })

        summary_row["accuracy"] = round(correct / len(fields), 3)
        prec, rec, f1 = calc_metrics(tp, fp, fn)
        summary_row.update(precision=prec, recall=rec, f1=f1)

        print(f"  ✓ Finished '{base_name}.pdf' with '{engine}' in {duration}s.")
        return {"summary": summary_row, "details": detail_rows}

    except Exception as e:
        print(f"  ✗ ERROR processing '{base_name}.pdf' with '{engine}': {e}")
        return None


# =============================================================================
# --- Main Benchmark Execution ---
# =============================================================================

def main():
    """
    Runs the main benchmarking process.
    """
    # 1. Load existing results to avoid re-processing
    completed_work = set()
    if os.path.exists(OUTPUT_SUMMARY_CSV):
        with open(OUTPUT_SUMMARY_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row: completed_work.add((row[0], row[1]))  # (invoice, pipeline)
    print(f"Found {len(completed_work)} previously completed tasks.")

    # 2. Build the "To-Do List" of tasks
    tasks_to_do = []
    all_pdfs = sorted(Path(INVOICES_DIR).glob("*.pdf"))
    all_engines = get_available_engines()

    for pdf_path in all_pdfs:
        base_name = pdf_path.stem
        label_path = Path(LABELS_DIR) / f"{base_name}.json"
        if not label_path.exists():
            print(f"  ! Warning: No label file for '{base_name}.pdf'. Skipping this file for all engines.")
            continue

        for engine in all_engines:
            if (base_name, engine) not in completed_work:
                tasks_to_do.append((pdf_path, engine, label_path))

    if not tasks_to_do:
        print("✓ No new tasks to process. Benchmark is up-to-date.")
        return
    print(f"Created {len(tasks_to_do)} new tasks to process.")

    # 3. Setup CSV writers
    summary_exists = os.path.exists(OUTPUT_SUMMARY_CSV)
    details_exists = os.path.exists(OUTPUT_DETAIL_CSV)

    with open(OUTPUT_SUMMARY_CSV, "a", newline="", encoding="utf-8", buffering=1) as sum_f, \
            open(OUTPUT_DETAIL_CSV, "a", newline="", encoding="utf-8", buffering=1) as det_f:

        # Determine headers from a label file
        first_label_path = Path(LABELS_DIR) / f"{all_pdfs[0].stem}.json"
        with open(first_label_path, encoding="utf-8") as fh:
            fields = list(json.load(fh).keys())

        header_summary = ["invoice", "pipeline", *fields, "accuracy", "precision", "recall", "f1", "duration"]
        header_details = ["invoice", "pipeline", "field", "expected", "predicted", "match"]

        summary_writer = csv.DictWriter(sum_f, fieldnames=header_summary)
        detail_writer = csv.DictWriter(det_f, fieldnames=header_details)

        if not summary_exists: summary_writer.writeheader()
        if not details_exists: detail_writer.writeheader()

        # 4. Execute tasks (either in parallel or sequentially)
        if NUM_WORKERS > 1:
            print(f"\n▶ Running in PARALLEL mode with {NUM_WORKERS} workers...")
            with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                for result in pool.imap_unordered(process_task, tasks_to_do):
                    if result:
                        summary_writer.writerow(result["summary"])
                        detail_writer.writerows(result["details"])
        else:
            print("\n▶ Running in SEQUENTIAL mode (invoice-centric)...")
            # Group tasks by PDF for invoice-centric processing
            tasks_by_pdf = {}
            for task in tasks_to_do:
                pdf_path = task[0]
                tasks_by_pdf.setdefault(pdf_path, []).append(task)

            for pdf_path, pdf_tasks in tasks_by_pdf.items():
                print(f"\n-- Processing PDF: {pdf_path.name} --")
                for task in pdf_tasks:
                    result = process_task(task)
                    if result:
                        summary_writer.writerow(result["summary"])
                        detail_writer.writerows(result["details"])

    print(f"\n✓ Benchmark complete. Results are in {OUTPUT_SUMMARY_CSV} & {OUTPUT_DETAIL_CSV}")


if __name__ == "__main__":
    # It's recommended to run this script from the project root directory
    # so that all relative paths in the config work correctly.
    main()