import json
import csv
import time
import os
from pathlib import Path
import multiprocessing
import collections
import statistics


from document_processing import extract_invoice_fields_from_pdf, get_available_engines

from evaluation_utils import is_match, check_acceptance

# --- Configuration for benchmark ---
from utils.config import (
    INVOICES_DIR,
    LABELS_DIR,
    OLLAMA_MODEL
)

NUM_WORKERS = 2
MODEL_SAFE = OLLAMA_MODEL.replace(":", "_")
OUTPUT_SUMMARY_CSV = f"summary_{MODEL_SAFE}.csv"
OUTPUT_DETAIL_CSV = f"details_{MODEL_SAFE}.csv"
OUTPUT_RESULTS_CSV = f"results_{MODEL_SAFE}.csv"


def calc_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Calculates precision, recall, and F1-score."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def process_task(task_args: tuple) -> dict | None:
    """Processes a single (pdf_path, engine, label_path) task."""
    pdf_path, engine, label_path = task_args
    base_name = pdf_path.stem

    print(f"  Processing '{base_name}.pdf' with '{engine}'...")
    try:
        start_time = time.perf_counter()
        with open(label_path, encoding="utf-8") as f:
            gt = json.load(f)

        pred = extract_invoice_fields_from_pdf(
            pdf_path=str(pdf_path),
            engine=engine,
            clean=True
        )
        duration = round(time.perf_counter() - start_time, 3)

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

        # --- NEW: Check business rule acceptance ---
        summary_row["acceptance"] = int(check_acceptance(gt, pred))

        print(f"  ✓ Finished '{base_name}.pdf' with '{engine}' in {duration}s.")
        return {"summary": summary_row, "details": detail_rows}

    except Exception as e:
        print(f"  ✗ ERROR processing '{base_name}.pdf' with '{engine}': {e}")
        return None


def generate_final_results():
    """Calculates and saves the aggregated results for each pipeline."""
    if not os.path.exists(OUTPUT_SUMMARY_CSV):
        print("Warning: summary.csv not found. Cannot generate final results.")
        return

    pipeline_metrics = collections.defaultdict(lambda: {
        'accuracies': [], 'precisions': [], 'recalls': [], 'f1s': [], 'durations': [], 'acceptances': []
    })

    with open(OUTPUT_SUMMARY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pipeline = row['pipeline']
            pipeline_metrics[pipeline]['accuracies'].append(float(row['accuracy']))
            pipeline_metrics[pipeline]['precisions'].append(float(row['precision']))
            pipeline_metrics[pipeline]['recalls'].append(float(row['recall']))
            pipeline_metrics[pipeline]['f1s'].append(float(row['f1']))
            pipeline_metrics[pipeline]['durations'].append(float(row['duration']))
            pipeline_metrics[pipeline]['acceptances'].append(float(row['acceptance']))

    header = ["pipeline", "mean_accuracy", "mean_precision", "mean_recall", "mean_f1", "mean_duration",
              "acceptance_rate"]
    with open(OUTPUT_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for pipeline, data in sorted(pipeline_metrics.items()):
            writer.writerow({
                "pipeline": pipeline,
                "mean_accuracy": round(statistics.mean(data['accuracies']), 3),
                "mean_precision": round(statistics.mean(data['precisions']), 3),
                "mean_recall": round(statistics.mean(data['recalls']), 3),
                "mean_f1": round(statistics.mean(data['f1s']), 3),
                "mean_duration": round(statistics.mean(data['durations']), 3),
                "acceptance_rate": round(statistics.mean(data['acceptances']), 3)
            })


def main():
    """Runs the main benchmarking process."""
    completed_work = set()
    if os.path.exists(OUTPUT_SUMMARY_CSV):
        with open(OUTPUT_SUMMARY_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row: completed_work.add((row[0], row[1]))
    print(f"Found {len(completed_work)} previously completed tasks.")

    tasks_to_do = []
    all_pdfs = sorted(Path(INVOICES_DIR).glob("*.pdf"))
    all_engines = get_available_engines()

    for pdf_path in all_pdfs:
        base_name = pdf_path.stem
        label_path = Path(LABELS_DIR) / f"{base_name}.json"
        if not label_path.exists():
            print(f"  ! Warning: No label file for '{base_name}.pdf'. Skipping.")
            continue
        for engine in all_engines:
            if (base_name, engine) not in completed_work:
                tasks_to_do.append((pdf_path, engine, label_path))

    if not tasks_to_do:
        print("✓ No new tasks to process. Benchmark is up-to-date.")
    else:
        print(f"Created {len(tasks_to_do)} new tasks to process.")
        summary_exists = os.path.exists(OUTPUT_SUMMARY_CSV)
        details_exists = os.path.exists(OUTPUT_DETAIL_CSV)

        with open(OUTPUT_SUMMARY_CSV, "a", newline="", encoding="utf-8", buffering=1) as sum_f, \
                open(OUTPUT_DETAIL_CSV, "a", newline="", encoding="utf-8", buffering=1) as det_f:

            first_label_path = Path(LABELS_DIR) / f"{all_pdfs[0].stem}.json"
            with open(first_label_path, encoding="utf-8") as fh:
                fields = list(json.load(fh).keys())

            # --- NEW: Add 'acceptance' to the summary header ---
            header_summary = ["invoice", "pipeline", *fields, "accuracy", "precision", "recall", "f1", "acceptance",
                              "duration"]
            header_details = ["invoice", "pipeline", "field", "expected", "predicted", "match"]

            summary_writer = csv.DictWriter(sum_f, fieldnames=header_summary)
            detail_writer = csv.DictWriter(det_f, fieldnames=header_details)

            if not summary_exists: summary_writer.writeheader()
            if not details_exists: detail_writer.writeheader()

            if NUM_WORKERS > 1:
                print(f"\n▶ Running in PARALLEL mode with {NUM_WORKERS} workers...")
                with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                    for result in pool.imap_unordered(process_task, tasks_to_do):
                        if result:
                            summary_writer.writerow(result["summary"])
                            detail_writer.writerows(result["details"])
            else:
                print("\n▶ Running in SEQUENTIAL mode (invoice-centric)...")
                tasks_by_pdf = collections.defaultdict(list)
                for task in tasks_to_do:
                    tasks_by_pdf[task[0]].append(task)
                for pdf_path, pdf_tasks in sorted(tasks_by_pdf.items()):
                    print(f"\n-- Processing PDF: {pdf_path.name} --")
                    for task in pdf_tasks:
                        result = process_task(task)
                        if result:
                            summary_writer.writerow(result["summary"])
                            detail_writer.writerows(result["details"])

    print(f"\n✓ Benchmark complete.")
    generate_final_results()
    print(f"✓ Final results summary generated at {OUTPUT_RESULTS_CSV}")


if __name__ == "__main__":
    main()