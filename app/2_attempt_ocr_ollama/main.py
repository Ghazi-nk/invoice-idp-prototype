import os
import json
from config import INPUT_DIR, OUTPUT_DIR, TMP_DIR
from pipeline import process_single_pdf

def main():
    # Verzeichnisse sicherstellen
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Alle PDFs im INPUT_DIR verarbeiten
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf = os.path.join(INPUT_DIR, fname)
        out = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}.json")

        if os.path.isfile(out):
            print(f"{out} existiert. Überspringe...")
            continue

        result = process_single_pdf(pdf)
        if result:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"[Info] Gespeichert: {out}")

    print("→ Fertig.")

if __name__ == "__main__":
    main()