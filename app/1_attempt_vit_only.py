import os
import json
from pdf2image import convert_from_path
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# This script extracts key fields from German invoice PDFs using a pretrained Donut model.
# Note: For improved accuracy on German invoices, consider fine-tuning on a relevant dataset.

def load_model(model_name: str = "naver-clova-ix/donut-base-finetuned-docvqa"):
    """
    Load the Donut processor and model for document question-answering.
    """
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

def extract_field(
    image: Image.Image,
    question: str,
    processor: DonutProcessor,
    model: VisionEncoderDecoderModel,
    dpi: int = 300
) -> str:
    """
    Given a PIL image and a question string, return the model's answer.
    Wraps the question in DocVQA tags and runs generation with beam search.
    """
    # 1) Ensure high-res RGB
    image = image.convert("RGB")
    # 2) Build DocVQA prompt
    task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    # 3) Get pixel values
    pixel_values = processor(image, return_tensors="pt").pixel_values
    # 4) Tokenize prompt without adding extra special tokens
    decoder_input_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids
    # 5) Generate answer tokens
    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=128,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    # 6) Decode and clean up
    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return answer

def process_invoice(
    pdf_path: str,
    processor: DonutProcessor,
    model: VisionEncoderDecoderModel,
    questions: dict,
    dpi: int = 300
) -> dict | None:
    """
    Convert the first page of a PDF to an image and extract each field via Donut QA.
    Returns a dict of field->answer, or None on failure.
    """
    try:
        pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    except Exception as e:
        print(f"[Error] Failed to convert '{pdf_path}': {e}")
        return None

    if not pages:
        print(f"[Warning] No pages found in '{pdf_path}'")
        return None

    image = pages[0]
    results: dict = {}
    for field, question in questions.items():
        try:
            answer = extract_field(image, question, processor, model, dpi=dpi)
        except Exception as e:
            print(f"[Error] Extracting '{field}' from '{pdf_path}': {e}")
            answer = ""
        results[field] = answer

    return results

def main(input_folder: str = "app/input", output_folder: str = "app/output"):
    os.makedirs(output_folder, exist_ok=True)
    processor, model = load_model()

    # German invoice fields mapped to QA prompts
    questions = {
        "Rechnungsnummer": "Was ist die Rechnungsnummer?",
        "Rechnungsdatum": "Was ist das Rechnungsdatum?",
        "Verkäufer": "Wer ist der Verkäufer?",
        "Käufer": "Wer ist der Käufer?",
        "Leistungsbeschreibung": "Was ist die Leistungsbeschreibung?",
        "Gesamtbetrag": "Wie hoch ist der Gesamtbetrag?",
        "Umsatzsteuer": "Wie hoch ist die Umsatzsteuer?"
    }

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(input_folder, fname)
        data = process_invoice(pdf_path, processor, model, questions)
        if data is None:
            continue

        out_name = os.path.splitext(fname)[0] + ".json"
        out_path = os.path.join(output_folder, out_name)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"[Info] Saved results to '{out_path}'")
        except Exception as e:
            print(f"[Error] Failed to write JSON for '{fname}': {e}")

if __name__ == "__main__":
    main()
