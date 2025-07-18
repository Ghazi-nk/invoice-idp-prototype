import os
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# =============================================================================
# --- Base resource paths ---
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
RESOURCES_DIR = PROJECT_ROOT / 'resources'

# =============================================================================
# --- Prompt files for API endpoints ---
# =============================================================================

# Extract prompts (for invoice extraction endpoint)
EXTRACT_PROMPTS_DIR = RESOURCES_DIR / 'extract_prompts'
EXTRACT_SYSTEM_PROMPT = str(EXTRACT_PROMPTS_DIR / 'system_prompt')
EXTRACT_USER_PROMPT = str(EXTRACT_PROMPTS_DIR / 'user_prompt')

# PDF Query prompts (for pdf-query endpoint)
PDF_QUERY_PROMPTS_DIR = RESOURCES_DIR / 'pdf_query_prompts'
PDF_QUERY_SYSTEM_PROMPT = str(PDF_QUERY_PROMPTS_DIR / 'system_prompt')
PDF_QUERY_USER_PROMPT = str(PDF_QUERY_PROMPTS_DIR / 'user_prompt')

# Legacy/compatibility variables - will now point to extract_prompts
SYSTEM_PROMPT_FILE = EXTRACT_SYSTEM_PROMPT
USER_PROMPT_OCR_FILE = EXTRACT_USER_PROMPT

# =============================================================================
# --- API Configuration ---
# =============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat" if OLLAMA_BASE_URL else None

# =============================================================================
# --- File storage paths ---
# =============================================================================
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
TMP_DIR = os.getenv("TMP_DIR")

# =============================================================================
# --- OCR Configuration ---
# =============================================================================
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

# =============================================================================
# --- Benchmark Configuration ---
# =============================================================================
LABELS_DIR = os.getenv("LABELS_DIR")
INVOICES_DIR = os.getenv("INVOICES_DIR")

# =============================================================================
# --- Sample files for testing ---
# =============================================================================
SAMPLE_PDF_PATH = os.getenv("SAMPLE_PDF_PATH")
SAMPLE_PNG_PATH = os.getenv("SAMPLE_PNG_PATH")

# =============================================================================
# --- Critical Configuration Validation ---
# =============================================================================
critical_vars = {
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    "OLLAMA_MODEL": OLLAMA_MODEL,
    "CHAT_ENDPOINT": CHAT_ENDPOINT,
    "INPUT_DIR": INPUT_DIR,
    "OUTPUT_DIR": OUTPUT_DIR,
    "TMP_DIR": TMP_DIR,
    "TESSERACT_CMD": TESSERACT_CMD
}

for var_name, value in critical_vars.items():
    if not value:
        print(f"Error: Critical configuration variable {var_name} is not set.", file=sys.stderr)
        sys.exit(1)

# Check that prompt files exist
for prompt_file in [EXTRACT_SYSTEM_PROMPT, EXTRACT_USER_PROMPT, PDF_QUERY_SYSTEM_PROMPT, PDF_QUERY_USER_PROMPT]:
    if not os.path.exists(prompt_file):
        print(f"Warning: Prompt file not found at {prompt_file}", file=sys.stderr)

# Optional variables used for benchmarking and demos
optional_vars = {
    "SAMPLE_PDF_PATH": SAMPLE_PDF_PATH,
    "LABELS_DIR": LABELS_DIR,
    "INVOICES_DIR": INVOICES_DIR
}

for var_name, value in optional_vars.items():
    if not value:
        print(f"Warning: Optional variable {var_name} is not set. Some functionality may be limited.", file=sys.stderr)
