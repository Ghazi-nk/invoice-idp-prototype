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
USER_PROMPT_FILE = EXTRACT_USER_PROMPT

# =============================================================================
# --- API Configuration ---
# =============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat" if OLLAMA_BASE_URL else None

# =============================================================================
# --- File storage paths ---
# =============================================================================
# Temporary directory for processing
TMP_DIR = str(PROJECT_ROOT / 'app' / 'results' / 'tmp')

# =============================================================================
# --- Benchmark Configuration ---
# =============================================================================
# Benchmark data directories - TEMPORARILY CHANGED FOR NEW DATASET
LABELS_DIR = str(PROJECT_ROOT / 'app' / 'benchmark' / 'labels')
INVOICES_DIR = str(PROJECT_ROOT / 'app' / 'benchmark' / 'invoices')

# =============================================================================
# --- Sample files for testing ---
# =============================================================================
# Sample files for development and testing
SAMPLES_DIR = RESOURCES_DIR / 'samples'
SAMPLE_PDF_PATH = str(SAMPLES_DIR / 'BMRE-01.pdf')
SAMPLE_PNG_PATH = str(SAMPLES_DIR / 'BRE-03.png')
# =============================================================================
# --- Directory Creation ---
# =============================================================================
# Ensure required directories exist
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# --- Critical Configuration Validation ---
# =============================================================================
critical_vars = {
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    "OLLAMA_MODEL": OLLAMA_MODEL,
}

for var_name, value in critical_vars.items():
    if not value:
        print(f"Error: Critical configuration variable {var_name} is not set.", file=sys.stderr)
        print(f"Please set {var_name} in your .env file or environment variables.", file=sys.stderr)
        sys.exit(1)

# Validate CHAT_ENDPOINT after OLLAMA_BASE_URL validation
if not CHAT_ENDPOINT:
    print("Error: Could not construct CHAT_ENDPOINT from OLLAMA_BASE_URL.", file=sys.stderr)
    sys.exit(1)

# Check that prompt files exist
for prompt_file in [EXTRACT_SYSTEM_PROMPT, EXTRACT_USER_PROMPT, PDF_QUERY_SYSTEM_PROMPT, PDF_QUERY_USER_PROMPT]:
    if not os.path.exists(prompt_file):
        print(f"Warning: Prompt file not found at {prompt_file}", file=sys.stderr)

# Validation for benchmark and sample files
if not Path(LABELS_DIR).exists():
    print(f"Warning: Labels directory not found at {LABELS_DIR}. Benchmark functionality may be limited.", file=sys.stderr)
if not Path(INVOICES_DIR).exists():
    print(f"Warning: Invoices directory not found at {INVOICES_DIR}. Benchmark functionality may be limited.", file=sys.stderr)
if not Path(SAMPLE_PDF_PATH).exists():
    print(f"Warning: Sample PDF not found at {SAMPLE_PDF_PATH}. Some testing functionality may be limited.", file=sys.stderr)
if not Path(SAMPLE_PNG_PATH).exists():
    print(f"Warning: Sample PNG not found at {SAMPLE_PNG_PATH}. Some testing functionality may be limited.", file=sys.stderr)
