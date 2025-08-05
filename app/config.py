"""
Zentrale Konfigurationsdatei für die IDP-Pipeline.

Diese Datei definiert alle wichtigen Pfade, URLs und Konfigurationsparameter
für das Intelligent Document Processing System. Die Konfiguration wird aus
Umgebungsvariablen geladen und stellt sinnvolle Defaults bereit.

Umgebungsvariablen:
- OLLAMA_BASE_URL: URL der Ollama-Instanz (z.B. http://localhost:11434)
- OLLAMA_MODEL: Name des zu verwendenden LLM-Modells (z.B. llama3.1:8b)

Autor: Ghazi Nakkash
Projekt: Konzeption und prototypische Implementierung einer KI-basierten und 
         intelligenten Dokumentenverarbeitung im Rechnungseingangsprozess
Institution: Hochschule für Technik und Wirtschaft Berlin
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from app.logging_config import config_logger

load_dotenv()

# =============================================================================
# --- Base resource paths ---
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
RESOURCES_DIR = PROJECT_ROOT / 'resources'
PROMPTS_DIR = RESOURCES_DIR / 'prompts'
# =============================================================================
# --- Prompt files for API endpoints ---
# =============================================================================

# Extract prompts (for invoice extraction endpoint)
EXTRACT_PROMPTS_DIR = PROMPTS_DIR /'extract_prompts'
EXTRACT_SYSTEM_PROMPT = str(EXTRACT_PROMPTS_DIR / 'system_prompt')
EXTRACT_USER_PROMPT = str(EXTRACT_PROMPTS_DIR / 'user_prompt')

# PDF Query prompts (for pdf-query endpoint)
PDF_QUERY_PROMPTS_DIR = PROMPTS_DIR /'pdf_query_prompts'
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
TMP_DIR = str(PROJECT_ROOT / 'app' / 'tmp')

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
        config_logger.error(f"Critical configuration variable {var_name} is not set.")
        config_logger.error(f"Please set {var_name} in your .env file or environment variables.")
        sys.exit(1)

# Validate CHAT_ENDPOINT after OLLAMA_BASE_URL validation
if not CHAT_ENDPOINT:
    config_logger.error("Could not construct CHAT_ENDPOINT from OLLAMA_BASE_URL.")
    sys.exit(1)

# Check that prompt files exist
for prompt_file in [EXTRACT_SYSTEM_PROMPT, EXTRACT_USER_PROMPT, PDF_QUERY_SYSTEM_PROMPT, PDF_QUERY_USER_PROMPT]:
    if not os.path.exists(prompt_file):
        config_logger.warning(f"Prompt file not found at {prompt_file}")

# Validation for benchmark and sample files
if not Path(LABELS_DIR).exists():
    config_logger.warning(f"Labels directory not found at {LABELS_DIR}. Benchmark functionality may be limited.")
if not Path(INVOICES_DIR).exists():
    config_logger.warning(f"Invoices directory not found at {INVOICES_DIR}. Benchmark functionality may be limited.")
if not Path(SAMPLE_PDF_PATH).exists():
    config_logger.warning(f"Sample PDF not found at {SAMPLE_PDF_PATH}. Some testing functionality may be limited.")
if not Path(SAMPLE_PNG_PATH).exists():
    config_logger.warning(f"Sample PNG not found at {SAMPLE_PNG_PATH}. Some testing functionality may be limited.")
