# ai_os_enhancer/config.py

import os
import pathlib

# --- Global Configuration & Constants ---

# Ollama Configuration
OLLAMA_API_ENDPOINT = "http://localhost:11434/api/generate"  # Default Ollama API
DEFAULT_MODEL = "qwen2.5vl"  # As specified by the user in the issue, was "your_chosen_ollama_model"

# Project Paths
# Get the absolute path of the project root directory
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
CONFIG_DATABASE_PATH = PROJECT_ROOT / "data" / "db"
LOG_FILE_PATH = CONFIG_DATABASE_PATH / "logs" / "ai_os_enhancer.log"
BACKUP_BASE_PATH = CONFIG_DATABASE_PATH / "backups"

# Ensure data directories exist
os.makedirs(CONFIG_DATABASE_PATH / "logs", exist_ok=True)
os.makedirs(BACKUP_BASE_PATH, exist_ok=True)

# Analysis & Enhancement Behavior
MAX_CONCURRENT_ANALYSES = 3  # As per pseudo code
HUMAN_APPROVAL_THRESHOLD = "HIGH"  # Options: "LOW", "MEDIUM", "HIGH" - for critical changes

# Monitored script paths - this should be customized by the user.
# For now, it's an empty list. Users should add paths to their actual scripts.
MONITORED_SCRIPTS_PATHS = [
    # Example: "/opt/scripts/skyscope_sentinel_optimizer.sh",
    # Add the actual path to your Skyscope Sentinel script here if applicable
    # For development, you might want to add a path to a test script within the project
    # e.g., str(PROJECT_ROOT / "sample_scripts" / "test_script.sh")
]

# Other constants can be added here as needed.

if __name__ == '__main__':
    # For testing the paths
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Config DB Path: {CONFIG_DATABASE_PATH}")
    print(f"Log File Path: {LOG_FILE_PATH}")
    print(f"Backup Base Path: {BACKUP_BASE_PATH}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print(f"Monitored Scripts: {MONITORED_SCRIPTS_PATHS}")
