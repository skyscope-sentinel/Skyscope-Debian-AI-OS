# ai_os_enhancer/config.py

import os
import pathlib

# --- Global Configuration & Constants ---

# Ollama Configuration
# OLLAMA_API_ENDPOINT: Specifies the API endpoint for the Ollama service.
# It can be overridden by the environment variable AIOS_OLLAMA_API_ENDPOINT.
# Defaults to "http://localhost:11434/api/generate" if the environment variable is not set.
OLLAMA_API_ENDPOINT = os.environ.get("AIOS_OLLAMA_API_ENDPOINT", "http://localhost:11434/api/generate")

# DEFAULT_MODEL: Defines the default Ollama model to be used for analysis.
# It can be overridden by the environment variable AIOS_DEFAULT_MODEL.
# Defaults to "qwen2.5vl" if the environment variable is not set.
DEFAULT_MODEL = os.environ.get("AIOS_DEFAULT_MODEL", "qwen2.5vl")

# Project Paths
# PROJECT_ROOT: Absolute path to the project's root directory.
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
# CONFIG_DATABASE_PATH: Path to the directory where configuration and database files are stored.
CONFIG_DATABASE_PATH = PROJECT_ROOT / "data" / "db"
# LOG_FILE_PATH: Path to the log file for the application.
LOG_FILE_PATH = CONFIG_DATABASE_PATH / "logs" / "ai_os_enhancer.log"
# BACKUP_BASE_PATH: Path to the directory where backups are stored.
BACKUP_BASE_PATH = CONFIG_DATABASE_PATH / "backups"

# Ensure data directories exist
os.makedirs(CONFIG_DATABASE_PATH / "logs", exist_ok=True)
os.makedirs(BACKUP_BASE_PATH, exist_ok=True)

# Analysis & Enhancement Behavior
# MAX_CONCURRENT_ANALYSES: Maximum number of concurrent analyses allowed.
MAX_CONCURRENT_ANALYSES = 3  # As per pseudo code

# HUMAN_APPROVAL_THRESHOLD: Defines the level of human approval required for critical changes.
# It can be overridden by the environment variable AIOS_HUMAN_APPROVAL_THRESHOLD.
# Valid values are "LOW", "MEDIUM", "HIGH".
# Defaults to "HIGH" if the environment variable is not set or an invalid value is provided.
_human_approval_threshold_env = os.environ.get("AIOS_HUMAN_APPROVAL_THRESHOLD", "HIGH").upper()
if _human_approval_threshold_env not in ["LOW", "MEDIUM", "HIGH"]:
    HUMAN_APPROVAL_THRESHOLD = "HIGH"
else:
    HUMAN_APPROVAL_THRESHOLD = _human_approval_threshold_env

# MIN_STABILITY_FOR_ENHANCEMENT: Minimum system stability score required for the orchestrator to attempt new enhancements.
# Can be overridden by environment variable AIOS_MIN_STABILITY_FOR_ENHANCEMENT.
_min_stability_env = os.environ.get("AIOS_MIN_STABILITY_FOR_ENHANCEMENT")
MIN_STABILITY_FOR_ENHANCEMENT = 60.0  # Default value
if _min_stability_env:
    try:
        MIN_STABILITY_FOR_ENHANCEMENT = float(_min_stability_env)
    except ValueError:
        print(f"Warning: Invalid value for AIOS_MIN_STABILITY_FOR_ENHANCEMENT: '{_min_stability_env}'. Using default: {MIN_STABILITY_FOR_ENHANCEMENT}")
        # Potentially log this warning as well if logger is available here

# CYCLE_INTERVAL_SECONDS: The time in seconds the orchestrator waits before starting a new enhancement cycle.
# Can be overridden by environment variable AIOS_CYCLE_INTERVAL_SECONDS.
_cycle_interval_env = os.environ.get("AIOS_CYCLE_INTERVAL_SECONDS")
CYCLE_INTERVAL_SECONDS = 300  # Default value (5 minutes)
if _cycle_interval_env:
    try:
        CYCLE_INTERVAL_SECONDS = int(_cycle_interval_env)
    except ValueError:
        print(f"Warning: Invalid value for AIOS_CYCLE_INTERVAL_SECONDS: '{_cycle_interval_env}'. Using default: {CYCLE_INTERVAL_SECONDS}")
        # Potentially log this warning

# Monitored script paths - this should be customized by the user.
# MONITORED_SCRIPTS_PATHS: A list of paths to scripts that the application should monitor.
# For now, it's an empty list. Users should add paths to their actual scripts.
MONITORED_SCRIPTS_PATHS = [
    # Example: "/opt/scripts/skyscope_sentinel_optimizer.sh",
    # Add the actual path to your Skyscope Sentinel script here if applicable
    # For development, you might want to add a path to a test script within the project
    # e.g., str(PROJECT_ROOT / "sample_scripts" / "test_script.sh")
]

# Other constants can be added here as needed.

# GitHub API Key (optional)
# GITHUB_API_KEY: Your GitHub Personal Access Token (PAT) for accessing GitHub APIs.
# It is read from the environment variable AIOS_GITHUB_API_KEY.
# This is optional and only needed if the application interacts with GitHub.
GITHUB_API_KEY = os.environ.get("AIOS_GITHUB_API_KEY")


if __name__ == '__main__':
    # For testing the paths
    print(f"Project Root: {PROJECT_ROOT}")
	@@ -43,3 +47,4 @@
    print(f"Backup Base Path: {BACKUP_BASE_PATH}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print(f"Human Approval Threshold: {HUMAN_APPROVAL_THRESHOLD}")
    print(f"Min Stability for Enhancement: {MIN_STABILITY_FOR_ENHANCEMENT}")
    print(f"Cycle Interval Seconds: {CYCLE_INTERVAL_SECONDS}")
    print(f"Monitored Scripts: {MONITORED_SCRIPTS_PATHS}")
    print(f"GitHub API Key: {'Loaded' if GITHUB_API_KEY else 'Not set / Not found in environment variables'}")
