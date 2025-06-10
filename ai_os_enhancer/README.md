# AI-Driven OS Enhancer for Debian

## Overview

The AI-Driven OS Enhancer is a Python application designed to conceptualize how an AI, powered by a Large Language Model (LLM) like those available through Ollama, could analyze and suggest enhancements for a Debian-based operating system. It can identify areas for improvement in system configuration files and user-provided scripts, propose changes, and (with human approval) apply these changes.

**Current Status: Experimental Proof-of-Concept**

This project translates a detailed pseudo code specification into a functional Python application. It implements the core logic for system analysis, LLM interaction, and change application, but it requires careful setup, configuration, and human oversight.

## !!! CRITICAL WARNINGS !!!

*   **EXPERIMENTAL SOFTWARE:** This is highly experimental software. It interacts with your operating system, modifies files, and executes commands. **USE WITH EXTREME CAUTION.**
*   **POTENTIAL FOR SYSTEM DAMAGE:** AI-generated suggestions or actions, if applied incorrectly or without thorough understanding, could potentially damage your system, lead to instability, security vulnerabilities, or data loss.
*   **RUN IN A SANDBOXED ENVIRONMENT:** It is **STRONGLY RECOMMENDED** to run this application only in a dedicated, isolated, and non-critical virtual machine or containerized environment (e.g., Docker, a dedicated VM). Do **NOT** run this on a production system or any system with important data without full backups and understanding of the risks.
*   **HUMAN APPROVAL IS KEY:** The system includes a human approval step for significant changes. Always review proposed changes carefully before approving. Do not blindly trust AI suggestions.
*   **OLLAMA MODEL DEPENDENCY:** The quality and safety of suggestions heavily depend on the chosen Ollama model, its capabilities, and its alignment. Ensure you are using a reliable and well-understood model.
*   **NO GUARANTEES:** This software is provided "as-is" without any warranties of any kind. The developers are not responsible for any damage caused by its use.

## Features

*   **System Analysis:** Gathers information about the Debian system (OS version, installed packages, service status, file details).
*   **Configuration & Script Scanning:** Identifies key system configuration areas and user-specified scripts for analysis.
*   **Ollama Integration:**
    *   Queries an Ollama-compatible LLM to analyze system items (configs/scripts).
    *   Leverages the LLM to conceive enhancement strategies based on analyses.
    *   Can request the LLM to generate code or modifications.
*   **Enhancement Application:**
    *   Backs up files before modification.
    *   Applies text-based changes to configuration files.
    *   Applies modifications to scripts (currently supports basic operations like full replacement, append/prepend, and simple regex-based function replacement for Bash).
    *   Includes basic syntax checking for Bash script modifications.
    *   Can create new files (e.g., new scripts).
*   **Rollback:** Can restore files from backups if an operation fails or leads to instability.
*   **Command Execution:** Can execute system commands and AI-generated scripts. Supports direct execution (with strong warnings about risks) or containerized execution via Docker (if Docker is available and the "DOCKER" sandbox level is selected) for enhanced isolation of script execution.
*   **Orchestration:** Manages the cycle of analysis, planning, approval, application, and monitoring.
*   **Human Approval Workflow:** Prompts for human confirmation for changes based on configurable risk/impact thresholds.
*   **Basic System Health Monitoring:** Includes a rudimentary system stability score and can trigger human intervention alerts.
*   **Configurable:** Key parameters like Ollama endpoint, model name, monitored paths, and approval thresholds are configurable.
*   **Logging:** Detailed logging of operations to `data/db/logs/ai_os_enhancer.log`.

## Prerequisites

*   **Python:** Python 3.8 or newer recommended.
*   **Ollama:** A running Ollama instance.
    *   You must have pulled the LLM model specified in `ai_os_enhancer/config.py` (default is `qwen2.5vl`, as per the original issue request). You can pull models using `ollama pull <model_name>`.
    *   Ensure the Ollama API endpoint (`http://localhost:11434` by default) is accessible from where you run the application.
*   **Debian-based System:** The system analysis tools (`lsb_release`, `dpkg-query`, `systemctl`) are designed for Debian-based systems (e.g., Debian, Ubuntu).
*   **Pip:** For installing Python package dependencies.
*   **Docker (Optional):** If you intend to use the "DOCKER" `sandbox_level` for executing AI-generated scripts in a containerized environment, Docker must be installed and running. The user executing the AI OS Enhancer application will need appropriate permissions to interact with the Docker daemon (e.g., by being a member of the `docker` group). If Docker is not available or not used, script execution will fall back to direct execution with associated risks.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ai-os-enhancer
    ```
    (Replace `<repository_url>` with the actual URL of this repository)

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The primary external Python dependency is `requests`.
    ```bash
    pip install requests
    ```
    (If other dependencies are added later, they should be included in a `requirements.txt` file.)

## Configuration

Before running, review and customize `ai_os_enhancer/config.py`:

*   **`OLLAMA_API_ENDPOINT`**: Ensure this points to your Ollama API.
*   **`DEFAULT_MODEL`**: Set this to the Ollama model you have pulled and wish to use (e.g., `qwen2.5vl`).
*   **`CONFIG_DATABASE_PATH`**: Defines where logs and backups are stored (defaults to `ai_os_enhancer/data/db`).
*   **`HUMAN_APPROVAL_THRESHOLD`**: Adjust the sensitivity for requiring human approval ("LOW", "MEDIUM", "HIGH").
    *   `LOW`: All changes require approval.
    *   `MEDIUM`: Medium/High risk or Significant impact changes require approval.
    *   `HIGH`: Only High risk or Significant impact changes require approval.
*   **`MONITORED_SCRIPTS_PATHS`**: **Crucial step!** This is an empty list by default. You **must** populate this list with absolute paths to any scripts you want the AI to analyze and potentially modify.
    Example:
    ```python
    MONITORED_SCRIPTS_PATHS = [
        "/home/user/myscripts/backup_script.sh", # Example of an external script
        # Example for a script within the project's example area:
        str(PROJECT_ROOT / "examples_and_test_scripts" / "sample_scripts_for_orchestrator" / "orchestrator_test_script.sh")
    ]
    ```
*   **`AIOS_GITHUB_API_KEY`** (Optional): For features that interact with the GitHub API (planned for future development phases), you'll need to provide a GitHub Personal Access Token with appropriate permissions.
    Set this environment variable before running the application:
    ```bash
    export AIOS_GITHUB_API_KEY="ghp_YourGitHubPersonalAccessTokenHere"
    ```
    For persistence, you can add this line to your shell's startup file (e.g., `~/.bashrc`, `~/.zshrc`). The application will function without this key, but GitHub-related capabilities will be disabled. Ensure the key has the necessary scopes (e.g., `public_repo` for reading public repositories, or more depending on planned features).

## How to Run

You can run the application as a Python module from the project's root directory (the one containing the `ai_os_enhancer` folder):

```bash
python -m ai_os_enhancer.main
```

Alternatively, depending on your `PYTHONPATH` setup, you might be able to run:

```bash
python ai_os_enhancer/main.py
```

The application will start, initialize, and begin its enhancement cycles. Follow the console output for logs and any prompts for human approval.

## Code Examples and Test Scripts

The `ai_os_enhancer/examples_and_test_scripts/` directory contains sample scripts and test configurations that were used during development or can serve as examples for understanding certain features (e.g., how `EnhancementApplier` might modify files, or sample scripts for the `Orchestrator` to analyze if configured in `MONITORED_SCRIPTS_PATHS`). You can explore this directory to see concrete examples.

## Logging

*   Detailed logs are written to: `ai_os_enhancer/data/db/logs/ai_os_enhancer.log`
*   Console output provides a summary of key actions and alerts. Log level for console and file can be adjusted in `ai_os_enhancer/logger_setup.py`.

## How It Works

The Orchestrator module drives the core logic in a loop:

1.  **Initialization:** Sets up paths, checks OS version.
2.  **Analysis Phase:**
    *   `SystemStateAnalyzer` lists key configuration areas and monitored scripts.
    *   Content of these items is read.
    *   `OllamaInterface` sends each item to the LLM for analysis, asking for potential issues and enhancement ideas in a structured JSON format.
3.  **Conception Phase:**
    *   A system snapshot and all analysis results are sent to the LLM via `OllamaInterface`.
    *   The LLM is tasked to conceive an overall strategy and a prioritized list of specific enhancements. This includes details about the proposed change, justification, risk, and impact.
4.  **Application Phase:**
    *   Each proposed enhancement is considered one by one.
    *   **Human approval** is requested if the enhancement meets the configured threshold for risk/impact.
    *   If approved:
        *   The target file is backed up by `EnhancementApplier`.
        *   If the LLM plan indicates code/content needs to be generated (e.g., a new function body), `OllamaInterface` requests this from the LLM.
        *   `EnhancementApplier` attempts to apply the change (e.g., modify config text, patch script, create new file).
        *   For Bash scripts, a syntax check (`bash -n`) is performed. If it fails, the change is automatically rolled back from the backup.
5.  **Monitoring:**
    *   `SystemStateAnalyzer` checks basic system health indicators.
    *   The system stability score is updated. If it drops critically low, human intervention is flagged.
6.  **Loop or Pause:**
    *   If human intervention is required, the system pauses and waits for acknowledgment.
    *   Otherwise, it sleeps for a configured interval before starting a new cycle.

## Disclaimer

This software is a research exploration into AI-driven system administration. It is not intended for use on production systems. The developers bear no responsibility for any outcomes resulting from its use or misuse. Always prioritize safety, backups, and thorough human review when dealing with automated system modifications.
```
