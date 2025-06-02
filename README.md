Skyscope Sentinel Intelligence presents
# AI-Driven OS Enhancer for Debian
Developer Casey Jay Topojani


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
    *   Includes syntax checking for Bash scripts (using `bash -n`) and Python scripts (using `python -m py_compile`). If a syntax error is detected after a modification, the change is automatically rolled back.
    *   Can create new files (e.g., new scripts).
*   **Rollback:** Can restore files from backups if an operation fails or leads to instability.
*   **Command Execution:** Can execute system commands and AI-generated scripts. Supports direct execution (with strong warnings about risks) or containerized execution via Docker (if Docker is available and the "DOCKER" sandbox level is selected) for enhanced isolation of script execution.
*   **Orchestration:** Manages the cycle of analysis, planning, approval, application, and monitoring.
*   **Human Approval Workflow:** Prompts for human confirmation for changes based on configurable risk/impact thresholds.
*   **Basic System Health Monitoring:** Includes a rudimentary system stability score and can trigger human intervention alerts.
	@@ -47,6 +47,7 @@ This project translates a detailed pseudo code specification into a functional P
    *   Ensure the Ollama API endpoint (`http://localhost:11434` by default) is accessible from where you run the application.
*   **Debian-based System:** The system analysis tools (`lsb_release`, `dpkg-query`, `systemctl`) are designed for Debian-based systems (e.g., Debian, Ubuntu).
*   **Pip:** For installing Python package dependencies.
*   **Docker (Optional):** If you intend to use the "DOCKER" `sandbox_level` for executing AI-generated scripts in a containerized environment, Docker must be installed and running. The user executing the AI OS Enhancer application will need appropriate permissions to interact with the Docker daemon (e.g., by being a member of the `docker` group). If Docker is not available or not used, script execution will fall back to direct execution with associated risks.

## Setup and Installation
  
*   **`GITHUB_API_KEY`** (Optional): For features that interact with the GitHub API (planned for future development phases), you'll need to provide a GitHub Personal Access Token with appropriate permissions.
    Set this environment variable before running the application:
    `
    export GITHUB_API_KEY="ghp_YourGitHubPersonalAccessTokenHere"
   `
    For persistence, you can add this line to your shell's startup file (e.g., `~/.bashrc`, `~/.zshrc`). The application will function without this key, but GitHub-related capabilities will be disabled. Ensure the key has the necessary scopes (e.g., `public_repo` for reading public repositories, or more depending on planned features).

*   **`AIOS_OLLAMA_API_ENDPOINT`** (Optional): Overrides the default Ollama API endpoint.
    *   If not set, defaults to: `"http://localhost:11434/api/generate"`
    *   Example: `export AIOS_OLLAMA_API_ENDPOINT="http://my-ollama-server:11434/api/generate"`

*   **`AIOS_DEFAULT_MODEL`** (Optional): Overrides the default Ollama model used for analysis and generation.
    *   If not set, defaults to: `"qwen2.5vl"` (or as specified in `config.py`)
    *   Example: `export AIOS_DEFAULT_MODEL="llama3"`

*   **`AIOS_HUMAN_APPROVAL_THRESHOLD`** (Optional): Defines the minimum severity level that requires human approval for applying changes.
    *   Valid values: `"LOW"`, `"MEDIUM"`, `"HIGH"`.
    *   If not set, defaults to: `"HIGH"` (meaning only high-risk/impact changes require approval, or medium risk with significant impact).
    *   `LOW`: Almost all changes require approval.
    *   `MEDIUM`: Medium and High risk/impact changes require approval.
    *   `HIGH`: Only High risk changes (or Medium risk with Significant impact) require approval.
    *   Example: `export AIOS_HUMAN_APPROVAL_THRESHOLD="MEDIUM"`

## How to Run

You can run the application as a Python module from the project's root directory from `https://github.com/skyscope-sentinel/Skyscope-Debian-AI-OS`
`
python -m ai_os_enhancer.main
`

The application will start, initialize, and begin its enhancement cycles. Follow the console output for logs and any prompts for human approval.

## Logging

*   Detailed logs are written to: `data/db/logs/ai_os_enhancer.log` (relative to the project root).
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
        *   For Bash scripts, a syntax check (`bash -n`) is performed. For Python scripts, a syntax check (`python -m py_compile`) is performed. If a syntax check fails, the change is automatically rolled back from the backup.
5.  **Monitoring:**
    *   `SystemStateAnalyzer` checks basic system health indicators.
    *   The system stability score is updated. If it drops critically low, human intervention is flagged.
6.  **Loop or Pause:**
    *   If human intervention is required, the system pauses and waits for acknowledgment.
    *   Otherwise, it sleeps for a configured interval before starting a new cycle.

## Disclaimer

This software is a research exploration into AI-driven system administration. It is not intended for use on production systems. The developers bear no responsibility for any outcomes resulting from its use or misuse. Always prioritize safety, backups, and thorough human review when dealing with automated system modifications.
