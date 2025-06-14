Skyscope Sentinel Intelligence presents
# Skyscope Sentinel AI OS
Developer Casey Jay Topojani


## Overview

The AI-Driven OS Enhancer is a Python application designed to conceptualize how an AI, powered by a Large Language Model (LLM) like those available through Ollama, could analyze and suggest enhancements for a Debian-based operating system. It can identify areas for improvement in system configuration files and user-provided scripts, propose changes, and (with human approval) apply these changes.

**Current Status: Experimental Proof-of-Concept**

This project translates a detailed pseudo code specification into a functional Python application. It implements the core logic for system analysis, LLM interaction, and change application, but it requires careful setup, configuration, and human oversight.

![Skyscope Sentinel Boot Splash](./Skyscope-Sentinel-AI-OS.png)


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
    *   Supports role-based configurations for more specialized LLM interactions (see "AI Personas (Roles)" section), including translation of natural language to shell commands.
*   **Enhancement Application:**
    *   Backs up files before modification.
    *   Applies text-based changes to configuration files.
    *   Applies modifications to scripts (currently supports basic operations like full replacement, append/prepend, and simple regex-based function replacement for Bash).
    *   Includes syntax checking for Bash scripts (using `bash -n`) and Python scripts (using `python -m py_compile`). If a syntax error is detected after a modification, the change is automatically rolled back.
    *   Can create new files (e.g., new scripts).
*   **Rollback:** Can restore files from backups if an operation fails or leads to instability.
*   **Command Execution:**
    *   Can execute system commands and AI-generated scripts.
    *   Supports **direct execution** (with strong warnings about inherent risks).
    *   Offers **containerized execution for script content** via Docker if `sandbox_level` is set to `"DOCKER"` in `execute_command_or_script` and Docker is available.
        *   This method attempts to run script content (e.g., Bash or Python scripts) by dynamically building a temporary Docker image and running a container.
        *   Default Docker images used:
            *   Shell scripts (`bash`, `sh`): `debian:stable-slim` (configurable via `AIOS_DOCKER_SHELL_IMAGE`)
            *   Python scripts: `python:3.9-slim` (configurable via `AIOS_DOCKER_PYTHON_IMAGE`)
        *   If Docker is requested for direct command strings (not script content), it currently falls back to direct execution with a warning.
    *   Approved AI-generated shell commands (originating from 'shell_task' proposals) can be executed, with optional Docker sandboxing as per the script execution settings.
*   **Orchestration:**
    *   Manages the cycle of analysis, planning, approval, application, and monitoring.
    *   Can now process 'shell_task' proposals from the AI strategist, which involves generating specific shell commands (using a dedicated AI role) for approval and subsequent execution.
*   **Human Approval Workflow:** Prompts for human confirmation for changes based on configurable risk/impact thresholds, including detailed display of proposed shell commands and their assessed risks.
*   **Basic System Health Monitoring:** Includes a rudimentary system stability score and can trigger human intervention alerts.
enhancements-suite-1
*   **Multi-Agent Foundations:** Includes initial support for managing multiple AI agents with distinct roles and conversation histories (see "Multi-Agent Architecture" section).
=======
main

## AI Personas (Roles)

To allow for more specialized and configurable LLM behaviors, the AI OS Enhancer uses a role-based system. Each "role" defines a specific persona for the AI, including its system prompt, the Ollama model to use, relevant knowledge base keywords, and the expected output format for its responses. This allows tailoring the AI's responses and capabilities to specific tasks like system analysis, strategy conception, or code generation.

### Role File Location and Naming

*   Role configurations are defined in YAML files.
*   These files must be located in the `ai_os_enhancer/roles/` directory within the project.
*   The filename convention is `<role_name>.yaml` (e.g., `generic_system_item_analyzer.yaml` for a role named `GenericSystemItemAnalyzer`). The `role_name` in the YAML file (case-sensitive) should correspond to the base name of the file (which is typically made lowercase).

### Role YAML Structure

Each role YAML file should follow this structure:

```yaml
enhancements-suite-1
role_name: MyCustomRole        # Mandatory, unique name for the role (case-sensitive)
author: "YourName/TeamName"  # Optional, creator of the role definition
version: "1.0.0"             # Optional, version of the role definition
description: "A human-readable description of what this role does." # Optional
tags:                        # Optional, list of keywords for categorization
  - "custom"
  - "specific_domain"
system_prompt: | # Mandatory, the detailed system prompt for the LLM
  You are a specialized AI assistant for [specific domain].
  Your primary task is to {task_description}.
  Analyze the {item_type} at '{item_path}'.
  Always follow these guidelines...
=======
role_name: MyCustomRole # Mandatory, unique name for the role (case-sensitive)
description: "A human-readable description of what this role does." # Optional
system_prompt: | # Mandatory, the detailed system prompt for the LLM
  You are a specialized AI assistant for [specific domain].
  Your primary task is to [describe task].
  Always follow these guidelines:
  1. Guideline one.
  2. Guideline two.
  When responding to [specific type of input], provide [specific type of output].
main
model_name: "ollama_model_name:tag" # Optional, overrides the global default model for this role
knowledge_base_keywords: # Optional, list of keywords to load relevant documents
  - "custom_topic_1"
  - "specific_tool_guide"
output_format: "json" # Optional, e.g., "json" or "text". Defines expected LLM output style.
enhancements-suite-1
expected_llm_output_keys: # Optional, list of keys expected in JSON output
  - "top_level_key"
  - "parent_object.nested_key"
  - "another_parent.child.grandchild_key"
```

**Key Fields:**

*   `role_name` (string, mandatory): The unique identifier for the role. This is used to load the role.
*   `author` (string, optional): The creator or maintainer of the role definition (e.g., "SkyscopeAI/Jules", "YourName").
*   `version` (string, optional): The version of the role definition (e.g., "1.0.0", "1.1.0-alpha"). Useful for tracking changes to role prompts and structures.
*   `description` (string, optional): Helps users understand the role's purpose.
*   `tags` (list of strings, optional): Keywords or tags for categorizing or filtering roles (e.g., `["analysis", "security", "python_coding"]`).
*   `system_prompt` (string, mandatory): The core instruction given to the LLM, defining its persona, task, and any specific output requirements. Multi-line YAML strings (using `|`) are recommended for readability.
    *   The `system_prompt` can include dynamic placeholders (e.g., `{item_path}`, `{item_type}`, `{language}`, `{task_description}`, `{existing_code_context}`). These placeholders are automatically replaced with actual values by the system when the role is used, allowing for more context-specific prompts. For example, a role for analyzing files might use: `Analyze the {item_type} at '{item_path}'.`
*   `model_name` (string, optional): If specified, this role will use this Ollama model, overriding the global default model (`AIOS_DEFAULT_MODEL` or the default in `config.py`).
*   `knowledge_base_keywords` (list of strings, optional): A list of keywords. If provided, the system will attempt to load corresponding `.txt` files from the `knowledge_base/keywords/` directory to augment the LLM's context when this role is active.
*   `output_format` (string, optional): Specifies the expected format of the LLM's response. Common values are `"json"` (if structured data is required, in which case the `system_prompt` should detail the JSON schema) or `"text"` (for free-form textual output). The default behavior in the application might vary by function if this is not set (e.g., analysis functions typically default to expecting JSON).
*   `expected_llm_output_keys` (list of strings, optional): If `output_format` is `json`, this field can list top-level or dot-notated nested keys (e.g., `"risk_assessment.risk_level"`) that the LLM's JSON response is expected to contain. The system will validate the presence of these keys, helping to ensure the LLM adheres to the desired output structure.
=======
                      # If "json", specific JSON structures are usually defined in the system_prompt.
                      # Defaults vary by function if not set here (e.g., analysis expects JSON).
```

**Key Fields:**
*   `role_name` (string, mandatory): The unique identifier for the role. This is used to load the role.
*   `description` (string, optional): Helps users understand the role's purpose.
*   `system_prompt` (string, mandatory): The core instruction given to the LLM, defining its persona, task, and any specific output requirements. Multi-line YAML strings (using `|`) are recommended for readability.
*   `model_name` (string, optional): If specified, this role will use this Ollama model, overriding the global default model (`AIOS_DEFAULT_MODEL` or the default in `config.py`).
*   `knowledge_base_keywords` (list of strings, optional): A list of keywords. If provided, the system will attempt to load corresponding `.txt` files from the `knowledge_base/keywords/` directory to augment the LLM's context when this role is active.
*   `output_format` (string, optional): Specifies the expected format of the LLM's response. Common values are `"json"` (if structured data is required, in which case the `system_prompt` should detail the JSON schema) or `"text"` (for free-form textual output). The default behavior in the application might vary by function if this is not set (e.g., analysis functions typically default to expecting JSON).
main

### Default Roles

The system currently includes the following predefined roles:

enhancements-suite-1
*   **`GenericSystemItemAnalyzer`**: Used for the general analysis of system configuration files and scripts. Its `system_prompt` uses `{item_path}` and `{item_type}` placeholders to guide the LLM. (See `ai_os_enhancer/roles/generic_system_item_analyzer.yaml`)
*   **`EnhancementStrategist`**: Used for conceiving the overall enhancement strategy from multiple analyses and a system snapshot. Its `system_prompt` instructs the LLM to produce a prioritized list of enhancement tasks, which can now include `shell_task` types for system-level operations. (See `ai_os_enhancer/roles/enhancement_strategist.yaml`)
*   **`ShellCommandGenerator`**: Translates natural language tasks (often proposed by the `EnhancementStrategist` as a `shell_task`) into specific shell command suggestions, including a structured risk assessment, prerequisites, safety notes, and alternatives. (See `ai_os_enhancer/roles/shell_command_generator.yaml`)
*   **`PythonCodeGenerator`**: Specializes in generating Python code snippets based on a task description. Uses placeholders for `{language}`, `{task_description}`, and `{existing_code_context}` in its prompt. Expects JSON output with a `code` field. (See `ai_os_enhancer/roles/python_code_generator.yaml`)
=======
*   **`GenericSystemItemAnalyzer`**: Used for the general analysis of system configuration files and scripts. Its `system_prompt` guides the LLM to identify issues and suggest enhancements for individual items. (See `ai_os_enhancer/roles/generic_system_item_analyzer.yaml`)
*   **`EnhancementStrategist`**: Used for conceiving the overall enhancement strategy from multiple analyses and a system snapshot. Its `system_prompt` instructs the LLM to produce a prioritized list of enhancement tasks, which can now include `shell_task` types for system-level operations. (See `ai_os_enhancer/roles/enhancement_strategist.yaml`)
*   **`ShellCommandGenerator`**: Translates natural language tasks (often proposed by the `EnhancementStrategist` as a `shell_task`) into specific shell command suggestions, including a structured risk assessment, prerequisites, safety notes, and alternatives. (See `ai_os_enhancer/roles/shell_command_generator.yaml`)
main

These files serve as examples of how to structure role configurations.

### Adding Custom Roles

Users can extend the system by adding their own roles:
1.  Create a new YAML file in the `ai_os_enhancer/roles/` directory (e.g., `mycoderrole.yaml`).
2.  Define the role using the structure described above, ensuring `role_name` inside the YAML matches how you intend to call it.
3.  To use this new role, you would typically modify the Python code (e.g., in the `Orchestrator` or other modules) that calls an `ollama_interface` function (like `analyze_system_item`, `conceive_enhancement_strategy`, `generate_code_or_modification`, or `generate_shell_command`) to pass your new `role_name` as an argument.

#### Example Role: Shell Command Generator (`ShellCommandGenerator`)

The system includes a specialized role named `ShellCommandGenerator` (defined in `ai_os_enhancer/roles/shell_command_generator.yaml`). This role is designed to:

*   Translate natural language task descriptions (e.g., "install the htop package", "list all files modified in the last 24 hours") into appropriate shell commands for a Debian 13 environment.
*   Return a structured JSON output. This output includes not only the suggested command but also a detailed `risk_assessment` (with `risk_level`, `operation_type`, `requires_privileges`), a list of `prerequisites`, optional `setup_commands` to meet those prerequisites, important `safety_notes`, potential `clarifications_needed` if the task is ambiguous, and `alternatives` to the suggested command.

**Example JSON Output from `ShellCommandGenerator`:**
```json
{
  "task_description": "Install the nginx package and ensure it starts on boot.",
  "generated_command": "sudo apt update && sudo apt install -y nginx && sudo systemctl enable nginx && sudo systemctl start nginx",
  "risk_assessment": {
    "risk_level": "Medium",
    "operation_type": "PackageManagementWrite",
    "requires_privileges": "root"
  },
  "prerequisites": ["Internet access to Debian repositories."],
  "setup_commands": [],
  "safety_notes": [
    "Installs software from Debian repositories.",
    "Modifies system services to start on boot.",
    "The '-y' flag automatically confirms installation."
  ],
  "clarifications_needed": [],
  "alternatives": [
    "Run commands separately for more control.",
    "Use 'apt-get' instead of 'apt' for older systems (though 'apt' is preferred on modern Debian)."
  ]
}
```

**Important:** The `generate_shell_command()` function in `ai_os_enhancer/ollama_interface.py` utilizes this role. When the `EnhancementStrategist` proposes a `shell_task`, the Orchestrator uses this function to get the detailed command proposal. The shell commands generated are suggestions for review and are **not automatically executed** without passing through the Orchestrator's approval workflow. If approved, the Orchestrator then instructs the `EnhancementApplier` to execute the command.

## Prerequisites

*   **Python:** Version 3.10 or higher recommended.
*   **Ollama:** An Ollama installation with one or more models pulled (e.g., `ollama pull llama3`, `ollama pull qwen2.5vl`).
    *   Ensure the Ollama API endpoint (`http://localhost:11434` by default) is accessible from where you run the application.
*   **PyYAML:** The `PyYAML` package is required for parsing role configuration files. Install it via pip:
    ```bash
    pip install PyYAML
    ```
*   **Debian-based System:** The system analysis tools (`lsb_release`, `dpkg-query`, `systemctl`) are designed for Debian-based systems (e.g., Debian, Ubuntu).
*   **Pip:** For installing Python package dependencies (like `requests`, `PyYAML`).
*   **Docker (Optional for sandboxed script execution):** If you intend to use the `"DOCKER"` `sandbox_level` for executing AI-generated *script content* in a containerized environment, Docker must be installed and running. The user executing the AI OS Enhancer application will typically need to be part of the `docker` group or have equivalent permissions to interact with the Docker daemon. If Docker is not available or not used for script execution, it will fall back to direct execution with associated risks.

## Setup and Installation
  
### Environment Variables
The application uses environment variables for certain configurations. These can be set in your shell before running the application.

*   **`GITHUB_API_KEY`** (Optional): For features that interact with the GitHub API (planned for future development phases), you'll need to provide a GitHub Personal Access Token with appropriate permissions.
    Set this environment variable before running the application:
    ```bash
    export GITHUB_API_KEY="ghp_YourGitHubPersonalAccessTokenHere"
    ```
    For persistence, you can add this line to your shell's startup file (e.g., `~/.bashrc`, `~/.zshrc`). The application will function without this key, but GitHub-related capabilities will be disabled. Ensure the key has the necessary scopes (e.g., `public_repo` for reading public repositories, or more depending on planned features).

*   **`AIOS_OLLAMA_API_ENDPOINT`** (Optional): Overrides the default Ollama API endpoint.
    *   Default: `"http://localhost:11434/api/generate"`
    *   Example: `export AIOS_OLLAMA_API_ENDPOINT="http://my-ollama-server:11434/api/generate"`

*   **`AIOS_DEFAULT_MODEL`** (Optional): Overrides the default Ollama model used for analysis and generation.
    *   Default: `"qwen2.5vl"` (or as specified in `config.py`)
    *   Example: `export AIOS_DEFAULT_MODEL="llama3"`

*   **`AIOS_HUMAN_APPROVAL_THRESHOLD`** (Optional): Defines the minimum severity level that requires human approval for applying changes.
    *   Valid values: `"LOW"`, `"MEDIUM"`, `"HIGH"`.
    *   Default: `"HIGH"`
    *   Example: `export AIOS_HUMAN_APPROVAL_THRESHOLD="MEDIUM"`

*   **`AIOS_MIN_STABILITY_FOR_ENHANCEMENT`** (Optional): Minimum system stability score (float, 0.0-100.0) required to attempt new enhancements.
    *   Default: `60.0`
    *   Example: `export AIOS_MIN_STABILITY_FOR_ENHANCEMENT="75.0"`

*   **`AIOS_CYCLE_INTERVAL_SECONDS`** (Optional): Time in seconds the orchestrator waits between enhancement cycles.
    *   Default: `300` (5 minutes)
    *   Example: `export AIOS_CYCLE_INTERVAL_SECONDS="600"`

*   **`AIOS_DOCKER_SHELL_IMAGE`** (Optional): Docker image used for sandboxing shell script execution.
    *   Default: `"debian:stable-slim"`
    *   Example: `export AIOS_DOCKER_SHELL_IMAGE="ubuntu:latest"`

*   **`AIOS_DOCKER_PYTHON_IMAGE`** (Optional): Docker image used for sandboxing Python script execution.
    *   Default: `"python:3.9-slim"`
    *   Example: `export AIOS_DOCKER_PYTHON_IMAGE="python:3.11-slim"`

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
    *   `OllamaInterface` sends each item to the LLM for analysis (potentially using a configured AI role, e.g., `GenericSystemItemAnalyzer`), asking for potential issues and enhancement ideas in a structured JSON format.
3.  **Conception Phase:**
    *   A system snapshot and all analysis results are sent to the LLM via `OllamaInterface` (potentially using a configured AI role, e.g., `EnhancementStrategist`).
    *   The LLM is tasked to conceive an overall strategy and a prioritized list of specific enhancements. This can include file modifications, script changes, or **system-level tasks described in natural language (as 'shell_task' types).**
4.  **Application Phase:**
    *   Each proposed enhancement is considered one by one.
    *   **Human approval** is requested if the enhancement meets the configured threshold for risk/impact. This includes displaying detailed information about the proposed change.
    *   If approved:
        *   For file-based changes, the target file is backed up by `EnhancementApplier`.
        *   If the LLM plan indicates code/content needs to be generated (e.g., a new function body for a script, or content for a new file), `OllamaInterface` requests this from the LLM (potentially using a configured AI role).
        *   If the enhancement is a `'shell_task'`:
            *   The Orchestrator uses the `natural_language_task` from the proposal to request a specific shell command, risk assessment, and safety notes from the `ShellCommandGenerator` role via `ollama_interface.generate_shell_command()`.
            *   These detailed command suggestions are presented again as part of the approval detail step (or could be a secondary approval step in a more complex workflow). If already approved, this step primarily enriches the enhancement data.
            *   If the command generation is successful and the task is still approved, the Orchestrator instructs `EnhancementApplier` (specifically, its `execute_command_or_script` function) to run the generated shell command, potentially using Docker sandboxing if configured.
        *   For file modifications or script changes, `EnhancementApplier` attempts to apply the change.
        *   For Bash scripts and Python scripts, a syntax check is performed. If it fails, the change is automatically rolled back from the backup.
5.  **Monitoring:**
    *   `SystemStateAnalyzer` checks basic system health indicators.
    *   The system stability score is updated. If it drops critically low, human intervention is flagged.
6.  **Loop or Pause:**
    *   If human intervention is required, the system pauses and waits for acknowledgment.
    *   Otherwise, it sleeps for a configured interval before starting a new cycle.

## Multi-Agent Architecture (Foundations)

The AI OS Enhancer is evolving to support a multi-agent architecture, allowing for more complex and collaborative AI behaviors. This system is designed to manage multiple specialized AI agents that can interact with the Ollama LLM and potentially with each other in the future.

The foundational components include:
- **`AgentContext`**: Manages the unique state, role, and conversation history for each individual agent.
- **`AgentManager`**: Oversees the lifecycle of agents and orchestrates their interaction turns with the LLM, ensuring context isolation.

For a more detailed explanation of this evolving architecture and the "Agent Multiplexing" concept, please see the [Multi-Agent System Design Document](./docs/AGENT_SYSTEM_DESIGN.md).

## Disclaimer

This software is a research exploration into AI-driven system administration. It is not intended for use on production systems. The developers bear no responsibility for any outcomes resulting from its use or misuse. Always prioritize safety, backups, and thorough human review when dealing with automated system modifications.
