import json
import requests # type: ignore
import logging
import pathlib
import re # For keyword extraction

# Relative imports for when this module is part of the package
try:
    from . import config  # If OllamaInterface is part of a package
    from . import logger_setup # For structured logging
except ImportError:
    # Fallback for standalone execution (e.g., for testing)
    import config  # type: ignore
    import logger_setup # type: ignore

# Initialize logger for this module
# This logger instance will be used by _load_knowledge_base_content
if __name__ == '__main__' and not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("OllamaInterface_direct_main") # Different name for direct execution
else: # If used as a module and logger is already configured
    logger = logger_setup.setup_logger("OllamaInterface")


# --- Knowledge Base Helper ---
def _extract_keywords_from_path(item_path_str: str) -> list[str]:
    """
    Extracts potential keywords from a file path string.
    Uses a simple regex to find words and then lowercases them.
    """
    if not item_path_str:
        return []
    
    # Regex to find sequences of word characters
    words = re.findall(r'\b\w+\b', item_path_str.lower())
    
    # Further refinement could be done here (e.g., removing common extensions like 'conf', 'sh', 'txt')
    # For now, keeping it simple as requested.
    # Example specific filtering:
    keywords = set(words)
    common_parts = {'etc', 'conf', 'd', 'sh', 'config', 'default'}
    keywords.difference_update(common_parts)

    # Add specific known config names without extension, if filename matches exactly
    filename = pathlib.Path(item_path_str).name.lower()
    known_configs = {
        "sysctl.conf": "sysctl", "sshd_config": "sshd", "named.conf": "named",
        "resolv.conf": "resolv", "fstab": "fstab", "crontab": "cron", 
        "sudoers": "sudo", "interfaces": "network", "grub": "grub",
    }
    if filename in known_configs:
        keywords.add(known_configs[filename])
    
    # Add parent directory names if they are common keywords
    path_obj = pathlib.Path(item_path_str)
    for parent_part in path_obj.parts:
        parent_part_lower = parent_part.lower()
        if parent_part_lower in ["network", "sysctl", "ssh", "cron", "apt", "kernel", "security", "logrotate", "ufw", "firewalld", "grub", "systemd"]:
            keywords.add(parent_part_lower)

    final_keywords = list(filter(None, keywords)) # Remove any empty strings
    logger.debug(f"Extracted keywords: {final_keywords} from path: {item_path_str}")
    return final_keywords

def _load_knowledge_base_content(keywords: list[str] | None = None, filename: str = "sysctl_debian_guide.txt") -> str | None:
    """
    Loads knowledge base content relevant to the given keywords or a specific fallback filename.
    If keywords are provided, searches for files named <keyword>.txt in knowledge_base/keywords/.
    If no content is found via keywords (or keywords are not provided), tries to load the fallback filename
    from knowledge_base/ (which might be redirected to knowledge_base/keywords/ if filename is e.g. "sysctl.txt").
    """
    loaded_contents = []
    loaded_files_log = []
    project_root_path = pathlib.Path(config.PROJECT_ROOT)
    kb_keywords_dir = project_root_path / "knowledge_base" / "keywords"

    if keywords:
        processed_keywords = set()
        for keyword in keywords:
            normalized_keyword = keyword.lower().strip()
            if not normalized_keyword or normalized_keyword in processed_keywords:
                continue
            
            kb_file_path = kb_keywords_dir / f"{normalized_keyword}.txt"
            try:
                if kb_file_path.is_file():
                    content = kb_file_path.read_text(encoding='utf-8')
                    loaded_contents.append(content)
                    loaded_files_log.append(kb_file_path.name)
                    logger.info(f"Successfully loaded KB file: '{kb_file_path.name}' for keyword '{normalized_keyword}'")
                    processed_keywords.add(normalized_keyword)
                else:
                    logger.debug(f"KB file not found for keyword '{normalized_keyword}': {kb_file_path}")
            except Exception as e:
                logger.error(f"Error loading KB file {kb_file_path} for '{normalized_keyword}': {e}", exc_info=True)

    if loaded_contents:
        logger.info(f"Knowledge base content aggregated from keyword files: {', '.join(loaded_files_log)}")
        return "\n\n--- KB: Additional Context Block ---\n\n".join(loaded_contents)
    
    # Fallback to the old mechanism if no keyword-based content was loaded
    # This handles the sysctl_debian_guide.txt case gracefully, assuming it's now sysctl.txt
    if filename: # Default is "sysctl_debian_guide.txt"
        # Adjust filename if it's the old default, to point to the new location/name
        fallback_filename = filename
        if filename == "sysctl_debian_guide.txt":
            fallback_filename = "sysctl.txt" # New name in keywords dir
        
        # Try loading from keywords directory first for the fallback filename (e.g. "sysctl.txt")
        kb_path = kb_keywords_dir / fallback_filename
        if not kb_path.is_file(): # If not in keywords, try the old base knowledge_base dir
             kb_path = project_root_path / "knowledge_base" / filename # Original filename for this path

        if kb_path.is_file():
            try:
                logger.info(f"Loading fallback knowledge base: {kb_path.name} from {kb_path.parent}")
                return kb_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"Error loading fallback knowledge base {kb_path.name}: {e}", exc_info=True)
        else:
            logger.info(f"No specific keyword files loaded, and fallback KB file '{filename}' (checked as '{kb_path}') not found.")
            
    return None


# --- Prompt Construction Helpers ---
def _build_analyze_item_prompt(item_content: str, item_path: str, item_type: str, knowledge_base_text: str | None) -> str:
    """Builds the prompt for analyzing a system item."""
    prompt_intro_text = ""
    if item_type == "script":
        prompt_intro_text = (
            f"You are an expert Debian system administrator and Bash script analyzer. "
            f"Analyze the following Bash script (from path: {item_path}):\n\n"
            f"```bash\n{item_content}\n```\n\n"
            "Your analysis should identify potential issues (e.g., bugs, security vulnerabilities, performance bottlenecks, non-standard practices) "
            "and suggest specific, actionable enhancement ideas (e.g., code changes for optimization, security hardening, improved error handling, better logging, or POSIX compliance). "
            "Provide necessary context (e.g., function name, relevant line numbers or patterns, variable names). "
        )
    else:  # Default to config file analysis
        prompt_intro_text = (
            f"You are an expert Debian system administrator. "
            f"Analyze the following Debian configuration file (from path: {item_path}):\n\n"
            f"```plaintext\n{item_content}\n```\n\n"
            "Your analysis should identify potential issues (e.g., misconfigurations, security risks, performance implications, deprecated settings) "
            "and suggest specific, actionable enhancement ideas (e.g., changes to settings for optimization, security hardening, or best practices). "
            "Provide necessary context (e.g., line number or pattern to find, specific setting name)."
        )

    full_prompt_parts = []
    if knowledge_base_text:
        full_prompt_parts.append("You are provided with the following expert guide(s) for context. Prioritize this guide when forming your analysis and suggestions:\n--- EXPERT GUIDE START ---\n" + knowledge_base_text + "\n--- EXPERT GUIDE END ---\n\n")

    full_prompt_parts.append(prompt_intro_text)

    json_format_instruction = (
        "Your response MUST be a single, valid JSON object. Ensure all strings are properly escaped. The JSON object should have these top-level keys: "
        "'analysis_summary' (a concise string summarizing your findings), "
        "'potential_issues' (a list of strings, each describing a distinct issue), "
        "and 'enhancement_ideas' (a list of dictionaries). Each dictionary in 'enhancement_ideas' must contain: "
        "'idea_description' (string, what to do), 'justification' (string, why it's beneficial), "
        "'suggested_change_type' (string, e.g., 'replace_line', 'add_line', 'modify_script_logic', 'set_value'), "
        "'target_pattern' (string, regex or simple string to locate the change, or 'N/A' if adding new content), "
        "'new_code_snippet' (string, the proposed code/config change), 'language' (string, e.g., 'bash', 'ini'). "
        "Focus on enhancing stability, security, performance, and maintainability. If no issues or ideas, return empty lists for 'potential_issues' and 'enhancement_ideas'."
    )
    full_prompt_parts.append(json_format_instruction)
    return "".join(full_prompt_parts)

def _build_strategy_prompt(system_snapshot: dict, analysis_results_list: list, knowledge_base_text: str | None) -> str:
    """Builds the prompt for conceiving an enhancement strategy."""
    prompt_parts = []
    if knowledge_base_text:
        prompt_parts.append("You are provided with the following expert guide(s) for context. Prioritize this guide when forming your strategy, especially for items related to the guide's topics:\n--- EXPERT GUIDE START ---\n" + knowledge_base_text + "\n--- EXPERT GUIDE END ---\n\n")

    prompt_parts.append(
        "You are an expert Debian 13 system optimization strategist. Based on the provided system snapshot and analysis of various items, "
        "generate a prioritized list of actionable enhancements. "
    )
    prompt_parts.append(f"System Snapshot:\n{_format_data_as_text(system_snapshot)}\n\n")
    prompt_parts.append(f"Analysis of System Items:\n{_format_data_as_text(analysis_results_list)}\n\n")

    prompt_instructions = (
        "Instructions for your response: Your response MUST be a single, valid JSON object. Ensure all strings are properly escaped. "
        "The JSON object should have two top-level keys: 'overall_strategy_summary' (string, your concise overall strategy) and "
        "'prioritized_enhancements' (a list of enhancement task dictionaries). "
        "Each task dictionary must include: 'item_path' (string, path to the item being enhanced), "
        "'item_type' (string, 'script' or 'config'), 'enhancement_description' (string, detailed description of the change), "
        "'justification' (string, why this change is important), "
        "'estimated_impact' (string, e.g., 'High', 'Medium', 'Low' - considering benefit and risk), "
        "'change_type' (string, e.g., 'replace_line', 'add_block', 'modify_script_code', 'set_config_value'), "
        "'target_criteria' (string, regex or specific line/block to target, or 'N/A'), "
        "'proposed_change_snippet' (string, the actual code/config snippet to apply), "
        "'verification_steps' (string, how to verify the change was successful, e.g., command to run, expected output), "
        "and 'rollback_steps' (string, how to revert the change if needed). "
        "Prioritize changes that offer significant benefits (security, performance, stability, maintainability) with manageable risk. "
        "If no enhancements are deemed necessary, 'prioritized_enhancements' should be an empty list."
    )
    prompt_parts.append(prompt_instructions)
    return "".join(prompt_parts)

# --- OllamaInterface Module --- 
def _format_data_as_text(data): # This function was already present
    # Simple text formatting for system snapshot and analysis results
    if isinstance(data, dict):
        return "\n".join([f"- {key}: {value}" for key, value in data.items()])
    elif isinstance(data, list):
        # Format list of dicts (like analysis_results) more nicely
        if all(isinstance(i, dict) for i in data):
            formatted_items = []
            for count, item_dict in enumerate(data, 1):
                item_details = [f"  Item {count}:"]
                for k, v in item_dict.items():
                    item_details.append(f"    {k}: {str(v)[:150]}{'...' if len(str(v)) > 150 else ''}") # Truncate long values
                formatted_items.append("\n".join(item_details))
            return "\n".join(formatted_items)
        return "\n".join([f"- {item}" for item in data]) # Simple list
    return str(data)

# --- Main Interface Functions ---
def query_ollama(prompt: str, model_name: str, is_json_response_expected: bool = False) -> dict | str | None:
    logger.debug(f"Querying Ollama with prompt (first 200 chars): {prompt[:200]}...")
    api_url = config.OLLAMA_API_ENDPOINT

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    if is_json_response_expected:
        payload["format"] = "json"

    try:
        response = requests.post(api_url, json=payload, timeout=180) 
        response.raise_for_status()
        
        logger.debug(f"Raw response from Ollama for model {model_name} (first 500 chars): {response.text[:500]}")

        if is_json_response_expected:
            try:
                return response.json() 
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response from Ollama: {e}", exc_info=True)
                logger.error(f"Non-JSON response received despite expecting JSON: {response.text}")
                return {
                    "error": "Failed to decode JSON response", "details": str(e), "response_text": response.text,
                    "analysis_summary": "Error: Ollama response was not valid JSON.", "potential_issues": ["Ollama response was not valid JSON."], "enhancement_ideas": [],
                    "overall_strategy_summary": "Error: Ollama response was not valid JSON.", "prioritized_enhancements": []
                }
        else: 
            return response.text

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}", exc_info=True)
        return {
            "error": f"Ollama API request failed: {str(e)}",
            "analysis_summary": "Error: Ollama API request failed.", "potential_issues": [f"API Error: {str(e)}"], "enhancement_ideas": [],
            "overall_strategy_summary": "Error: Ollama API request failed.", "prioritized_enhancements": []
        }
    except Exception as e: 
        logger.error(f"An unexpected error occurred during Ollama query: {e}", exc_info=True)
        return {
            "error": f"Unexpected error: {str(e)}",
            "analysis_summary": "Error: An unexpected error occurred.", "potential_issues": [f"Unexpected error: {str(e)}"], "enhancement_ideas": [],
            "overall_strategy_summary": "Error: An unexpected error occurred.", "prioritized_enhancements": []
        }


def analyze_system_item(item_content: str, item_path: str, item_type: str, model_name: str | None = None) -> dict | None:
    """
    Analyzes a system item (script or config file) using Ollama.
    Returns a dictionary (parsed JSON) or None if a fundamental error occurs before Ollama query.
    """
    if model_name is None:
        model_name = config.DEFAULT_MODEL

    item_path_str = str(item_path) 
    keywords = _extract_keywords_from_path(item_path_str)
    # Pass keywords to _load_knowledge_base_content.
    # The old filename "sysctl_debian_guide.txt" is now a default fallback within _load_knowledge_base_content.
    knowledge_base_text = _load_knowledge_base_content(keywords=keywords, filename="sysctl_debian_guide.txt") 
    
    prompt = _build_analyze_item_prompt(item_content, item_path_str, item_type, knowledge_base_text)

    logger.info(f"Requesting analysis for {item_type} at '{item_path_str}' using model '{model_name}'. Keywords for KB: {keywords if keywords else 'None'}.")
    
    response = query_ollama(prompt, model_name, is_json_response_expected=True)
    
    if isinstance(response, dict):
        return response
    
    logger.warning(f"analyze_system_item received non-dict response from query_ollama when dict was expected: {type(response)}. Path: {item_path_str}")
    return {"error": "Unexpected response type from Ollama query.", 
            "analysis_summary": "Error: Non-dictionary response from query.", 
            "potential_issues": ["Invalid response from Ollama interface."], 
            "enhancement_ideas": []}


def conceive_enhancement_strategy(system_snapshot: dict, analysis_results_list: list, model_name: str | None = None) -> dict | None:
    """
    Generates a prioritized list of enhancement tasks.
    Returns a dictionary (parsed JSON) or None if a fundamental error occurs.
    """
    if model_name is None:
        model_name = config.DEFAULT_MODEL

    all_keywords = set()
    if isinstance(analysis_results_list, list):
        for result in analysis_results_list:
            if isinstance(result, dict):
                item_path_str = result.get("item_path")
                if isinstance(item_path_str, str):
                    extracted_keywords = _extract_keywords_from_path(item_path_str)
                    for kw in extracted_keywords:
                        all_keywords.add(kw)
                else: 
                    logger.debug(f"item_path is not a string or missing in analysis_results_list item: {result.get('item_path')}")
            else: 
                logger.warning(f"Item in analysis_results_list is not a dict: {result}")
    else: 
        logger.warning(f"analysis_results_list is not a list: {analysis_results_list}")
    
    final_keywords_list = list(all_keywords)
    knowledge_base_text = _load_knowledge_base_content(keywords=final_keywords_list) # Pass all unique keywords
    
    prompt = _build_strategy_prompt(system_snapshot, analysis_results_list, knowledge_base_text)

    logger.info(f"Requesting enhancement strategy using model '{model_name}'. Keywords for KB: {final_keywords_list if final_keywords_list else 'None'}.")
    
    response = query_ollama(prompt, model_name, is_json_response_expected=True)

    if isinstance(response, dict):
        return response
        
    logger.warning(f"conceive_enhancement_strategy received non-dict response from query_ollama: {type(response)}.")
    return {"error": "Unexpected response type from Ollama query.",
            "overall_strategy_summary": "Error: Non-dictionary response from query.",
            "prioritized_enhancements": []}


def generate_code_or_modification(task_description: str, language: str, existing_code_context: str | None = None, model_name: str | None = None) -> str | None:
    """
    Generates new code or modifies existing code.
    Returns the code string or None if an error occurs or format is incorrect.
    """
    if model_name is None:
        model_name = config.DEFAULT_MODEL

    if existing_code_context:
        prompt = (
            f"You are an expert {language} programmer. Modify the following {language} code based on the task description. "
            f"Task: {task_description}\n\n"
            f"Existing Code:\n```\n{existing_code_context}\n```\n\n"
            f"Your response MUST be a single, valid JSON object with a single key 'code' containing ONLY the modified code as a string. Ensure all special characters in the code string are properly escaped for JSON."
        )
    else:
        prompt = (
            f"You are an expert {language} programmer. Generate a new {language} code snippet based on the task description. "
            f"Task: {task_description}\n\n"
            f"Your response MUST be a single, valid JSON object with a single key 'code' containing ONLY the generated code as a string. Ensure all special characters in the code string are properly escaped for JSON."
        )
    
    logger.info(f"Requesting code generation/modification for language '{language}' using model '{model_name}'.")
    
    response_data = query_ollama(prompt, model_name, is_json_response_expected=True)

    if isinstance(response_data, dict):
        if "code" in response_data and isinstance(response_data["code"], str):
            return response_data["code"]
        elif "error" in response_data: # Check for structured error from query_ollama
            logger.error(f"Code generation failed: {response_data['error']}. Details: {response_data.get('details')}")
            return None
        else: # Fallback for unexpected dict structure
            logger.error(f"Code generation response JSON is missing 'code' key or it's not a string. Response: {response_data}")
            return None
    # Handle cases where query_ollama might not return a dict (e.g., if it returned a string or None, though current impl aims for dicts)
    else:
        logger.error(f"Code generation query returned non-dict response: {response_data}")
        return None

# --- Main execution for testing ---
if __name__ == '__main__':
    # Ensure logger is configured for direct script execution with debug level
    # This specifically targets the logger used in this module ("OllamaInterface" or "OllamaInterface_direct_main")
    # If logger_setup.setup_logger was used, it might have returned a logger with a specific name.
    # For simplicity, let's re-get the logger by the name it would have if this is run directly.
    # This assumes the module-level logger was named "OllamaInterface" by logger_setup.setup_logger.
    # If it's the __main__ block's "OllamaInterface_direct_main", that one is used.
    
    # The logger instance 'logger' is already defined at the module level.
    # We just need to ensure its level and handlers are appropriate for testing.
    logger.setLevel(logging.DEBUG) 
    
    # Ensure there's a console handler for direct script execution output
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        log_format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if hasattr(logger_setup, 'LOG_FORMAT'):
             log_format_str = logger_setup.LOG_FORMAT
        formatter = logging.Formatter(log_format_str)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    logger.info("--- Starting OllamaInterface Test (Direct Execution with Full Features) ---")

    # Create dummy knowledge base files for testing
    # This setup should ideally be part of a test fixture if using a test framework
    try:
        kb_dir = pathlib.Path(config.PROJECT_ROOT) / "knowledge_base" / "keywords"
        kb_dir.mkdir(parents=True, exist_ok=True)
        (kb_dir / "sysctl.txt").write_text("# Sysctl specific knowledge\nvm.swappiness = 10\n")
        (kb_dir / "network.txt").write_text("# Network specific knowledge\n# Example: How to configure a static IP.\n")
        (kb_dir / "ssh.txt").write_text("# SSH best practices\nPermitRootLogin no\n")
        logger.info(f"Test KB files created in {kb_dir}")
    except Exception as e:
        logger.error(f"Could not create test KB files: {e}", exc_info=True)


    test_model = config.DEFAULT_MODEL 

    # Test 1: Simple query (non-JSON)
    logger.info("Test 1: Simple query...")
    simple_prompt = "What is the capital of France? Respond with just the city name."
    simple_response = query_ollama(simple_prompt, test_model) 
    if isinstance(simple_response, str):
        logger.info(f"Simple query response (Test 1): {simple_response.strip()}")
    elif isinstance(simple_response, dict) and "error" in simple_response:
        logger.error(f"Simple query (Test 1) failed: {simple_response['error']}")
    else:
        logger.error(f"Failed to get simple query response for Test 1 or response was unexpected: {simple_response}")

    # Test 2: JSON query
    logger.info("Test 2: JSON query...")
    json_prompt = "Provide a JSON object with two keys: 'city' and 'country', for Paris, France."
    json_response = query_ollama(json_prompt, test_model, is_json_response_expected=True)
    if isinstance(json_response, dict) and "error" not in json_response:
        logger.info(f"JSON query response (Test 2): {json.dumps(json_response, indent=2)}")
    else:
        err_msg = json_response.get('error') if isinstance(json_response, dict) else "Response was not a dict or None"
        logger.error(f"Failed to get JSON query response for Test 2: {err_msg}. Raw: {json_response}")

    # Test 3: Analyze system item (script)
    logger.info("Test 3: analyze_system_item (script)...")
    mock_script_content = "#!/bin/bash\n# A simple script\necho 'Hello World'\nls /nonexistentpath || echo 'Error expected'\nexit 0"
    script_analysis_result = analyze_system_item(mock_script_content, "/opt/custom_scripts/myscript.sh", "script", model_name=test_model)
    if script_analysis_result and isinstance(script_analysis_result, dict) and "error" not in script_analysis_result:
        logger.info(f"Mock script analysis (Test 3): {json.dumps(script_analysis_result, indent=2)}")
    else:
        err_msg = script_analysis_result.get('error') if isinstance(script_analysis_result, dict) else "Result was None or not a dict"
        logger.error(f"Failed to get script analysis for Test 3: {err_msg}. Raw: {script_analysis_result}")

    # Test 4: Analyze system item (config - generic with potential 'network' keyword)
    logger.info("Test 4: analyze_system_item (config - /etc/network/interfaces)...")
    mock_config_content = "auto eth0\niface eth0 inet static\n  address 192.168.1.100\n  netmask 255.255.255.0"
    config_analysis_result = analyze_system_item(mock_config_content, "/etc/network/interfaces", "config", model_name=test_model)
    if config_analysis_result and isinstance(config_analysis_result, dict) and "error" not in config_analysis_result:
        logger.info(f"Mock config analysis (Test 4 - network/interfaces): {json.dumps(config_analysis_result, indent=2)}")
    else:
        err_msg = config_analysis_result.get('error') if isinstance(config_analysis_result, dict) else "Result was None or not a dict"
        logger.error(f"Failed to get config analysis for Test 4: {err_msg}. Raw: {config_analysis_result}")

    # Test 4b: Analyze sysctl.conf (should trigger keyword-based KB loading for 'sysctl')
    logger.info("Test 4b: analyze_system_item (sysctl.conf - should load sysctl.txt KB)...")
    mock_sysctl_content = "vm.swappiness = 60\nnet.ipv4.tcp_syncookies = 1"
    sysctl_analysis_result = analyze_system_item(mock_sysctl_content, "/etc/sysctl.conf", "config", model_name=test_model)
    if sysctl_analysis_result and isinstance(sysctl_analysis_result, dict) and "error" not in sysctl_analysis_result:
        logger.info(f"Mock sysctl.conf analysis (Test 4b): {json.dumps(sysctl_analysis_result, indent=2)}")
    else:
        err_msg = sysctl_analysis_result.get('error') if isinstance(sysctl_analysis_result, dict) else "Result was None or not a dict"
        logger.error(f"Failed to get sysctl.conf analysis for Test 4b: {err_msg}. Raw: {sysctl_analysis_result}")
    
    # Test 4c: Analyze sshd_config (should trigger keyword-based KB loading for 'sshd' or 'ssh')
    logger.info("Test 4c: analyze_system_item (sshd_config - should load ssh.txt KB)...")
    mock_sshd_content = "PermitRootLogin yes\nPasswordAuthentication yes"
    sshd_analysis_result = analyze_system_item(mock_sshd_content, "/etc/ssh/sshd_config", "config", model_name=test_model)
    if sshd_analysis_result and isinstance(sshd_analysis_result, dict) and "error" not in sshd_analysis_result:
        logger.info(f"Mock sshd_config analysis (Test 4c): {json.dumps(sshd_analysis_result, indent=2)}")
    else:
        err_msg = sshd_analysis_result.get('error') if isinstance(sshd_analysis_result, dict) else "Result was None or not a dict"
        logger.error(f"Failed to get sshd_config analysis for Test 4c: {err_msg}. Raw: {sshd_analysis_result}")


    # Test 5: Conceive enhancement strategy (will use KB based on keywords from results)
    logger.info("Test 5: conceive_enhancement_strategy...")
    mock_snapshot = {"debian_version": "Debian GNU/Linux 13 (Trixie)", "kernel_version": "6.1.0-17-amd64", "load_avg": [0.1, 0.15, 0.05]}
    mock_analysis_results_for_strategy = []
    # Add results if they were successful
    if script_analysis_result and isinstance(script_analysis_result, dict) and "error" not in script_analysis_result:
         mock_analysis_results_for_strategy.append({
             "item_path": "/opt/custom_scripts/myscript.sh", "item_type": "script", 
             "analysis_summary": script_analysis_result.get("analysis_summary", "N/A"),
             "potential_issues": script_analysis_result.get("potential_issues", []),
             "enhancement_ideas": script_analysis_result.get("enhancement_ideas", [])
        })
    if sysctl_analysis_result and isinstance(sysctl_analysis_result, dict) and "error" not in sysctl_analysis_result: 
         mock_analysis_results_for_strategy.append({
             "item_path": "/etc/sysctl.conf", "item_type": "config", 
             "analysis_summary": sysctl_analysis_result.get("analysis_summary", "N/A"),
             "potential_issues": sysctl_analysis_result.get("potential_issues", []),
             "enhancement_ideas": sysctl_analysis_result.get("enhancement_ideas", [])
        })
    if sshd_analysis_result and isinstance(sshd_analysis_result, dict) and "error" not in sshd_analysis_result: 
         mock_analysis_results_for_strategy.append({
             "item_path": "/etc/ssh/sshd_config", "item_type": "config", 
             "analysis_summary": sshd_analysis_result.get("analysis_summary", "N/A"),
             "potential_issues": sshd_analysis_result.get("potential_issues", []),
             "enhancement_ideas": sshd_analysis_result.get("enhancement_ideas", [])
        })


    if not mock_analysis_results_for_strategy: 
        logger.warning("Using fallback analysis results for conceive_enhancement_strategy test as prior analyses might have failed or yielded errors.")
        mock_analysis_results_for_strategy.append({"item_path": "/etc/network/interfaces", "item_type": "config", "analysis_summary": "Outdated network setting", "potential_issues": ["Security risk"], "enhancement_ideas": [{"idea_description": "Update setting X", "justification": "Improves security", "suggested_change_type": "modify_config_line", "target_pattern": "X=old", "new_code_snippet": "X=new"}]})

    strategy = conceive_enhancement_strategy(mock_snapshot, mock_analysis_results_for_strategy, model_name=test_model)
    if strategy and isinstance(strategy, dict) and "error" not in strategy:
        logger.info(f"Enhancement strategy (Test 5): {json.dumps(strategy, indent=2)}")
    else:
        err_msg = strategy.get('error') if isinstance(strategy, dict) else "Result was None or not a dict"
        logger.error(f"Failed to get enhancement strategy for Test 5: {err_msg}. Raw: {strategy}")


    # Test 6: Generate new code (bash script)
    logger.info("Test 6: generate_code_or_modification (new bash script)...")
    new_script_task = "Create a bash script that prints 'Hello from new script' and then lists files in /tmp."
    new_bash_script = generate_code_or_modification(new_script_task, "bash", model_name=test_model)
    if new_bash_script:
        logger.info(f"Generated new bash script (Test 6):\n{new_bash_script}")
    else:
        logger.error("Failed to generate new bash script for Test 6 (it was None or empty).")

    # Test 7: Modify existing code (python script)
    logger.info("Test 7: generate_code_or_modification (modify python script)...")
    existing_python_code = "def greet(name):\n    print(f'Hello, {name}')"
    modification_task = "Modify the python function to also print 'Welcome!' on a new line after the greeting."
    modified_python_code = generate_code_or_modification(modification_task, "python", existing_python_code, model_name=test_model)
    if modified_python_code:
        logger.info(f"Modified python code (Test 7):\n{modified_python_code}")
    else:
        logger.error("Failed to generate modified python code for Test 7 (it was None or empty).")

    logger.info("--- End of OllamaInterface Test (Direct Execution with Full Features) ---")
