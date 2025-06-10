# ai_os_enhancer/ollama_interface.py

import json
import requests # type: ignore
import logging
import pathlib # Added for knowledge base path handling

# Relative imports for when this module is part of the package
try:
    from . import config
    from . import logger_setup
except ImportError:
    # Fallback for direct execution (e.g., if __name__ == '__main__')
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    import logger_setup

# Initialize logger for this module
# This logger instance will be used by _load_knowledge_base_content
if __name__ == '__main__' and not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("OllamaInterface_direct")
    logger.info("Running OllamaInterface directly, basic logging configured.")
else:
    logger = logger_setup.setup_logger("OllamaInterface")


# --- Knowledge Base Helper ---
def _load_knowledge_base_content(filename="sysctl_debian_guide.txt"):
    # Ensure config.PROJECT_ROOT is a Path object for correct path joining
    project_root_path = pathlib.Path(config.PROJECT_ROOT)
    kb_path = project_root_path / "knowledge_base" / filename
    try:
        if kb_path.is_file():
            logger.info(f"Loading knowledge base: {filename} from {kb_path}")
            return kb_path.read_text(encoding='utf-8')
        else:
            logger.warning(f"Knowledge base file not found: {kb_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading knowledge base {filename} from {kb_path}: {e}", exc_info=True)
        return None

# --- OllamaInterface Module ---

def _format_data_as_text(data):
    """
    Helper function to format various data types (snapshots, analysis results) as text for prompts.
    """
    if isinstance(data, (dict, list)):
        try:
            return json.dumps(data, indent=2, sort_keys=True) # Added sort_keys for consistency
        except TypeError:
            logger.warning(f"Could not JSON serialize data of type {type(data)}, falling back to str().")
            return str(data)
    return str(data)


def query_ollama(prompt_text, model_name=None, context_data=None, is_json_response_expected=True):
    """
    Sends a query to the Ollama API.
    Returns: Raw string response or parsed JSON dictionary if successful and expected.
             Returns None on error.
    """
    if model_name is None:
        model_name = config.DEFAULT_MODEL

    full_prompt = prompt_text
    if context_data:
        # Ensure context_data is formatted nicely for inclusion in the prompt
        formatted_context = _format_data_as_text(context_data)
        full_prompt = f"Context:\n{formatted_context}\n\nPrompt:\n{prompt_text}"

    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False
    }

    if is_json_response_expected:
        payload["format"] = "json"

    # Obfuscate or shorten sensitive parts of the prompt if necessary for logging
    log_payload_prompt = payload['prompt']
    if len(log_payload_prompt) > 300: # Log only a part of very long prompts
        log_payload_prompt = log_payload_prompt[:150] + "..." + log_payload_prompt[-150:]

    logger.debug(f"Sending query to Ollama. Model: {model_name}, JSON expected: {is_json_response_expected}, API: {config.OLLAMA_API_ENDPOINT}")
    logger.debug(f"Payload (prompt excerpt): {json.dumps({**payload, 'prompt': log_payload_prompt})}")


    try:
        response = requests.post(config.OLLAMA_API_ENDPOINT, json=payload, timeout=180) # 180s timeout
        response.raise_for_status()

        response_data = response.json() # Main response from Ollama API

        raw_response_text = response_data.get("response")

        if raw_response_text is None: # Should not happen if API call was successful
            logger.error("Ollama response content (field 'response') was missing or null.")
            logger.debug(f"Full Ollama response data: {response_data}")
            return None

        if is_json_response_expected:
            try:
                # The 'response' field itself should be a JSON string if format=json was used
                parsed_json = json.loads(raw_response_text)
                logger.debug("Successfully parsed JSON content from Ollama 'response' field.")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Ollama 'response' field. Error: {e}")
                logger.error(f"Raw 'response' field content from Ollama: {raw_response_text[:500]}...") # Log snippet
                return None
        else:
            logger.debug("Returning raw text response from Ollama 'response' field.")
            return raw_response_text

    except requests.exceptions.Timeout:
        logger.error(f"Timeout after 180s connecting to Ollama API at {config.OLLAMA_API_ENDPOINT}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error with Ollama API at {config.OLLAMA_API_ENDPOINT}. Is Ollama running and accessible?")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"Ollama API HTTP error: {e.response.status_code}")
        try:
            # Try to get more detailed error from Ollama's response if available
            error_details = e.response.json()
            logger.error(f"Ollama error details: {error_details.get('error', e.response.text)}")
        except json.JSONDecodeError:
            logger.error(f"Raw HTTP error response: {e.response.text}")
        return None
    except json.JSONDecodeError: # If the main response from requests.post itself isn't JSON
        logger.error("Failed to parse the main Ollama API response structure (expected JSON from requests lib).")
        logger.debug(f"Raw response content from server: {response.text[:500]}...")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while querying Ollama: {e}", exc_info=True)
        return None


def analyze_system_item(item_content, item_path, item_type, model_name=None):
    """
    Analyzes a system item (config or script) using Ollama.
    item_type: "config" or "script"
    Returns: Dictionary {analysis_summary, potential_issues, enhancement_ideas} or None on error.
    """
    if model_name is None:
        model_name = config.DEFAULT_MODEL

    knowledge_base_text = None
    # Check if item_path is a string before performing string operations
    if isinstance(item_path, str) and ("sysctl.conf" in item_path or "/etc/sysctl.d/" in item_path):
        knowledge_base_text = _load_knowledge_base_content("sysctl_debian_guide.txt")

    prompt_intro_text = ""
    if item_type == "script":
        prompt_intro_text = (
            f"You are an expert Debian system administrator and Bash script analyzer. "
            f"Analyze the following Bash script (from path: {item_path}):\n\n"
            f"```bash\n{item_content}\n```\n\n"
            "Identify potential bugs, inefficiencies, security vulnerabilities, areas for improvement, and adherence to best practices for Debian 13. "
            "Suggest specific, actionable enhancement ideas. For each idea, detail the type of change (e.g., 'replace_bash_function', 'add_bash_commands_after_line', 'modify_bash_variable', 'refactor_bash_loop', 'add_error_handling'). "
            "Provide necessary context (e.g., function name, relevant line numbers or patterns, variable names). "
        )
    else:  # Default to config file analysis
        prompt_intro_text = (
            f"You are an expert Debian system administrator. "
            f"Analyze the following Debian configuration file (from path: {item_path}):\n\n"
            f"```plaintext\n{item_content}\n```\n\n"
            "Identify potential issues (e.g., misconfigurations, security weaknesses, deprecated settings) and inefficiencies. "
            "Suggest specific, actionable enhancement ideas relevant to a Debian 13 system. For each idea, detail 'suggested_change_type' (e.g., 'modify_config_line', 'add_config_block_after_pattern', 'remove_config_line'). "
            "Provide necessary context (e.g., line number or pattern to find, specific setting name)."
        )

    # Construct the main prompt
    full_prompt_parts = []
    if knowledge_base_text:
        full_prompt_parts.append("You are provided with the following expert guide for context. Prioritize this guide when forming your analysis and suggestions:\n--- EXPERT GUIDE START ---\n" + knowledge_base_text + "\n--- EXPERT GUIDE END ---\n\n")

    full_prompt_parts.append(prompt_intro_text)

    json_format_instruction = (
        "Your response MUST be a single, valid JSON object. Ensure all strings are properly escaped. The JSON object should have these top-level keys: "
        "'analysis_summary' (a concise string summarizing your findings), "
        "'potential_issues' (a list of strings, each describing a distinct issue), "
        "and 'enhancement_ideas' (a list of dictionaries). Each dictionary in 'enhancement_ideas' must contain: "
        "'idea_description' (string, what to do), "
        "'justification' (string, why it's beneficial), "
        "'suggested_change_type' (string, e.g., 'replace_bash_function', 'modify_config_line'), "
        "and can optionally include other relevant details such as 'function_name' (string), 'target_pattern' (string, regex or exact match pattern to locate the change), "
        "'new_code_snippet' (string, the proposed code/config change), 'language' (string, e.g., 'bash', 'ini'). "
        "Focus on enhancing stability, security, performance, and maintainability. If no issues or ideas, return empty lists for 'potential_issues' and 'enhancement_ideas'."
    )
    full_prompt_parts.append(json_format_instruction)

    prompt = "".join(full_prompt_parts)

    logger.info(f"Requesting analysis for {item_type} at {item_path} using model {model_name}.")
    return query_ollama(prompt, model_name, is_json_response_expected=True)


def conceive_enhancement_strategy(system_snapshot, analysis_results_list, model_name=None):
    """
    Conceives an enhancement strategy based on system snapshot and analysis results.
    Returns: Dictionary {overall_strategy_summary, prioritized_enhancements} or None on error.
    """
    if model_name is None:
        model_name = config.DEFAULT_MODEL

    knowledge_base_text = None
    # Check if sysctl related items are in the analysis results
    has_sysctl_context = any(
        isinstance(result.get("item_path"), str) and \
        ("sysctl.conf" in result.get("item_path", "") or "/etc/sysctl.d/" in result.get("item_path", ""))
        for result in analysis_results_list
    )

    if has_sysctl_context:
        knowledge_base_text = _load_knowledge_base_content("sysctl_debian_guide.txt")

    # Construct the main prompt
    prompt_parts = []
    if knowledge_base_text:
        prompt_parts.append("You are provided with the following expert guide for context regarding sysctl configurations. Prioritize this guide when forming your strategy for sysctl-related enhancements:\n--- EXPERT GUIDE START ---\n" + knowledge_base_text + "\n--- EXPERT GUIDE END ---\n\n")

    prompt_parts.append(
        "You are an expert Debian 13 system optimization strategist. Based on the provided system snapshot and analysis of various items, "
        "generate a prioritized list of actionable enhancements. "
    )
    prompt_parts.append(f"System Snapshot:\n{_format_data_as_text(system_snapshot)}\n\n")
    prompt_parts.append(f"Analysis of System Items:\n{_format_data_as_text(analysis_results_list)}\n\n")

    # Rest of the instructions for the prompt
    prompt_instructions = (
        "Instructions for your response: Your response MUST be a single, valid JSON object. Ensure all strings are properly escaped. "
        "The JSON object should have two top-level keys: 'overall_strategy_summary' (string, your concise overall strategy) and "
        "'prioritized_enhancements' (a list of enhancement task dictionaries). "
        "Each enhancement task dictionary in 'prioritized_enhancements' must include: "
        "1. 'item_path': string (target file or script path). "
        "2. 'item_type': string ('config' or 'script'). "
        "3. 'current_relevant_content_snippet': string (a brief, relevant snippet of current code/config, max 10 lines, for context). "
        "4. 'proposed_change_details': A nested JSON object detailing the change. This object MUST include a 'type' field (string, e.g., 'replace_bash_function', 'insert_bash_commands_after_pattern', 'modify_config_line', 'create_new_config_file', 'create_new_bash_script'). "
        "   It should also include other fields relevant to the 'type', such as 'function_name' (string), 'target_pattern' (string, regex or exact match), 'code_to_insert_or_replace' (string), 'language' (string), 'new_line_content' (string), 'block_content' (string). "
        "   Crucially, 'proposed_change_details' MUST also include a 'requires_code_generation' boolean field: set to true if new code/config content needs to be generated by a subsequent AI step, or false if the 'code_to_insert_or_replace' or similar fields already contain the complete, final content. "
        "5. 'justification': string (why this change is important). "
        "6. 'risk_assessment': string (LOW, MEDIUM, or HIGH). "
        "7. 'impact_level': string (MINOR, MODERATE, or SIGNIFICANT). "
        "Prioritize changes that offer significant benefits (security, performance, stability, maintainability) with manageable risk. "
        "If no enhancements are deemed necessary, 'prioritized_enhancements' should be an empty list."
    )
    prompt_parts.append(prompt_instructions)

    prompt = "".join(prompt_parts)

    logger.info(f"Requesting enhancement strategy using model {model_name}.")
    return query_ollama(prompt, model_name, is_json_response_expected=True)


def generate_code_or_modification(task_description, language, existing_code_context=None, modification_target_details=None, model_name=None):
    """
    Generates code or a modification for existing code using Ollama.
    Returns: String (generated code/modification) or None on error.
    """
    if model_name is None:
        model_name = config.DEFAULT_MODEL

    # Constructing the prompt based on whether it's a modification or new code generation
    if modification_target_details:
        # This is a modification task
        prompt = (
            f"You are an expert {language} programmer. Your task is to modify the existing code as described. "
            f"Modification Details: {_format_data_as_text(modification_target_details)}. "
            f"Specific Task: {task_description}\n\n"
        )
        if existing_code_context:
            prompt += f"Existing {language} code context (the part to be modified or that provides context for the modification):\n"
            prompt += f"```{language}\n{existing_code_context}\n```\n\n"
        prompt += (
            f"Provide ONLY the modified or new {language} code block that achieves the task. "
            f"Do not include any explanations, comments outside the code, or markdown formatting like ```{language} ... ```. "
            f"The output should be directly usable to replace the original section or insert as new code. "
            f"Ensure the generated code is complete for the described modification."
        )
    else:
        # This is a new code generation task
        prompt = (
            f"You are an expert {language} programmer. Your task is to generate {language} code. "
            f"Specific Task: {task_description}\n\n"
        )
        if existing_code_context: # Context can still be useful for new code
            prompt += f"Consider the following existing code or context if relevant:\n"
            prompt += f"```{language}\n{existing_code_context}\n```\n\n"
        prompt += (
            f"Provide ONLY the {language} code block that achieves the task. "
            f"Do not include any explanations, comments outside the code, or markdown formatting like ```{language} ... ```. "
            f"Ensure the code is robust, secure, and follows best practices for {language} on Debian. "
            f"The output should be the raw code itself."
        )

    logger.info(f"Requesting code generation/modification for language '{language}' using model '{model_name}'. Task: {task_description}")
    # For code generation, we expect raw text, not JSON.
    return query_ollama(prompt, model_name, is_json_response_expected=False)


if __name__ == '__main__':
    # Ensure logger_setup.setup_logger is called for the __main__ block if not already by import fallbacks.
    # The basicConfig in the header is a last resort.
    if not logger.hasHandlers() or isinstance(logger, logging.RootLogger): # Check if it's the root logger or has no handlers
        logger = logger_setup.setup_logger("OllamaInterface_main", level=logging.DEBUG)
        logger.info("Logger re-initialized for __main__ block with DEBUG level.")
    else:
        logger.setLevel(logging.DEBUG)
        logger.info("Logger already initialized. Set level to DEBUG for __main__ block if not already.")

    logger.info("--- OllamaInterface Test ---")
    logger.warning(f"NOTE: Ollama must be running at {config.OLLAMA_API_ENDPOINT} with model '{config.DEFAULT_MODEL}' (e.g., 'qwen2.5vl') pulled for these tests to pass.")

    # Test 1: Basic non-JSON query
    logger.info(f"Test 1: Basic query with model: {config.DEFAULT_MODEL}")
    prompt_test = "What is the capital of France? Respond with only the name of the city."
    raw_response = query_ollama(prompt_test, is_json_response_expected=False)
    if raw_response is not None: # Check for None explicitly
        logger.info(f"Raw response from Ollama: '{raw_response.strip()}'")
    else:
        logger.error("Failed to get raw response from Ollama for Test 1.")

    # Test 2: JSON query
    logger.info(f"Test 2: JSON query with model: {config.DEFAULT_MODEL}")
    prompt_json_test = "Provide a JSON object with two keys: 'city' and 'country', for the capital of Germany. Ensure the response is only the JSON object."
    json_response = query_ollama(prompt_json_test, is_json_response_expected=True)
    if json_response is not None:
        logger.info(f"JSON response from Ollama: {json.dumps(json_response, indent=2)}")
        if isinstance(json_response, dict) and "city" in json_response and json_response["city"].lower() == "berlin":
            logger.info("JSON test (Test 2) seems successful.")
        else:
            logger.warning(f"JSON test (Test 2) response was not as expected: {json_response}")
    else:
        logger.error("Failed to get JSON response from Ollama for Test 2.")

    # Test 3: Analyze system item (script)
    logger.info("Test 3: analyze_system_item (script)...")
    mock_script_content = "#!/bin/bash\n# A simple script\necho 'Hello World'\nls /nonexistentpath || echo 'Error expected'\nexit 0"
    script_analysis_result = analyze_system_item(mock_script_content, "/tmp/mock_script.sh", "script")
    if script_analysis_result is not None:
        logger.info(f"Mock script analysis (Test 3): {json.dumps(script_analysis_result, indent=2)}")
    else:
        logger.error("Failed to get script analysis for Test 3.")

    # Test 4: Analyze system item (config)
    logger.info("Test 4: analyze_system_item (config - generic)...")
    mock_config_content = "# Sample config\nTIMEOUT=30\nUSER=admin\n# Check this old_setting\nOLD_SETTING=true"
    config_analysis_result = analyze_system_item(mock_config_content, "/etc/mock_config.conf", "config")
    if config_analysis_result is not None:
        logger.info(f"Mock config analysis (Test 4): {json.dumps(config_analysis_result, indent=2)}")
    else:
        logger.error("Failed to get config analysis for Test 4.")

    # Test 4b: Analyze sysctl.conf (should trigger knowledge base loading)
    logger.info("Test 4b: analyze_system_item (sysctl.conf - should load KB)...")
    mock_sysctl_content = "vm.swappiness = 60\nnet.ipv4.tcp_syncookies = 1"
    # Ensure the knowledge base file exists for this test if it's being created in a prior step
    # For local test, assume it's there or _load_knowledge_base_content handles absence.
    sysctl_analysis_result = analyze_system_item(mock_sysctl_content, "/etc/sysctl.conf", "config")
    if sysctl_analysis_result is not None:
        logger.info(f"Mock sysctl.conf analysis (Test 4b): {json.dumps(sysctl_analysis_result, indent=2)}")
    else:
        logger.error("Failed to get sysctl.conf analysis for Test 4b.")


    # Test 5: Conceive enhancement strategy
    logger.info("Test 5: conceive_enhancement_strategy...")
    mock_snapshot = {"debian_version": "Debian GNU/Linux 13 (Trixie)", "kernel_version": "6.1.0-17-amd64", "load_avg": [0.1, 0.15, 0.05]}
    mock_analysis_results_for_strategy = []
    if script_analysis_result and isinstance(script_analysis_result.get("enhancement_ideas"), list): # Use result from Test 3
         mock_analysis_results_for_strategy.append({
             "item_path": "/tmp/mock_script.sh", "item_type": "script",
             "analysis_summary": script_analysis_result.get("analysis_summary"),
             "potential_issues": script_analysis_result.get("potential_issues"),
             "enhancement_ideas": script_analysis_result.get("enhancement_ideas")
        })
    if sysctl_analysis_result and isinstance(sysctl_analysis_result.get("enhancement_ideas"), list): # Use result from Test 4b
         mock_analysis_results_for_strategy.append({
             "item_path": "/etc/sysctl.conf", "item_type": "config",
             "analysis_summary": sysctl_analysis_result.get("analysis_summary"),
             "potential_issues": sysctl_analysis_result.get("potential_issues"),
             "enhancement_ideas": sysctl_analysis_result.get("enhancement_ideas")
        })

    if not mock_analysis_results_for_strategy: # Fallback if previous analyses failed
        logger.warning("Using fallback analysis results for conceive_enhancement_strategy test as prior analyses might have failed.")
        mock_analysis_results_for_strategy.append({"item_path": "/etc/some.conf", "item_type": "config", "analysis_summary": "Outdated setting", "potential_issues": ["Security risk"], "enhancement_ideas": [{"idea_description": "Update setting X", "justification": "Improves security", "suggested_change_type": "modify_config_line", "target_pattern": "X=old", "new_code_snippet": "X=new"}]})

    strategy = conceive_enhancement_strategy(mock_snapshot, mock_analysis_results_for_strategy)
    if strategy is not None:
        logger.info(f"Enhancement strategy (Test 5): {json.dumps(strategy, indent=2)}")
    else:
        logger.error("Failed to conceive enhancement strategy for Test 5.")

    # Test 6: Generate code (new bash script)
    logger.info("Test 6: generate_code_or_modification (new bash script)...")
    code_gen_task = "Write a simple bash script that creates a temporary file in /tmp, writes 'Hello from generated script' into it, and then prints the file content to stdout before deleting the file."
    generated_bash_code = generate_code_or_modification(code_gen_task, "bash")
    if generated_bash_code is not None:
        logger.info(f"Generated bash code (Test 6):\n{generated_bash_code.strip()}")
    else:
        logger.error("Failed to generate bash code for Test 6.")

    # Test 7: Generate code (modification of Python code)
    logger.info("Test 7: generate_code_or_modification (modify python code)...")
    python_modification_task = "Refactor the following Python code to use an f-string for printing the message."
    existing_python_code = "message = 'Hello, World!'\nprint('Message: ' + message)"
    modification_details = {
        "type": "refactor_string_concatenation",
        "target_construct": "string concatenation with '+' for print",
        "desired_construct": "f-string"
    }

    modified_python_code = generate_code_or_modification(
        task_description=python_modification_task,
        language="python",
        existing_code_context=existing_python_code,
        modification_target_details=modification_details
    )
    if modified_python_code is not None:
        logger.info(f"Modified python code (Test 7):\n{modified_python_code.strip()}")
    else:
        logger.error("Failed to generate modified python code for Test 7.")

    logger.info("--- End of OllamaInterface Test ---")
```
