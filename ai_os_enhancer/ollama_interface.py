import json
import requests # type: ignore
import logging
import pathlib
import re # For keyword extraction
import yaml # For loading role configurations

# Relative imports for when this module is part of the package
try:
    from . import config
    from . import logger_setup
except ImportError:
    # Fallback for standalone execution
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from ai_os_enhancer import config
    from ai_os_enhancer import logger_setup

# Initialize logger for this module
if __name__ == '__main__' and not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("OllamaInterface_direct_main")
else:
    logger = logger_setup.setup_logger("OllamaInterface")

# Cache for loaded role configurations
_LOADED_ROLES_CACHE = {}

def load_role_config(role_name: str) -> dict | None:
    """
    Loads a role configuration from a YAML file in the 'ai_os_enhancer/roles/' directory.
    Implements caching to avoid redundant file reads and parsing.
    """
    if role_name in _LOADED_ROLES_CACHE:
        logger.debug(f"Returning cached role config for '{role_name}'")
        return _LOADED_ROLES_CACHE[role_name]

    try:
        roles_dir = pathlib.Path(config.PROJECT_ROOT) / "ai_os_enhancer" / "roles"
        role_file_path = roles_dir / f"{role_name.lower()}.yaml"

        if not role_file_path.is_file():
            logger.error(f"Role configuration file not found: {role_file_path}")
            return None

        logger.info(f"Loading role configuration from: {role_file_path}")
        with open(role_file_path, 'r', encoding='utf-8') as f:
            role_data = yaml.safe_load(f)

        if not isinstance(role_data, dict):
            logger.error(f"Role configuration in {role_file_path} is not a valid dictionary.")
            return None

        required_keys = ["role_name", "system_prompt"]
        if not all(key in role_data for key in required_keys):
            logger.error(f"Role configuration in {role_file_path} is missing one or more required keys: {required_keys}")
            return None

        if role_data.get("role_name") != role_name:
            logger.warning(f"Role name in file ('{role_data.get('role_name')}') does not match requested role_name ('{role_name}') in {role_file_path}. Using requested name for cache key.")

        _LOADED_ROLES_CACHE[role_name] = role_data
        return role_data

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML for role '{role_name}' from {role_file_path}: {e}", exc_info=True)
        return None
    except FileNotFoundError:
        logger.error(f"Role configuration file not found (FileNotFoundError during open): {role_file_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading role '{role_name}': {e}", exc_info=True)
        return None

# --- Knowledge Base Helper ---
def _extract_keywords_from_path(item_path_str: str) -> list[str]:
    if not item_path_str: return []
    words = re.findall(r'\b\w+\b', item_path_str.lower())
    keywords = set(words)
    common_parts = {'etc', 'conf', 'd', 'sh', 'config', 'default', 'lib', 'usr', 'bin', 'local'}
    keywords.difference_update(common_parts)
    filename = pathlib.Path(item_path_str).name.lower()
    known_configs = {
        "sysctl.conf": "sysctl", "sshd_config": "sshd", "named.conf": "named",
        "resolv.conf": "resolv", "fstab": "fstab", "crontab": "cron",
        "sudoers": "sudo", "interfaces": "network", "grub": "grub",
    }
    if filename in known_configs: keywords.add(known_configs[filename])
    path_obj = pathlib.Path(item_path_str)
    for parent_part in path_obj.parts:
        parent_part_lower = parent_part.lower()
        if parent_part_lower in ["network", "sysctl", "ssh", "cron", "apt", "kernel", "security", "logrotate", "ufw", "firewalld", "grub", "systemd"]:
            keywords.add(parent_part_lower)
    final_keywords = list(filter(None, keywords))
    logger.debug(f"Extracted keywords: {final_keywords} from path: {item_path_str}")
    return final_keywords

def _load_knowledge_base_content(keywords: list[str] | None = None, filename: str | None = None) -> str | None:
    loaded_contents = []
    loaded_files_log = []
    kb_keywords_dir = config.PROJECT_ROOT / "knowledge_base" / "keywords"

    if keywords:
        processed_keywords = set()
        for keyword in keywords:
            normalized_keyword = keyword.lower().strip()
            if not normalized_keyword or normalized_keyword in processed_keywords: continue
            kb_file_path = kb_keywords_dir / f"{normalized_keyword}.txt"
            try:
                if kb_file_path.is_file():
                    content = kb_file_path.read_text(encoding='utf-8')
                    loaded_contents.append(content)
                    loaded_files_log.append(kb_file_path.name)
                    logger.info(f"Successfully loaded KB file: '{kb_file_path.name}' for keyword '{normalized_keyword}'")
                    processed_keywords.add(normalized_keyword)
                else: logger.debug(f"KB file not found for keyword '{normalized_keyword}': {kb_file_path}")
            except Exception as e: logger.error(f"Error loading KB file {kb_file_path} for '{normalized_keyword}': {e}", exc_info=True)

    # Fallback or direct filename loading (if keywords didn't yield anything or filename specifically given)
    if filename and not loaded_contents: # Only try fallback if no keyword files loaded OR if filename is explicitly passed for direct load
        fallback_filename_to_check = filename
        if filename == "sysctl_debian_guide.txt": fallback_filename_to_check = "sysctl.txt" # Legacy

        kb_path = kb_keywords_dir / fallback_filename_to_check
        if not kb_path.is_file(): kb_path = config.PROJECT_ROOT / "knowledge_base" / filename

        if kb_path.is_file():
            try:
                logger.info(f"Loading direct/fallback knowledge base: {kb_path.name} from {kb_path.parent}")
                loaded_contents.append(kb_path.read_text(encoding='utf-8')) # Append to allow combination if needed
                loaded_files_log.append(kb_path.name)
            except Exception as e: logger.error(f"Error loading direct/fallback KB {kb_path.name}: {e}", exc_info=True)
        elif not keywords: # Only log "not found" for fallback if no keywords were even attempted
            logger.info(f"Fallback/direct KB file '{filename}' (checked as '{kb_path}') not found.")

    if loaded_contents:
        logger.info(f"Knowledge base content aggregated from: {', '.join(loaded_files_log)}")
        return "\n\n--- KB: Additional Context Block ---\n\n".join(loaded_contents)
    if not keywords and not filename: # Explicitly no keywords and no fallback filename
         logger.info("No keywords or fallback filename provided for knowledge base loading.")
    return None

# --- Prompt Construction Helpers ---
def _build_analyze_item_prompt(item_content: str, item_path: str, item_type: str, knowledge_base_text: str | None, role_system_prompt: str | None, output_format: str = "json") -> str:
    full_prompt_parts = []

    if role_system_prompt:
        full_prompt_parts.append(role_system_prompt + "\n\n") # Use role's system prompt

    # Specific instructions for analysis based on item type
    if item_type == "script":
        full_prompt_parts.append(
            f"Analyze the following Bash script (from path: {item_path}):\n\n"
            f"```bash\n{item_content}\n```\n\n"
            "Your analysis should identify potential issues (e.g., bugs, security vulnerabilities, performance bottlenecks, non-standard practices) "
            "and suggest specific, actionable enhancement ideas (e.g., code changes for optimization, security hardening, improved error handling, better logging, or POSIX compliance). "
            "Provide necessary context (e.g., function name, relevant line numbers or patterns, variable names). "
        )
    else:  # Default to config file analysis
        full_prompt_parts.append(
            f"Analyze the following Debian configuration file (from path: {item_path}):\n\n"
            f"```plaintext\n{item_content}\n```\n\n"
            "Your analysis should identify potential issues (e.g., misconfigurations, security risks, performance implications, deprecated settings) "
            "and suggest specific, actionable enhancement ideas (e.g., changes to settings for optimization, security hardening, or best practices). "
            "Provide necessary context (e.g., line number or pattern to find, specific setting name)."
        )

    if knowledge_base_text:
        full_prompt_parts.append("\n\nYou are provided with the following expert guide(s) for context. Prioritize this guide when forming your analysis and suggestions:\n--- EXPERT GUIDE START ---\n" + knowledge_base_text + "\n--- EXPERT GUIDE END ---\n\n")

    if output_format == "json":
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
        full_prompt_parts.append("\n\n" + json_format_instruction)
    return "".join(full_prompt_parts)

def _build_strategy_prompt(system_snapshot: dict, analysis_results_list: list, knowledge_base_text: str | None, role_system_prompt: str | None, output_format: str = "json") -> str:
    prompt_parts = []

    if role_system_prompt:
        prompt_parts.append(role_system_prompt + "\n\n")
    else:
        prompt_parts.append(
            "You are an expert Debian 13 system optimization strategist. Based on the provided system snapshot and analysis of various items, "
            "generate a prioritized list of actionable enhancements. \n\n"
        )

    prompt_parts.append(f"System Snapshot:\n{_format_data_as_text(system_snapshot)}\n\n")
    prompt_parts.append(f"Analysis of System Items:\n{_format_data_as_text(analysis_results_list)}\n\n")

    if knowledge_base_text:
        prompt_parts.append("You are provided with the following expert guide(s) for context. Prioritize this guide when forming your strategy, especially for items related to the guide's topics:\n--- EXPERT GUIDE START ---\n" + knowledge_base_text + "\n--- EXPERT GUIDE END ---\n\n")

    if output_format == "json":
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
        prompt_parts.append("\n\n" + prompt_instructions)
    return "".join(prompt_parts)

def _format_data_as_text(data):
    if isinstance(data, dict):
        return "\n".join([f"- {key}: {value}" for key, value in data.items()])
    elif isinstance(data, list):
        if all(isinstance(i, dict) for i in data):
            formatted_items = []
            for count, item_dict in enumerate(data, 1):
                item_details = [f"  Item {count}:"]
                for k, v in item_dict.items():
                    item_details.append(f"    {k}: {str(v)[:150]}{'...' if len(str(v)) > 150 else ''}")
                formatted_items.append("\n".join(item_details))
            return "\n".join(formatted_items)
        return "\n".join([f"- {item}" for item in data])
    return str(data)

# --- Main Interface Functions ---
def query_ollama(prompt: str, model_name: str, is_json_response_expected: bool = False) -> dict | str | None:
    logger.debug(f"Querying Ollama (model: {model_name}) with prompt (first 200 chars): {prompt[:200]}...")
    api_url = config.OLLAMA_API_ENDPOINT
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    if is_json_response_expected: payload["format"] = "json"

    try:
        response = requests.post(api_url, json=payload, timeout=180)
        response.raise_for_status()
        logger.debug(f"Raw response from Ollama (model {model_name}, JSON expected: {is_json_response_expected}): {response.text[:500]}")
        return response.json() if is_json_response_expected else response.text
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from Ollama (model {model_name}): {e}", exc_info=True)
        logger.error(f"Non-JSON response received: {response.text}")
        return {"error": "Failed to decode JSON response", "details": str(e), "response_text": response.text}
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed (model {model_name}): {e}", exc_info=True)
        return {"error": f"Ollama API request failed: {str(e)}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred during Ollama query (model {model_name}): {e}", exc_info=True)
        return {"error": f"Unexpected error: {str(e)}"}


def analyze_system_item(item_content: str, item_path: str, item_type: str, model_name: str | None = None, role_name: str | None = None) -> dict | None:
    item_path_str = str(item_path)
    role_config = load_role_config(role_name) if role_name else None

    effective_model_name = model_name or config.DEFAULT_MODEL
    role_system_prompt = None
    role_kb_keywords = []
    output_format = "json" # Default for analysis

    if role_config:
        effective_model_name = role_config.get("model_name", effective_model_name)
        role_system_prompt = role_config.get("system_prompt")
        role_kb_keywords = role_config.get("knowledge_base_keywords", [])
        output_format = role_config.get("output_format", output_format)
        logger.info(f"Using role '{role_name}' for analysis. Effective model: {effective_model_name}. Output format: {output_format}")

    path_keywords = _extract_keywords_from_path(item_path_str)
    combined_keywords = list(set(path_keywords + role_kb_keywords))

    knowledge_base_text = _load_knowledge_base_content(keywords=combined_keywords, filename="sysctl_debian_guide.txt" if not combined_keywords else None)

    prompt = _build_analyze_item_prompt(item_content, item_path_str, item_type, knowledge_base_text, role_system_prompt, output_format)

    logger.info(f"Requesting analysis for {item_type} at '{item_path_str}' using model '{effective_model_name}'. Role: {role_name or 'None'}. Keywords for KB: {combined_keywords if combined_keywords else 'None'}.")

    response = query_ollama(prompt, effective_model_name, is_json_response_expected=(output_format == "json"))

    if isinstance(response, dict): return response
    if output_format != "json" and isinstance(response, str): # Handle non-JSON string output if role specified that
        return {"analysis_summary": response, "potential_issues": [], "enhancement_ideas": []} # Wrap in expected dict structure

    logger.warning(f"analyze_system_item received unexpected response type ({type(response)}) for output_format '{output_format}'. Path: {item_path_str}")
    return {"error": "Unexpected response type from Ollama query.", "analysis_summary": "Error: Non-dictionary/string response.", "potential_issues": [], "enhancement_ideas": []}


def conceive_enhancement_strategy(system_snapshot: dict, analysis_results_list: list, model_name: str | None = None, role_name: str | None = None) -> dict | None:
    role_config = load_role_config(role_name) if role_name else None

    effective_model_name = model_name or config.DEFAULT_MODEL
    role_system_prompt = None
    role_kb_keywords = []
    output_format = "json" # Default for strategy

    if role_config:
        effective_model_name = role_config.get("model_name", effective_model_name)
        role_system_prompt = role_config.get("system_prompt")
        role_kb_keywords = role_config.get("knowledge_base_keywords", [])
        output_format = role_config.get("output_format", output_format)
        logger.info(f"Using role '{role_name}' for strategy. Effective model: {effective_model_name}. Output format: {output_format}")

    path_keywords = set()
    if isinstance(analysis_results_list, list):
        for result in analysis_results_list:
            if isinstance(result, dict) and isinstance(result.get("item_path"), str):
                extracted = _extract_keywords_from_path(result["item_path"])
                for kw in extracted: path_keywords.add(kw)

    combined_keywords = list(set(list(path_keywords) + role_kb_keywords))
    knowledge_base_text = _load_knowledge_base_content(keywords=combined_keywords)

    prompt = _build_strategy_prompt(system_snapshot, analysis_results_list, knowledge_base_text, role_system_prompt, output_format)

    logger.info(f"Requesting enhancement strategy using model '{effective_model_name}'. Role: {role_name or 'None'}. Keywords for KB: {combined_keywords if combined_keywords else 'None'}.")

    response = query_ollama(prompt, effective_model_name, is_json_response_expected=(output_format == "json"))

    if isinstance(response, dict): return response
    if output_format != "json" and isinstance(response, str):
        return {"overall_strategy_summary": response, "prioritized_enhancements": []}

    logger.warning(f"conceive_enhancement_strategy received unexpected response type ({type(response)}) for output_format '{output_format}'.")
    return {"error": "Unexpected response type from Ollama query.", "overall_strategy_summary": "Error: Non-dictionary/string response.", "prioritized_enhancements": []}


def generate_code_or_modification(task_description: str, language: str, existing_code_context: str | None = None, model_name: str | None = None, role_name: str | None = None) -> str | None:
    role_config = load_role_config(role_name) if role_name else None

    effective_model_name = model_name or config.DEFAULT_MODEL
    role_system_prompt_template = None # Store template if role has one
    output_format = "json" # Default for code gen, expects {"code": "..."}

    if role_config:
        effective_model_name = role_config.get("model_name", effective_model_name)
        role_system_prompt_template = role_config.get("system_prompt") # This might contain placeholders like {language} or {task}
        output_format = role_config.get("output_format", output_format)
        logger.info(f"Using role '{role_name}' for code generation. Effective model: {effective_model_name}. Output format: {output_format}")

    # Construct the main part of the prompt
    main_prompt_content = ""
    if existing_code_context:
        main_prompt_content = (
            f"Modify the following {language} code based on the task description.\n"
            f"Task: {task_description}\n\n"
            f"Existing Code:\n```\n{existing_code_context}\n```\n\n"
        )
    else:
        main_prompt_content = (
            f"Generate a new {language} code snippet based on the task description.\n"
            f"Task: {task_description}\n\n"
        )

    # Prepend role system prompt if available
    final_prompt_parts = []
    if role_system_prompt_template:
        # Basic placeholder replacement
        role_system_prompt = role_system_prompt_template.replace("{language}", language).replace("{task_description}", task_description)
        final_prompt_parts.append(role_system_prompt + "\n\n")
    else: # Default preamble if no role
        final_prompt_parts.append(f"You are an expert {language} programmer. ")

    final_prompt_parts.append(main_prompt_content)

    if output_format == "json":
         final_prompt_parts.append(f"Your response MUST be a single, valid JSON object with a single key 'code' containing ONLY the generated/modified {language} code as a string. Ensure all special characters in the code string are properly escaped for JSON.")
    else: # For raw code output
        final_prompt_parts.append(f"Ensure your response is ONLY the {language} code block, without explanations or surrounding text, unless the task explicitly asks for it.")

    prompt = "".join(final_prompt_parts)

    logger.info(f"Requesting code generation/modification for language '{language}' using model '{effective_model_name}'. Role: {role_name or 'None'}")

    response_data = query_ollama(prompt, effective_model_name, is_json_response_expected=(output_format == "json"))

    if output_format == "json":
        if isinstance(response_data, dict):
            if "code" in response_data and isinstance(response_data["code"], str):
                return response_data["code"]
            elif "error" in response_data:
                logger.error(f"Code generation failed (JSON error from Ollama): {response_data['error']}. Details: {response_data.get('details')}")
                return None
            else:
                logger.error(f"Code generation response JSON is missing 'code' key or it's not a string. Response: {response_data}")
                return None
        else: # Not a dict when expecting JSON
            logger.error(f"Code generation query returned non-dict response when JSON was expected: {response_data}")
            return None
    elif isinstance(response_data, str): # Raw code string expected
        return response_data
    elif isinstance(response_data, dict) and "error" in response_data: # Error from query_ollama
         logger.error(f"Code generation failed (query_ollama error): {response_data['error']}. Details: {response_data.get('details')}")
         return None
    else: # Unexpected type for raw output
        logger.error(f"Code generation query returned unexpected type ({type(response_data)}) for raw output: {response_data}")
        return None


# --- Main execution for testing ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        log_format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if hasattr(logger_setup, 'LOG_FORMAT'): log_format_str = logger_setup.LOG_FORMAT
        formatter = logging.Formatter(log_format_str)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("--- Starting OllamaInterface Test (Direct Execution with Full Features) ---")

    try:
        # Create dummy roles and KB files
        roles_dir_main = pathlib.Path(config.PROJECT_ROOT) / "ai_os_enhancer" / "roles"
        roles_dir_main.mkdir(parents=True, exist_ok=True)
        kb_dir = config.PROJECT_ROOT / "knowledge_base" / "keywords"
        kb_dir.mkdir(parents=True, exist_ok=True)

        analyst_role_content = {
            "role_name": "DebianAnalyst",
            "system_prompt": "You are a meticulous Debian system analyst. Focus on identifying potential issues in the provided file.",
            "description": "A role for detailed analysis of Debian system files.",
            "knowledge_base_keywords": ["debian", "security"], # Example keywords for this role
            "output_format": "json" # Default, but explicit
        }
        coder_role_content = {
            "role_name": "PythonCoder",
            "system_prompt": "You are an expert Python 3 programmer. Generate clean, efficient, and well-commented Python code for the given task. Task: {task_description}. Language: {language}.",
            "model_name": config.DEFAULT_MODEL, # Or a specific model for coding
            "output_format": "json" # Expecting {"code": "..."}
        }
        with open(roles_dir_main / "debiananalyst.yaml", "w", encoding="utf-8") as f_role: yaml.dump(analyst_role_content, f_role)
        with open(roles_dir_main / "pythoncoder.yaml", "w", encoding="utf-8") as f_role: yaml.dump(coder_role_content, f_role)
        (kb_dir / "debian.txt").write_text("General Debian best practices document.")
        (kb_dir / "security.txt").write_text("Common security pitfalls on Linux.")
        logger.info(f"Created dummy roles and KB files for testing.")

        # Test role loading
        logger.info("\n--- Testing Role Loading ---")
        analyst_config = load_role_config("DebianAnalyst")
        assert analyst_config and analyst_config["role_name"] == "DebianAnalyst", "Failed to load DebianAnalyst role."
        logger.info(f"DebianAnalyst role loaded: {analyst_config}")
        python_coder_config = load_role_config("PythonCoder")
        assert python_coder_config, "Failed to load PythonCoder role."
        logger.info(f"PythonCoder role loaded: {python_coder_config}")


    except Exception as e: logger.error(f"Could not create test KB/Role files or test role loading: {e}", exc_info=True)

    test_model = config.DEFAULT_MODEL
    logger.info(f"\nUsing test model: {test_model}")

    logger.info("\n--- Testing analyze_system_item with Role ---")
    mock_config_content = "USER=admin\nTIMEOUT=30"
    # Pass role_name="DebianAnalyst"
    analysis_with_role = analyze_system_item(mock_config_content, "/etc/testapp/app.conf", "config", role_name="DebianAnalyst")
    if analysis_with_role and not analysis_with_role.get("error"):
        logger.info(f"Analysis with role 'DebianAnalyst': {json.dumps(analysis_with_role, indent=2)}")
    else:
        logger.error(f"Analysis with role failed or returned error: {analysis_with_role}")

    logger.info("\n--- Testing generate_code_or_modification with Role ---")
    code_gen_task = "Write a Python function that adds two numbers."
    generated_python_code = generate_code_or_modification(code_gen_task, "python", role_name="PythonCoder")
    if generated_python_code:
        logger.info(f"Generated Python code with 'PythonCoder' role:\n{generated_python_code}")
    else:
        logger.error("Code generation with role failed.")

    logger.info("\n--- (Original tests would follow, potentially adapted for roles) ---")
    logger.info("--- End of OllamaInterface Test ---")
