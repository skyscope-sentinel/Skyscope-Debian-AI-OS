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
    # This assumes orchestrator.py is in ai_os_enhancer directory, so parent is project root
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
        with open(role_file_path, 'r', encoding='utf-8') as f: role_data = yaml.safe_load(f)
        if not isinstance(role_data, dict):
            logger.error(f"Role configuration in {role_file_path} is not a valid dictionary."); return None
        required_keys = ["role_name", "system_prompt"]
        if not all(key in role_data for key in required_keys):
            logger.error(f"Role config {role_file_path} missing required keys: {required_keys}"); return None
        if role_data.get("role_name") != role_name:
            logger.warning(f"Role name in file ('{role_data.get('role_name')}') does not match requested '{role_name}' in {role_file_path}.")
        _LOADED_ROLES_CACHE[role_name] = role_data
        return role_data
    except yaml.YAMLError as e: logger.error(f"Error parsing YAML for role '{role_name}' from {role_file_path}: {e}", exc_info=True); return None
    except FileNotFoundError: logger.error(f"Role config file not found (FileNotFoundError): {role_file_path}"); return None
    except Exception as e: logger.error(f"Unexpected error loading role '{role_name}': {e}", exc_info=True); return None

def _extract_keywords_from_path(item_path_str: str) -> list[str]:
    if not item_path_str: return []
    words = re.findall(r'\b\w+\b', item_path_str.lower())
    keywords = set(words); common_parts = {'etc', 'conf', 'd', 'sh', 'config', 'default', 'lib', 'usr', 'bin', 'local'}; keywords.difference_update(common_parts)
    filename = pathlib.Path(item_path_str).name.lower()
    known_configs = {"sysctl.conf": "sysctl", "sshd_config": "sshd", "named.conf": "named", "resolv.conf": "resolv", "fstab": "fstab", "crontab": "cron", "sudoers": "sudo", "interfaces": "network", "grub": "grub"}
    if filename in known_configs: keywords.add(known_configs[filename])
    path_obj = pathlib.Path(item_path_str)
    for parent_part in path_obj.parts:
        if parent_part.lower() in ["network", "sysctl", "ssh", "cron", "apt", "kernel", "security", "logrotate", "ufw", "firewalld", "grub", "systemd"]: keywords.add(parent_part.lower())
    final_keywords = list(filter(None, keywords)); logger.debug(f"Extracted keywords: {final_keywords} from path: {item_path_str}"); return final_keywords

def _load_knowledge_base_content(keywords: list[str] | None = None, filename: str | None = None) -> str | None:
    # ... (content of this function remains largely the same as before, ensuring it uses config.PROJECT_ROOT correctly) ...
    loaded_contents = []
    loaded_files_log = []
    kb_keywords_dir = config.PROJECT_ROOT / "knowledge_base" / "keywords" # config.PROJECT_ROOT is Path object
    if keywords:
        processed_keywords = set()
        for keyword in keywords:
            normalized_keyword = keyword.lower().strip()
            if not normalized_keyword or normalized_keyword in processed_keywords: continue
            kb_file_path = kb_keywords_dir / f"{normalized_keyword}.txt"
            try:
                if kb_file_path.is_file():
                    content = kb_file_path.read_text(encoding='utf-8')
                    loaded_contents.append(content); loaded_files_log.append(kb_file_path.name)
                    logger.info(f"Loaded KB file: '{kb_file_path.name}' for keyword '{normalized_keyword}'")
                    processed_keywords.add(normalized_keyword)
                else: logger.debug(f"KB file not found for keyword '{normalized_keyword}': {kb_file_path}")
            except Exception as e: logger.error(f"Error loading KB file {kb_file_path} for '{normalized_keyword}': {e}", exc_info=True)
    if filename and not loaded_contents:
        fn_to_check = "sysctl.txt" if filename == "sysctl_debian_guide.txt" else filename
        kb_path = kb_keywords_dir / fn_to_check
        if not kb_path.is_file(): kb_path = config.PROJECT_ROOT / "knowledge_base" / filename
        if kb_path.is_file():
            try:
                logger.info(f"Loading direct/fallback KB: {kb_path.name} from {kb_path.parent}")
                loaded_contents.append(kb_path.read_text(encoding='utf-8')); loaded_files_log.append(kb_path.name)
            except Exception as e: logger.error(f"Error loading direct/fallback KB {kb_path.name}: {e}", exc_info=True)
        elif not keywords: logger.info(f"Fallback/direct KB file '{filename}' (checked as '{kb_path}') not found.")
    if loaded_contents: logger.info(f"KB content aggregated from: {', '.join(loaded_files_log)}"); return "\n\n--- KB ---\n\n".join(loaded_contents)
    if not keywords and not filename: logger.info("No keywords or fallback filename for KB loading."); return None
    return None


def _format_system_prompt(role_system_prompt_template: str, context: dict) -> str:
    try:
        return role_system_prompt_template.format(**context)
    except KeyError as e:
        logger.warning(f"Missing placeholder key '{e}' in role system prompt. Using raw prompt. Context: {context}")
        return role_system_prompt_template # Return raw prompt if formatting fails

def _validate_llm_response(response: dict, role_config: dict, role_name: str) -> tuple[bool, list[str] | None]:
    if role_config.get("output_format") == "json":
        expected_keys = role_config.get("expected_llm_output_keys")
        if isinstance(expected_keys, list):
            missing_keys = [key for key in expected_keys if key not in response]
            if missing_keys:
                logger.error(f"LLM response for role '{role_name}' missing expected keys: {missing_keys}. Response: {response}")
                return False, missing_keys
    return True, None


def _build_analyze_item_prompt(item_content: str, item_path: str, item_type: str, knowledge_base_text: str | None, formatted_role_system_prompt: str | None, output_format: str = "json") -> str:
    # ... (content as before, but uses formatted_role_system_prompt)
    full_prompt_parts = []
    if formatted_role_system_prompt: # This is already formatted with placeholders
        full_prompt_parts.append(formatted_role_system_prompt)
        # Add a clear separator before appending item-specific instructions if role prompt is generic
        full_prompt_parts.append(f"\n\n--- Specific Item for Analysis ({item_type} at '{item_path}') ---")
    else: # Fallback to default prompt structure if no role_system_prompt
        if item_type == "script":
            full_prompt_parts.append(f"You are an expert Debian system administrator and Bash script analyzer. Analyze the following Bash script (from path: {item_path}):\n\n```bash\n{item_content}\n```\n\nYour analysis should identify potential issues and suggest specific, actionable enhancement ideas. Provide context.")
        else:
            full_prompt_parts.append(f"You are an expert Debian system administrator. Analyze the following Debian configuration file (from path: {item_path}):\n\n```plaintext\n{item_content}\n```\n\nYour analysis should identify potential issues and suggest specific, actionable enhancement ideas. Provide context.")

    if knowledge_base_text: full_prompt_parts.append(f"\n\n--- Knowledge Base Context ---\n{knowledge_base_text}\n--- End of Knowledge Base ---")
    if output_format == "json":
        json_instruction = "\n\nYour response MUST be a single, valid JSON object with keys: 'analysis_summary', 'potential_issues', 'enhancement_ideas' (each idea a dict with 'idea_description', 'justification', 'suggested_change_type', 'target_pattern', 'new_code_snippet', 'language')."
        full_prompt_parts.append(json_instruction)
    return "\n".join(full_prompt_parts)


def _build_strategy_prompt(system_snapshot: dict, analysis_results_list: list, knowledge_base_text: str | None, formatted_role_system_prompt: str | None, output_format: str = "json") -> str:
    # ... (content as before, but uses formatted_role_system_prompt and conditional JSON instruction) ...
    prompt_parts = []
    if formatted_role_system_prompt: prompt_parts.append(formatted_role_system_prompt)
    else: prompt_parts.append("You are an expert Debian 13 system optimization strategist. Generate a prioritized list of actionable enhancements based on the following system snapshot and analyses.")
    prompt_parts.append(f"\n\n--- System Snapshot ---\n{_format_data_as_text(system_snapshot)}")
    prompt_parts.append(f"\n\n--- Analysis of System Items ---\n{_format_data_as_text(analysis_results_list)}")
    if knowledge_base_text: prompt_parts.append(f"\n\n--- Knowledge Base Context ---\n{knowledge_base_text}\n--- End of Knowledge Base ---")
    if output_format == "json":
        json_instruction = "\n\nYour response MUST be a single, valid JSON object with keys: 'overall_strategy_summary' and 'prioritized_enhancements' (list of dicts, each with 'item_path', 'item_type', 'enhancement_description', 'justification', 'estimated_impact', 'change_type', 'target_criteria', 'proposed_change_snippet', 'verification_steps', 'rollback_steps')."
        prompt_parts.append(json_instruction)
    return "\n".join(prompt_parts)

def _format_data_as_text(data):
    # ... (existing implementation)
    if isinstance(data, dict): return "\n".join([f"- {k}: {v}" for k, v in data.items()])
    if isinstance(data, list):
        if all(isinstance(i,dict) for i in data): return "\n".join(f"  Item {c+1}:\n" + "\n".join(f"    {k}: {str(v)[:150]}" for k,v in i.items()) for c,i in enumerate(data))
        return "\n".join([f"- {i}" for i in data])
    return str(data)

def query_ollama(prompt: str, model_name: str, is_json_response_expected: bool = False) -> dict | str | None:
    # ... (existing implementation, ensure error dicts are consistent)
    logger.debug(f"Querying Ollama (model: {model_name}) with prompt (first 200 chars): {prompt[:200]}...")
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    if is_json_response_expected: payload["format"] = "json"
    try:
        response = requests.post(config.OLLAMA_API_ENDPOINT, json=payload, timeout=180)
        response.raise_for_status()
        logger.debug(f"Raw response from Ollama (model {model_name}, JSON: {is_json_response_expected}): {response.text[:500]}")
        return response.json() if is_json_response_expected else response.text
    except json.JSONDecodeError as e: logger.error(f"JSON decode failed (model {model_name}): {e}. Response: {response.text}", exc_info=True); return {"error": "JSONDecodeError", "details": str(e), "response_text": response.text}
    except requests.exceptions.RequestException as e: logger.error(f"API request failed (model {model_name}): {e}", exc_info=True); return {"error": "RequestException", "details": str(e)}
    except Exception as e: logger.error(f"Unexpected Ollama query error (model {model_name}): {e}", exc_info=True); return {"error": "UnexpectedError", "details": str(e)}


def analyze_system_item(item_content: str, item_path: str, item_type: str, model_name: str | None = None, role_name: str | None = None) -> dict | None:
    item_path_str = str(item_path)
    role_cfg = load_role_config(role_name) if role_name else None
    eff_model = model_name or (role_cfg.get("model_name") if role_cfg else None) or config.DEFAULT_MODEL
    out_fmt = (role_cfg.get("output_format") if role_cfg else "json")
    is_json = out_fmt == "json"

    fmt_sys_prompt = None
    if role_cfg and (sys_prompt_template := role_cfg.get("system_prompt")):
        prompt_ctx = {"item_path": item_path_str, "item_type": item_type}
        fmt_sys_prompt = _format_system_prompt(sys_prompt_template, prompt_ctx)

    kb_keywords = list(set(_extract_keywords_from_path(item_path_str) + (role_cfg.get("knowledge_base_keywords", []) if role_cfg else [])))
    kb_text = _load_knowledge_base_content(keywords=kb_keywords, filename="sysctl_debian_guide.txt" if not kb_keywords else None)

    prompt = _build_analyze_item_prompt(item_content, item_path_str, item_type, kb_text, fmt_sys_prompt, out_fmt)
    logger.info(f"Requesting analysis: path='{item_path_str}', role='{role_name or 'Default'}', model='{eff_model}'. KB keywords: {kb_keywords}")
    response = query_ollama(prompt, eff_model, is_json_response_expected=is_json)

    if isinstance(response, dict):
        if "error" in response: return response # Propagate error from query_ollama
        if role_cfg and is_json:
            valid, missing = _validate_llm_response(response, role_cfg, role_name)
            if not valid: return {"error": "LLM response missing expected keys", "missing_keys": missing, "response_data": response}
        return response
    if not is_json and isinstance(response, str): return {"analysis_summary": response} # Wrap raw text
    logger.warning(f"Analyze: Unexpected response type {type(response)} for format '{out_fmt}'. Path: {item_path_str}"); return {"error": "Unexpected response type"}


def conceive_enhancement_strategy(system_snapshot: dict, analysis_results_list: list, model_name: str | None = None, role_name: str | None = None) -> dict | None:
    role_cfg = load_role_config(role_name) if role_name else None
    eff_model = model_name or (role_cfg.get("model_name") if role_cfg else None) or config.DEFAULT_MODEL
    out_fmt = (role_cfg.get("output_format") if role_cfg else "json")
    is_json = out_fmt == "json"
    fmt_sys_prompt = role_cfg.get("system_prompt") if role_cfg else None # No placeholders expected for strategy role by default

    path_kws = set()
    if isinstance(analysis_results_list, list):
        for res in analysis_results_list:
            if isinstance(res, dict) and (ip_str := res.get("item_path")) and isinstance(ip_str, str):
                for kw in _extract_keywords_from_path(ip_str): path_kws.add(kw)
    combined_kws = list(set(list(path_kws) + (role_cfg.get("knowledge_base_keywords", []) if role_cfg else [])))
    kb_text = _load_knowledge_base_content(keywords=combined_kws)

    prompt = _build_strategy_prompt(system_snapshot, analysis_results_list, kb_text, fmt_sys_prompt, out_fmt)
    logger.info(f"Requesting strategy: role='{role_name or 'Default'}', model='{eff_model}'. KB keywords: {combined_kws}")
    response = query_ollama(prompt, eff_model, is_json_response_expected=is_json)

    if isinstance(response, dict):
        if "error" in response: return response
        if role_cfg and is_json:
            valid, missing = _validate_llm_response(response, role_cfg, role_name)
            if not valid: return {"error": "LLM response missing expected keys", "missing_keys": missing, "response_data": response}
        return response
    if not is_json and isinstance(response, str): return {"overall_strategy_summary": response}
    logger.warning(f"Strategy: Unexpected response type {type(response)} for format '{out_fmt}'."); return {"error": "Unexpected response type"}


def generate_code_or_modification(task_description: str, language: str, existing_code_context: str | None = None, model_name: str | None = None, role_name: str | None = None) -> str | None:
    role_cfg = load_role_config(role_name) if role_name else None
    eff_model = model_name or (role_cfg.get("model_name") if role_cfg else None) or config.DEFAULT_MODEL
    out_fmt = (role_cfg.get("output_format") if role_cfg else "json") # Default to JSON for {"code": "..."}
    is_json = out_fmt == "json"

    fmt_sys_prompt = None
    if role_cfg and (sys_prompt_template := role_cfg.get("system_prompt")):
        prompt_ctx = {"language": language, "task_description": task_description, "existing_code_context": existing_code_context or ""}
        fmt_sys_prompt = _format_system_prompt(sys_prompt_template, prompt_ctx)

    main_prompt = f"Modify the {language} code:\n```\n{existing_code_context}\n```\nBased on: {task_description}" if existing_code_context else f"Generate {language} code for: {task_description}"
    prompt = f"{fmt_sys_prompt}\n\n{main_prompt}" if fmt_sys_prompt else f"You are an expert {language} programmer. {main_prompt}"
    if is_json: prompt += f"\n\nYour response MUST be a single, valid JSON object with a single key 'code' containing ONLY the {language} code as a string."
    else: prompt += f"\n\nEnsure your response is ONLY the {language} code block."

    logger.info(f"Requesting code gen: lang='{language}', role='{role_name or 'Default'}', model='{eff_model}'")
    response = query_ollama(prompt, eff_model, is_json_response_expected=is_json)

    if is_json:
        if isinstance(response, dict):
            if "error" in response: logger.error(f"Code gen failed (Ollama error): {response}"); return None
            if role_cfg: # Validate only if role provided specific keys
                valid, missing = _validate_llm_response(response, role_cfg, role_name)
                if not valid: return None # Error already logged
            if "code" in response and isinstance(response["code"], str): return response["code"]
            logger.error(f"Code gen JSON response missing 'code' key or not string: {response}"); return None
        logger.error(f"Code gen expected JSON, got {type(response)}: {response}"); return None
    elif isinstance(response, str): return response # Raw code string
    logger.error(f"Code gen unexpected response: {response}"); return None


def generate_shell_command(natural_language_task: str, role_name: str = "ShellCommandGenerator") -> dict | None:
    role_cfg = load_role_config(role_name)
    if not role_cfg: return {"error": f"Role config '{role_name}' not loaded."} # Return error dict

    eff_model = role_cfg.get("model_name", config.DEFAULT_MODEL)
    sys_prompt_template = role_cfg.get("system_prompt", "You are a shell command generator.")
    # For ShellCommandGenerator, task is appended after the main system prompt.
    # Placeholders like {natural_language_task} are not used in its current system_prompt.
    # If they were, _format_system_prompt would be used here.
    # prompt_ctx = {"natural_language_task": natural_language_task}
    # fmt_sys_prompt = _format_system_prompt(sys_prompt_template, prompt_ctx)
    # For now, stick to appending:
    prompt_for_llm = f"{sys_prompt_template}\n\nUser Task: {natural_language_task}"

    kb_keywords = role_cfg.get("knowledge_base_keywords", [])
    kb_text = _load_knowledge_base_content(keywords=kb_keywords)
    if kb_text: prompt_for_llm += f"\n\nRelevant Knowledge Base Context:\n{kb_text}"

    out_fmt = role_cfg.get("output_format", "json")
    is_json = (out_fmt == "json")
    logger.info(f"Requesting shell command: task='{natural_language_task}', role='{role_name}', model='{eff_model}'. JSON expected: {is_json}")
    response = query_ollama(prompt=prompt_for_llm, model_name=eff_model, is_json_response_expected=is_json)

    if isinstance(response, dict):
        if "error" in response: logger.error(f"Shell cmd gen failed (Ollama error): {response}. Task: '{natural_language_task}'"); return response
        if is_json: # This validation should use role_cfg
            valid, missing = _validate_llm_response(response, role_cfg, role_name)
            if not valid: return {"error": "LLM response missing expected keys for ShellCommandGenerator", "missing_keys": missing, "response_data": response}
        return response
    if not is_json and isinstance(response, str):
        logger.warning(f"Shell cmd gen got raw string for role '{role_name}' (JSON expected).")
        return {"generated_command": response, "safety_notes": ["Raw output, format not standard."]}
    logger.error(f"Shell cmd gen unexpected response: {response}. Task: '{natural_language_task}'")
    return {"error": "Unexpected response from LLM for shell command", "response_data": str(response)}


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(); console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')); logger.addHandler(console_handler)
    logger.info("--- Starting OllamaInterface Test (Direct Execution) ---")

    try:
        roles_dir = config.PROJECT_ROOT / "ai_os_enhancer" / "roles"
        roles_dir.mkdir(parents=True, exist_ok=True)
        kb_dir = config.PROJECT_ROOT / "knowledge_base" / "keywords"
        kb_dir.mkdir(parents=True, exist_ok=True)

        # Define and write role files including expected_llm_output_keys
        role_files_content = {
            "generic_system_item_analyzer": {
                "role_name": "GenericSystemItemAnalyzer", "description": "Analyzes system items.",
                "system_prompt": "Analyze {item_type} at '{item_path}'. Focus on issues and enhancements.",
                "knowledge_base_keywords": ["debian", "best_practices"], "output_format": "json",
                "expected_llm_output_keys": ["analysis_summary", "potential_issues", "enhancement_ideas"]
            },
            "enhancement_strategist": {
                "role_name": "EnhancementStrategist", "description": "Creates enhancement strategies.",
                "system_prompt": "You are a Debian optimization strategist. Prioritize enhancements based on snapshot and analyses.",
                "knowledge_base_keywords": ["optimization", "security_best_practices"], "output_format": "json",
                "expected_llm_output_keys": ["overall_strategy_summary", "prioritized_enhancements"]
            },
            "shell_command_generator": {
                "role_name": "ShellCommandGenerator", "description": "Generates shell commands from natural language.",
                "system_prompt": "You are a Debian shell command expert. Task: Generate command. Respond in JSON with keys: task_description, generated_command, risk_assessment (with risk_level, operation_type, requires_privileges), prerequisites, setup_commands, safety_notes, clarifications_needed, alternatives.",
                "knowledge_base_keywords": ["bash", "debian_cli", "apt_commands", "systemd_cli", "security_best_practices"], "output_format": "json",
                "expected_llm_output_keys": ["task_description", "generated_command", "risk_assessment", "prerequisites", "safety_notes", "clarifications_needed", "alternatives"]
            },
            "python_code_generator": {
                "role_name": "PythonCodeGenerator", "description": "Generates Python code.",
                "system_prompt": "You are an expert Python programmer. Language: {language}. Task: {task_description}. Existing Code (if any): {existing_code_context}. Respond with JSON: {'code': 'your_code'}",
                "knowledge_base_keywords": ["python", "coding_best_practices"], "output_format": "json",
                "expected_llm_output_keys": ["code"]
            }
        }
        for name, content in role_files_content.items():
            with open(roles_dir / f"{name}.yaml", "w", encoding="utf-8") as f: yaml.dump(content, f)

        # Dummy KB files
        for kw in ["debian", "best_practices", "optimization", "security_best_practices", "bash", "debian_cli", "apt_commands", "systemd_cli", "python", "coding_best_practices"]:
            (kb_dir / f"{kw}.txt").write_text(f"# Dummy KB content for {kw}\n")
        logger.info(f"Created dummy roles and KB files in {roles_dir} and {kb_dir}")

    except Exception as e: logger.error(f"Error setting up test files: {e}", exc_info=True)

    logger.info("\n--- Testing analyze_system_item with Role (GenericSystemItemAnalyzer) ---")
    analysis = analyze_system_item("some content", "/etc/test.conf", "config", role_name="GenericSystemItemAnalyzer")
    logger.info(f"Analysis (Generic): {json.dumps(analysis, indent=2) if analysis else 'Failed'}")

    logger.info("\n--- Testing generate_code_or_modification with Role (PythonCodeGenerator) ---")
    py_code = generate_code_or_modification("create a hello world function", "python", role_name="PythonCodeGenerator")
    logger.info(f"Generated Python Code:\n{py_code if py_code else 'Failed'}")

    logger.info("\n--- Testing generate_shell_command with Role (ShellCommandGenerator) ---")
    shell_cmd_info = generate_shell_command("list files in /tmp")
    logger.info(f"Generated Shell Command Info: {json.dumps(shell_cmd_info, indent=2) if shell_cmd_info else 'Failed'}")

    logger.info("\n--- End of OllamaInterface Test ---")
