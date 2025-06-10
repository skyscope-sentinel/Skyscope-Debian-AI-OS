import json
import requests # type: ignore
import logging
import pathlib
import re # For keyword extraction
import yaml # For loading role configurations
enhancements-suite-1
from typing import List, Dict, Any, Optional # For type hinting
from urllib.parse import urlparse # For query_ollama_chat
=======
main

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
enhancements-suite-1
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
            logger.warning(f"Role name in file ('{role_data.get('role_name')}') does not exactly match requested role_name ('{role_name}') in {role_file_path}. Using requested name for cache key, but check consistency.")
        _LOADED_ROLES_CACHE[role_name] = role_data
        return role_data
    except yaml.YAMLError as e: logger.error(f"Error parsing YAML for role '{role_name}' from {role_file_path}: {e}", exc_info=True); return None
    except FileNotFoundError: logger.error(f"Role config file not found (FileNotFoundError): {role_file_path}"); return None
    except Exception as e: logger.error(f"Unexpected error loading role '{role_name}': {e}", exc_info=True); return None

def _extract_keywords_from_path(item_path_str: str) -> list[str]:
    # ... (implementation from previous steps) ...
    if not item_path_str: return []
    words = re.findall(r'\b\w+\b', item_path_str.lower())
    keywords = set(words); common_parts = {'etc', 'conf', 'd', 'sh', 'config', 'default', 'lib', 'usr', 'bin', 'local', 'log', 'var'}; keywords.difference_update(common_parts)
    filename = pathlib.Path(item_path_str).name.lower()
    known_configs = {"sysctl.conf": "sysctl", "sshd_config": "sshd", "sshd_config.d": "sshd", "named.conf": "named", "resolv.conf": "resolv", "fstab": "fstab", "crontab": "cron", "sudoers": "sudo", "interfaces": "network", "grub": "grub", "grub.cfg": "grub"}
    if filename in known_configs: keywords.add(known_configs[filename])
    path_obj = pathlib.Path(item_path_str)
    for parent_part in path_obj.parts:
        if parent_part.lower() in ["network", "sysctl", "ssh", "cron", "apt", "kernel", "security", "logrotate", "ufw", "firewalld", "grub", "systemd", "nginx", "apache2", "mysql", "postgresql"]: keywords.add(parent_part.lower())
    name_part = pathlib.Path(filename).stem
    if name_part and name_part not in keywords and name_part not in common_parts: keywords.add(name_part)
    final_keywords = list(filter(None, keywords)); logger.debug(f"Extracted keywords: {final_keywords} from path: {item_path_str}"); return final_keywords


def _load_knowledge_base_content(keywords: list[str] | None = None, filename: str | None = None) -> str | None:
    # ... (implementation from previous steps) ...
    loaded_contents = []
    loaded_files_log = []
    kb_keywords_dir = config.PROJECT_ROOT / "knowledge_base" / "keywords"
=======
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

main
    if keywords:
        processed_keywords = set()
        for keyword in keywords:
            normalized_keyword = keyword.lower().strip()
            if not normalized_keyword or normalized_keyword in processed_keywords: continue
            kb_file_path = kb_keywords_dir / f"{normalized_keyword}.txt"
            try:
                if kb_file_path.is_file():
                    content = kb_file_path.read_text(encoding='utf-8')
enhancements-suite-1
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
        elif not keywords: logger.debug(f"Fallback/direct KB file '{filename}' (checked as '{kb_path}') not found.")
    if loaded_contents: logger.info(f"KB content aggregated from: {', '.join(loaded_files_log)}"); return "\n\n--- KB ---\n\n".join(loaded_contents)
    if not keywords and not filename: logger.debug("No keywords or fallback filename for KB loading.");
    return None

def _format_system_prompt(role_system_prompt_template: str, context: Dict[str, Any]) -> str:
    # ... (implementation from previous steps) ...
    try:
        return role_system_prompt_template.format(**context)
    except KeyError as e:
        logger.warning(f"Missing placeholder key {e} in role system prompt. Using raw prompt. Context keys available: {list(context.keys())}")
        return role_system_prompt_template
    except Exception as e_fmt:
        logger.error(f"Error formatting system prompt: {e_fmt}. Using raw prompt. Template: '{role_system_prompt_template[:100]}...' Context: {context}", exc_info=True)
        return role_system_prompt_template

def _validate_llm_response(response_content: Dict[str, Any], role_config: Dict[str, Any], role_name: str) -> tuple[bool, Optional[List[str]]]:
    # ... (implementation from previous steps, including dot-notation handling) ...
    if not isinstance(response_content, dict):
        logger.error(f"Validation Error: LLM response content for role '{role_name}' is not a dictionary. Type: {type(response_content)}. Content: {str(response_content)[:200]}")
        return False, ["__response_content_not_dict__"]
    expected_keys_paths = role_config.get("expected_llm_output_keys")
    if isinstance(expected_keys_paths, list):
        missing_paths = []
        for key_path_str in expected_keys_paths:
            path_segments = key_path_str.split('.')
            current_level = response_content; found = True
            for segment in path_segments:
                if isinstance(current_level, dict) and segment in current_level: current_level = current_level[segment]
                else: missing_paths.append(key_path_str); found = False; break
            if not found: continue
        if missing_paths:
            logger.error(f"LLM response content for role '{role_name}' missing expected keys/paths: {missing_paths}. Response content: {json.dumps(response_content, indent=2)}")
            return False, missing_paths
    return True, None

def _build_analyze_item_prompt(item_content: str, item_path: str, item_type: str, knowledge_base_text: str | None, formatted_role_system_prompt: str | None, output_format: str = "json") -> str:
    # ... (implementation from previous steps) ...
    full_prompt_parts = []
    main_instruction = ""
    if formatted_role_system_prompt:
        full_prompt_parts.append(formatted_role_system_prompt)
        if "{item_content}" not in formatted_role_system_prompt: # Check if content placeholder was used
            if item_type == "script": main_instruction = f"\n\n--- Specific Item for Analysis ---\nScript Path: '{item_path}'\n```bash\n{item_content}\n```"
            else: main_instruction = f"\n\n--- Specific Item for Analysis ---\nConfig Path: '{item_path}'\n```plaintext\n{item_content}\n```"
    else:
        if item_type == "script": main_instruction = (f"You are an expert Debian system administrator and Bash script analyzer. Analyze the following Bash script (from path: {item_path}):\n\n```bash\n{item_content}\n```\n\nYour analysis should identify potential issues and suggest specific, actionable enhancement ideas. Provide context.")
        else: main_instruction = (f"You are an expert Debian system administrator. Analyze the following Debian configuration file (from path: {item_path}):\n\n```plaintext\n{item_content}\n```\n\nYour analysis should identify potential issues and suggest specific, actionable enhancement ideas. Provide context.")
    full_prompt_parts.append(main_instruction)
    if knowledge_base_text: full_prompt_parts.append(f"\n\n--- Knowledge Base Context ---\n{knowledge_base_text}\n--- End of Knowledge Base ---")
    if output_format == "json":
        json_instruction = "\n\nYour response MUST be a single, valid JSON object. If using a role that defines specific keys, adhere to that role's expected output structure. Otherwise, for generic analysis, use keys: 'analysis_summary', 'potential_issues', 'enhancement_ideas' (each idea a dict with 'idea_description', 'justification', 'suggested_change_type', 'target_pattern', 'new_code_snippet', 'language')."
        full_prompt_parts.append(json_instruction)
    return "\n".join(full_prompt_parts)

def _build_strategy_prompt(system_snapshot: dict, analysis_results_list: list, knowledge_base_text: str | None, formatted_role_system_prompt: str | None, output_format: str = "json") -> str:
    # ... (implementation from previous steps) ...
    prompt_parts = []
    if formatted_role_system_prompt: prompt_parts.append(formatted_role_system_prompt)
    else: prompt_parts.append("You are an expert Debian 13 system optimization strategist. Generate a prioritized list of actionable enhancements based on the following system snapshot and analyses.")
    prompt_parts.append(f"\n\n--- System Snapshot ---\n{_format_data_as_text(system_snapshot)}")
    prompt_parts.append(f"\n\n--- Analysis of System Items ---\n{_format_data_as_text(analysis_results_list)}")
    if knowledge_base_text: prompt_parts.append(f"\n\n--- Knowledge Base Context ---\n{knowledge_base_text}\n--- End of Knowledge Base ---")
    if output_format == "json":
        json_instruction = "\n\nYour response MUST be a single, valid JSON object with keys: 'overall_strategy_summary' and 'prioritized_enhancements' (list of dicts, each with 'item_path', 'item_type', 'enhancement_description', 'justification', 'estimated_impact', 'change_type', 'target_criteria', 'proposed_change_snippet', 'verification_steps', 'rollback_steps', and potentially 'natural_language_task' if change_type is 'shell_task')."
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
    # ... (existing implementation) ...
    logger.debug(f"Querying Ollama /api/generate (model: {model_name}) with prompt (first 200 chars): {prompt[:200]}...")
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    if is_json_response_expected: payload["format"] = "json"
    raw_text_response = ""
    try:
        response = requests.post(config.OLLAMA_API_ENDPOINT, json=payload, timeout=180)
        raw_text_response = response.text
        response.raise_for_status()
        logger.debug(f"Raw response from Ollama /api/generate (model {model_name}, JSON: {is_json_response_expected}): {raw_text_response[:500]}")
        if is_json_response_expected: return response.json()
        return raw_text_response
    except json.JSONDecodeError as e: logger.error(f"JSON decode failed for /api/generate (model {model_name}): {e}. Response: {raw_text_response}", exc_info=True); return {"error": "JSONDecodeError", "details": str(e), "response_text": raw_text_response}
    except requests.exceptions.RequestException as e: logger.error(f"API request failed for /api/generate (model {model_name}): {e}", exc_info=True); return {"error": "RequestException", "details": str(e)}
    except Exception as e: logger.error(f"Unexpected Ollama query error for /api/generate (model {model_name}): {e}", exc_info=True); return {"error": "UnexpectedError", "details": str(e)}

# New function to be added
def query_ollama_chat(model_name: str, messages: List[Dict[str, str]], instruct_llm_to_return_json_content: bool = False) -> Optional[Dict[str, Any]]:
    """
    Sends a chat-style request to Ollama's /api/chat endpoint.
    The entire response from /api/chat is a JSON object.
    If instruct_llm_to_return_json_content is True, it also sets 'format: json' in the payload
    and attempts to parse the 'content' field of the assistant's message as JSON.
    """
    logger.debug(f"Querying Ollama /api/chat (model: {model_name}) with {len(messages)} messages. Instruct JSON content: {instruct_llm_to_return_json_content}")

    try:
        parsed_config_endpoint = urlparse(config.OLLAMA_API_ENDPOINT)
        # Construct base_url like http://host:port
        base_url_for_chat = f"{parsed_config_endpoint.scheme}://{parsed_config_endpoint.netloc}"
        chat_api_url = f"{base_url_for_chat}/api/chat" # Standard /api/chat path
        logger.debug(f"Derived chat API URL: {chat_api_url}")
    except Exception as e:
        logger.error(f"Could not derive chat API URL from OLLAMA_API_ENDPOINT ('{config.OLLAMA_API_ENDPOINT}'): {e}")
        return {"error": "Invalid OLLAMA_API_ENDPOINT for chat.", "message": {"role":"assistant", "content":""}}

    payload = { "model": model_name, "messages": messages, "stream": False }
    if instruct_llm_to_return_json_content:
        payload["format"] = "json"

    raw_response_text = ""
    try:
        response = requests.post(chat_api_url, json=payload, timeout=180)
        raw_response_text = response.text
        response.raise_for_status()
        response_json = response.json()
        logger.debug(f"Raw JSON response from Ollama /api/chat (model {model_name}): {json.dumps(response_json, indent=2)}")

        if instruct_llm_to_return_json_content and response_json.get("message", {}).get("role") == "assistant":
            msg_content_str = response_json["message"].get("content", "")
            try:
                parsed_content = json.loads(msg_content_str)
                response_json["message"]["content"] = parsed_content
                logger.debug("Successfully parsed assistant's message content string as JSON.")
            except json.JSONDecodeError:
                logger.warning(f"LLM was instructed for JSON content, but assistant's message.content is not valid JSON: {msg_content_str[:200]}")
                # Keep response_json as is; the content is a string. AgentManager's validation will handle it.

        return response_json

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama /api/chat request failed (model {model_name}): {e}", exc_info=True)
        return {"error": f"Ollama /api/chat request failed: {str(e)}", "message": {"role":"assistant", "content":""}}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode main JSON structure from Ollama /api/chat (model {model_name}): {e}. Response: {raw_response_text}", exc_info=True)
        return {"error": "Failed to decode main JSON from /api/chat", "details": str(e), "response_text": raw_response_text, "message": {"role":"assistant", "content":""}}
    except Exception as e:
        logger.error(f"Unexpected error in query_ollama_chat (model {model_name}): {e}", exc_info=True)
        return {"error": f"Unexpected error in query_ollama_chat: {str(e)}", "message": {"role":"assistant", "content":""}}


def analyze_system_item(item_content: str, item_path: str, item_type: str, model_name: str | None = None, role_name: str | None = None) -> dict | None:
    # ... (implementation from previous steps, uses _validate_llm_response) ...
    item_path_str = str(item_path); role_cfg = load_role_config(role_name) if role_name else None
    eff_model = model_name or (role_cfg.get("model_name") if role_cfg else None) or config.DEFAULT_MODEL
    out_fmt = (role_cfg.get("output_format") if role_cfg else "json"); is_json = out_fmt == "json"
    fmt_sys_prompt = None
    if role_cfg and (sys_prompt_template := role_cfg.get("system_prompt")):
        prompt_ctx = {"item_path": item_path_str, "item_type": item_type, "item_content": item_content}
        fmt_sys_prompt = _format_system_prompt(sys_prompt_template, prompt_ctx)
    kb_keywords = list(set(_extract_keywords_from_path(item_path_str) + (role_cfg.get("knowledge_base_keywords", []) if role_cfg else [])))
    kb_text = _load_knowledge_base_content(keywords=kb_keywords, filename="sysctl_debian_guide.txt" if not kb_keywords else None)
    prompt = _build_analyze_item_prompt(item_content, item_path_str, item_type, kb_text, fmt_sys_prompt, out_fmt)
    logger.info(f"Requesting analysis: path='{item_path_str}', role='{role_name or 'Default'}', model='{eff_model}'.")
    response = query_ollama(prompt, eff_model, is_json_response_expected=is_json)
    if isinstance(response, dict):
        if "error" in response: return response
        if role_cfg and is_json:
            valid, missing = _validate_llm_response(response, role_cfg, role_name)
            if not valid: return {"error": "LLM response missing expected keys", "missing_keys": missing, "response_data": response}
        return response
    if not is_json and isinstance(response, str): return {"analysis_summary": response}
    logger.warning(f"Analyze: Unexpected response type {type(response)} for format '{out_fmt}'. Path: {item_path_str}"); return {"error": "Unexpected response type", "response_data": str(response)}


def conceive_enhancement_strategy(system_snapshot: dict, analysis_results_list: list, model_name: str | None = None, role_name: str | None = None) -> dict | None:
    # ... (implementation from previous steps, uses _validate_llm_response) ...
    role_cfg = load_role_config(role_name) if role_name else None
    eff_model = model_name or (role_cfg.get("model_name") if role_cfg else None) or config.DEFAULT_MODEL
    out_fmt = (role_cfg.get("output_format") if role_cfg else "json"); is_json = out_fmt == "json"
    fmt_sys_prompt = role_cfg.get("system_prompt") if role_cfg else None
    path_kws = set()
    if isinstance(analysis_results_list, list):
        for res in analysis_results_list:
            if isinstance(res, dict) and (ip_str := res.get("item_path")) and isinstance(ip_str, str):
                for kw in _extract_keywords_from_path(ip_str): path_kws.add(kw)
    combined_kws = list(set(list(path_kws) + (role_cfg.get("knowledge_base_keywords", []) if role_cfg else [])))
    kb_text = _load_knowledge_base_content(keywords=combined_kws)
    prompt = _build_strategy_prompt(system_snapshot, analysis_results_list, kb_text, fmt_sys_prompt, out_fmt)
    logger.info(f"Requesting strategy: role='{role_name or 'Default'}', model='{eff_model}'.")
    response = query_ollama(prompt, eff_model, is_json_response_expected=is_json)
    if isinstance(response, dict):
        if "error" in response: return response
        if role_cfg and is_json:
            valid, missing = _validate_llm_response(response, role_cfg, role_name)
            if not valid: return {"error": "LLM response missing expected keys", "missing_keys": missing, "response_data": response}
        return response
    if not is_json and isinstance(response, str): return {"overall_strategy_summary": response}
    logger.warning(f"Strategy: Unexpected response type {type(response)} for format '{out_fmt}'."); return {"error": "Unexpected response type", "response_data": str(response)}


def generate_code_or_modification(task_description: str, language: str, existing_code_context: str | None = None, model_name: str | None = None, role_name: str | None = None) -> str | None:
    # ... (implementation from previous steps, uses _validate_llm_response) ...
    role_cfg = load_role_config(role_name) if role_name else None
    eff_model = model_name or (role_cfg.get("model_name") if role_cfg else None) or config.DEFAULT_MODEL
    out_fmt = (role_cfg.get("output_format") if role_cfg else "json"); is_json = out_fmt == "json"
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
            if role_cfg:
                valid, missing = _validate_llm_response(response, role_cfg, role_name)
                if not valid: return None
            if "code" in response and isinstance(response["code"], str): return response["code"]
            logger.error(f"Code gen JSON response missing 'code' key or not string: {response}"); return None
        logger.error(f"Code gen expected JSON, got {type(response)}: {response}"); return None
    elif isinstance(response, str): return response
    logger.error(f"Code gen unexpected response: {response}"); return None


def generate_shell_command(natural_language_task: str, role_name: str = "ShellCommandGenerator") -> dict | None:
    # ... (implementation from previous steps, uses _validate_llm_response) ...
    role_cfg = load_role_config(role_name)
    if not role_cfg: return {"error": f"Role config '{role_name}' not loaded."}
    eff_model = role_cfg.get("model_name", config.DEFAULT_MODEL)
    sys_prompt_template = role_cfg.get("system_prompt", "You are a shell command generator.")
    prompt_for_llm = f"{sys_prompt_template}\n\nUser Task: {natural_language_task}"
    kb_keywords = role_cfg.get("knowledge_base_keywords", [])
    kb_text = _load_knowledge_base_content(keywords=kb_keywords)
    if kb_text: prompt_for_llm += f"\n\nRelevant Knowledge Base Context:\n{kb_text}"
    out_fmt = role_cfg.get("output_format", "json"); is_json = (out_fmt == "json")
    logger.info(f"Requesting shell command: task='{natural_language_task}', role='{role_name}', model='{eff_model}'.")
    response = query_ollama(prompt=prompt_for_llm, model_name=eff_model, is_json_response_expected=is_json) # ShellCommandGenerator uses /api/generate
    if isinstance(response, dict):
        if "error" in response: logger.error(f"Shell cmd gen failed (Ollama error): {response}. Task: '{natural_language_task}'"); return response
        if is_json and role_cfg:
            valid, missing = _validate_llm_response(response, role_cfg, role_name)
            if not valid: return {"error": "LLM response missing expected keys for ShellCommandGenerator", "missing_keys": missing, "response_data": response}
        return response
    if not is_json and isinstance(response, str):
        logger.warning(f"Shell cmd gen got raw string for role '{role_name}' (JSON expected).")
        return {"generated_command": response, "safety_notes": ["Raw output, format not standard."]}
    logger.error(f"Shell cmd gen unexpected response: {response}. Task: '{natural_language_task}'")
    return {"error": "Unexpected response from LLM for shell command", "response_data": str(response)}


if __name__ == '__main__':
    # ... (content from previous step, ensuring roles and KBs for testing are created) ...
    logger.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(); console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')); logger.addHandler(console_handler)
    logger.info("--- Starting OllamaInterface Test (Direct Execution with Chat and Validation Logic) ---")

    try:
        roles_dir = config.PROJECT_ROOT / "ai_os_enhancer" / "roles"
        roles_dir.mkdir(parents=True, exist_ok=True)
        kb_dir = config.PROJECT_ROOT / "knowledge_base" / "keywords"
        kb_dir.mkdir(parents=True, exist_ok=True)
        # Define and write role files
        role_files_content_data = {
            "generic_system_item_analyzer": {
                "role_name": "GenericSystemItemAnalyzer", "version": "1.1.0", "author": "SkyscopeAI/Jules",
                "system_prompt": "Analyze {item_type} at '{item_path}'. Content:\n{item_content}",
                "expected_llm_output_keys": ["analysis_summary", "potential_issues", "enhancement_ideas"]
            },
            "python_code_generator": {
                "role_name": "PythonCodeGenerator", "version": "1.1.0", "author": "SkyscopeAI/Jules",
                "system_prompt": "Lang: {language}. Task: {task_description}. Code:\n{existing_code_context}\nJSON: {'code': '...'}",
                "output_format": "json", "expected_llm_output_keys": ["code"]
            },
            "shell_command_generator": { # Ensure this matches the refined version
                "role_name": "ShellCommandGenerator", "version": "1.1.0", "author": "SkyscopeAI/Jules",
                "description": "Translates NL to shell commands with structured risk assessment.",
                "system_prompt": "You are an expert Debian 13 shell command generator...\n(full prompt as defined in previous steps, including JSON structure for output with risk_assessment object)... Respond in JSON.",
                "knowledge_base_keywords": ["bash", "debian_cli", "apt_commands", "systemd_cli", "security_best_practices"],
                "output_format": "json",
                "expected_llm_output_keys": ["task_description", "generated_command", "risk_assessment", "risk_assessment.risk_level", "risk_assessment.operation_type", "risk_assessment.requires_privileges", "prerequisites", "safety_notes", "clarifications_needed", "alternatives"],
            }
        }
        # Simplified shell_command_generator prompt for __main__ test brevity
        role_files_content_data["shell_command_generator"]["system_prompt"] = "You are a test shell command generator. User task: {natural_language_task}. Respond in JSON with keys: task_description, generated_command, risk_assessment (with risk_level, operation_type, requires_privileges), prerequisites, safety_notes, clarifications_needed, alternatives."


        for name, content in role_files_content_data.items():
            with open(roles_dir / f"{name.lower().replace('_', '')}.yaml", "w", encoding="utf-8") as f: yaml.dump(content, f)
        for kw in ["debian", "best_practices", "python", "coding_best_practices", "bash", "debian_cli", "apt_commands", "systemd_cli", "security_best_practices"]:
            (kb_dir / f"{kw}.txt").write_text(f"# Dummy KB for {kw}\n")
        logger.info(f"Created/Updated dummy roles and KB files in {roles_dir} and {kb_dir}")
    except Exception as e: logger.error(f"Error setting up test files: {e}", exc_info=True)

    logger.info("\n--- Testing query_ollama_chat ---")
    chat_messages = [
        {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    chat_response = query_ollama_chat(config.DEFAULT_MODEL, chat_messages)
    if chat_response and not chat_response.get("error"):
        logger.info(f"Chat API response: {json.dumps(chat_response, indent=2)}")
        if chat_response.get("message", {}).get("content"):
             logger.info(f"Chatbot says: {chat_response['message']['content']}")
    else:
        logger.error(f"query_ollama_chat failed: {chat_response}")

    logger.info("\n--- Testing query_ollama_chat with JSON content instruction ---")
    chat_messages_json = [
        {"role": "system", "content": "You are a helpful assistant. Respond in JSON format with a key 'capital'"},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    chat_response_json = query_ollama_chat(config.DEFAULT_MODEL, chat_messages_json, instruct_llm_to_return_json_content=True)
    if chat_response_json and not chat_response_json.get("error"):
        logger.info(f"Chat API response (expecting JSON content): {json.dumps(chat_response_json, indent=2)}")
        if isinstance(chat_response_json.get("message", {}).get("content"), dict):
            logger.info(f"Chatbot's parsed JSON content: {chat_response_json['message']['content']}")
        else:
            logger.warning(f"Chatbot's content was not a parsed dict: {chat_response_json.get('message', {}).get('content')}")
    else:
        logger.error(f"query_ollama_chat with JSON instruction failed: {chat_response_json}")


    logger.info("\n--- Testing generate_code_or_modification with Role (PythonCodeGenerator - with placeholders & validation) ---")
    py_code_task = "create a python function that adds two numbers and returns the sum"
    py_code_context = "# No existing code, create new."
    py_code = generate_code_or_modification(py_code_task, "python", existing_code_context=py_code_context, role_name="PythonCodeGenerator")
    if py_code : logger.info(f"Generated Python Code (PythonCodeGenerator):\n{py_code}")
    else: logger.error("Python code generation with role failed or returned None.")
=======
                    loaded_contents.append(content)
                    loaded_files_log.append(kb_file_path.name)
                    logger.info(f"Successfully loaded KB file: '{kb_file_path.name}' for keyword '{normalized_keyword}'")
                    processed_keywords.add(normalized_keyword)
                else: logger.debug(f"KB file not found for keyword '{normalized_keyword}': {kb_file_path}")
            except Exception as e: logger.error(f"Error loading KB file {kb_file_path} for '{normalized_keyword}': {e}", exc_info=True)

    if filename and not loaded_contents:
        fallback_filename_to_check = filename
        if filename == "sysctl_debian_guide.txt": fallback_filename_to_check = "sysctl.txt"

        kb_path = kb_keywords_dir / fallback_filename_to_check
        if not kb_path.is_file(): kb_path = config.PROJECT_ROOT / "knowledge_base" / filename

        if kb_path.is_file():
            try:
                logger.info(f"Loading direct/fallback knowledge base: {kb_path.name} from {kb_path.parent}")
                loaded_contents.append(kb_path.read_text(encoding='utf-8'))
                loaded_files_log.append(kb_path.name)
            except Exception as e: logger.error(f"Error loading direct/fallback KB {kb_path.name}: {e}", exc_info=True)
        elif not keywords:
            logger.info(f"Fallback/direct KB file '{filename}' (checked as '{kb_path}') not found.")

    if loaded_contents:
        logger.info(f"Knowledge base content aggregated from: {', '.join(loaded_files_log)}")
        return "\n\n--- KB: Additional Context Block ---\n\n".join(loaded_contents)
    if not keywords and not filename:
         logger.info("No keywords or fallback filename provided for knowledge base loading.")
    return None

# --- Prompt Construction Helpers ---
def _build_analyze_item_prompt(item_content: str, item_path: str, item_type: str, knowledge_base_text: str | None, role_system_prompt: str | None, output_format: str = "json") -> str:
    full_prompt_parts = []

    if role_system_prompt:
        full_prompt_parts.append(role_system_prompt + "\n\n")

    if item_type == "script":
        full_prompt_parts.append(
            f"Analyze the following Bash script (from path: {item_path}):\n\n"
            f"```bash\n{item_content}\n```\n\n"
            "Your analysis should identify potential issues (e.g., bugs, security vulnerabilities, performance bottlenecks, non-standard practices) "
            "and suggest specific, actionable enhancement ideas (e.g., code changes for optimization, security hardening, improved error handling, better logging, or POSIX compliance). "
            "Provide necessary context (e.g., function name, relevant line numbers or patterns, variable names). "
        )
    else:
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
            # ... (rest of the detailed JSON instruction)
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
    role_config_loaded = load_role_config(role_name) if role_name else None
    effective_model_name = model_name or config.DEFAULT_MODEL
    role_system_prompt = None
    role_kb_keywords = []
    output_format = "json"
    if role_config_loaded:
        effective_model_name = role_config_loaded.get("model_name", effective_model_name)
        role_system_prompt = role_config_loaded.get("system_prompt")
        role_kb_keywords = role_config_loaded.get("knowledge_base_keywords", [])
        output_format = role_config_loaded.get("output_format", output_format)
        logger.info(f"Using role '{role_name}' for analysis. Effective model: {effective_model_name}. Output format: {output_format}")
    path_keywords = _extract_keywords_from_path(item_path_str)
    combined_keywords = list(set(path_keywords + role_kb_keywords))
    knowledge_base_text = _load_knowledge_base_content(keywords=combined_keywords, filename="sysctl_debian_guide.txt" if not combined_keywords else None)
    prompt = _build_analyze_item_prompt(item_content, item_path_str, item_type, knowledge_base_text, role_system_prompt, output_format)
    logger.info(f"Requesting analysis for {item_type} at '{item_path_str}' using model '{effective_model_name}'. Role: {role_name or 'None'}. Keywords for KB: {combined_keywords if combined_keywords else 'None'}.")
    response = query_ollama(prompt, effective_model_name, is_json_response_expected=(output_format == "json"))
    if isinstance(response, dict): return response
    if output_format != "json" and isinstance(response, str):
        return {"analysis_summary": response, "potential_issues": [], "enhancement_ideas": []}
    logger.warning(f"analyze_system_item received unexpected response type ({type(response)}) for output_format '{output_format}'. Path: {item_path_str}")
    return {"error": "Unexpected response type from Ollama query.", "analysis_summary": "Error: Non-dictionary/string response.", "potential_issues": [], "enhancement_ideas": []}

def conceive_enhancement_strategy(system_snapshot: dict, analysis_results_list: list, model_name: str | None = None, role_name: str | None = None) -> dict | None:
    role_config_loaded = load_role_config(role_name) if role_name else None
    effective_model_name = model_name or config.DEFAULT_MODEL
    role_system_prompt = None
    role_kb_keywords = []
    output_format = "json"
    if role_config_loaded:
        effective_model_name = role_config_loaded.get("model_name", effective_model_name)
        role_system_prompt = role_config_loaded.get("system_prompt")
        role_kb_keywords = role_config_loaded.get("knowledge_base_keywords", [])
        output_format = role_config_loaded.get("output_format", output_format)
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
    role_config_loaded = load_role_config(role_name) if role_name else None
    effective_model_name = model_name or config.DEFAULT_MODEL
    role_system_prompt_template = None
    output_format = "json"
    if role_config_loaded:
        effective_model_name = role_config_loaded.get("model_name", effective_model_name)
        role_system_prompt_template = role_config_loaded.get("system_prompt")
        output_format = role_config_loaded.get("output_format", output_format)
        logger.info(f"Using role '{role_name}' for code generation. Effective model: {effective_model_name}. Output format: {output_format}")
    main_prompt_content = f"Modify the following {language} code based on the task description.\nTask: {task_description}\n\nExisting Code:\n```\n{existing_code_context}\n```\n\n" if existing_code_context else f"Generate a new {language} code snippet based on the task description.\nTask: {task_description}\n\n"
    final_prompt_parts = []
    if role_system_prompt_template:
        role_system_prompt = role_system_prompt_template.replace("{language}", language).replace("{task_description}", task_description)
        final_prompt_parts.append(role_system_prompt + "\n\n")
    else: final_prompt_parts.append(f"You are an expert {language} programmer. ")
    final_prompt_parts.append(main_prompt_content)
    if output_format == "json":
         final_prompt_parts.append(f"Your response MUST be a single, valid JSON object with a single key 'code' containing ONLY the generated/modified {language} code as a string. Ensure all special characters in the code string are properly escaped for JSON.")
    else: final_prompt_parts.append(f"Ensure your response is ONLY the {language} code block, without explanations or surrounding text, unless the task explicitly asks for it.")
    prompt = "".join(final_prompt_parts)
    logger.info(f"Requesting code generation/modification for language '{language}' using model '{effective_model_name}'. Role: {role_name or 'None'}")
    response_data = query_ollama(prompt, effective_model_name, is_json_response_expected=(output_format == "json"))
    if output_format == "json":
        if isinstance(response_data, dict):
            if "code" in response_data and isinstance(response_data["code"], str): return response_data["code"]
            elif "error" in response_data: logger.error(f"Code generation failed (JSON error from Ollama): {response_data['error']}. Details: {response_data.get('details')}"); return None
            else: logger.error(f"Code generation response JSON is missing 'code' key or it's not a string. Response: {response_data}"); return None
        else: logger.error(f"Code generation query returned non-dict response when JSON was expected: {response_data}"); return None
    elif isinstance(response_data, str): return response_data
    elif isinstance(response_data, dict) and "error" in response_data: logger.error(f"Code generation failed (query_ollama error): {response_data['error']}. Details: {response_data.get('details')}"); return None
    else: logger.error(f"Code generation query returned unexpected type ({type(response_data)}) for raw output: {response_data}"); return None

def generate_shell_command(natural_language_task: str, role_name: str = "ShellCommandGenerator") -> dict | None:
    """
    Translates a natural language task into a shell command using a specified LLM role.
    """
    logger.info(f"Attempting to generate shell command for task: '{natural_language_task}' using role '{role_name}'")
    role_config_loaded = load_role_config(role_name)
    if not role_config_loaded:
        logger.error(f"Could not load role config for '{role_name}'. Cannot generate shell command.")
        return {"error": f"Role config for '{role_name}' not loaded.", "generated_command": None, "safety_notes": [], "alternatives": []}

    effective_model_name = role_config_loaded.get("model_name", config.DEFAULT_MODEL)
    role_system_prompt = role_config_loaded.get("system_prompt", "You are a helpful shell command assistant.") # Fallback system prompt

    # The role_system_prompt for ShellCommandGenerator is designed to frame the task.
    # The natural_language_task is the user's specific request.
    # We combine these to form the final prompt for the LLM.
    # The system prompt already includes "The user will provide a task description... Now, process the user's actual task..."
    # So, the natural_language_task should be appended as the user's turn.
    prompt_for_llm = f"{role_system_prompt}\n\nUser Task: {natural_language_task}"

    role_kb_keywords = role_config_loaded.get("knowledge_base_keywords", [])
    knowledge_base_text = _load_knowledge_base_content(keywords=role_kb_keywords)
    if knowledge_base_text:
        prompt_for_llm = f"{prompt_for_llm}\n\nRelevant Knowledge Base Context:\n{knowledge_base_text}"

    output_format = role_config_loaded.get("output_format", "json")
    is_json_expected = (output_format == "json")

    logger.info(f"Requesting shell command generation using model '{effective_model_name}' with role '{role_name}'. JSON expected: {is_json_expected}")

    response = query_ollama(
        prompt=prompt_for_llm,
        model_name=effective_model_name,
        is_json_response_expected=is_json_expected
    )

    if is_json_expected:
        if isinstance(response, dict):
            if "error" in response:
                logger.error(f"Shell command generation failed (Ollama error): {response.get('error')}. Task: '{natural_language_task}'")
                return response # Return the error dict from query_ollama

            expected_keys = ["generated_command", "safety_notes"] # "task_description", "alternatives" are also good
            if not all(key in response for key in expected_keys):
                logger.error(f"Shell command generation response missing some expected keys ({expected_keys}). Response: {response}")
                # Return a structured error, but include what was received.
                return {"error": "Response missing expected keys", "response_data": response, "generated_command": response.get("generated_command"), "safety_notes": response.get("safety_notes", ["Response format error."])}
            return response
        else:
            logger.error(f"Shell command generation expected JSON but received type {type(response)}. Task: '{natural_language_task}'. Response: {response}")
            return {"error": f"Expected JSON, got {type(response)}", "response_data": str(response), "generated_command": None, "safety_notes": ["Type error from LLM."]}
    elif isinstance(response, str):
        logger.warning(f"Shell command generation received raw string output for role '{role_name}' (JSON usually expected). Returning as raw string in a dict.")
        return {"generated_command": response, "safety_notes": ["Raw output, format not as standard JSON shell command structure."], "task_description": natural_language_task, "alternatives": []}

    logger.error(f"Unexpected response from shell command generation. Response: {response}")
    return {"error": "Unexpected response from LLM", "response_data": str(response), "generated_command": None, "safety_notes": ["Unexpected LLM response."]}


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
        roles_dir_main = pathlib.Path(config.PROJECT_ROOT) / "ai_os_enhancer" / "roles"
        roles_dir_main.mkdir(parents=True, exist_ok=True)
        kb_dir = config.PROJECT_ROOT / "knowledge_base" / "keywords"
        kb_dir.mkdir(parents=True, exist_ok=True)

        # Dummy roles for general testing
        analyst_role_content = {"role_name": "DebianAnalyst", "system_prompt": "You are a meticulous Debian system analyst...", "knowledge_base_keywords": ["debian", "security"], "output_format": "json"}
        coder_role_content = {"role_name": "PythonCoder", "system_prompt": "You are an expert Python 3 programmer for task: {task_description}.", "output_format": "json"}
        with open(roles_dir_main / "debiananalyst.yaml", "w", encoding="utf-8") as f: yaml.dump(analyst_role_content, f)
        with open(roles_dir_main / "pythoncoder.yaml", "w", encoding="utf-8") as f: yaml.dump(coder_role_content, f)

        # ShellCommandGenerator role definition
        shell_generator_role_content = {
            "role_name": "ShellCommandGenerator",
            "description": "Translates natural language task descriptions into Debian 13 shell commands, providing the command and safety considerations.",
            "system_prompt": """You are an expert Debian 13 system administrator and shell command generator.
Your task is to translate a given natural language instruction into an accurate and safe shell command or a sequence of commands.
The user will provide a task description. You must respond with a single, valid JSON object containing the following keys:
- "task_description": A restatement or summary of the input task.
- "generated_command": A string containing the shell command(s) you recommend for accomplishing the task on a Debian 13 system.
- "safety_notes": A list of strings, where each string is a brief note about potential risks or prerequisites.
- "alternatives": A list of strings, describing alternative commands or approaches.

Prioritize correctness, safety, and common Debian utilities.""",
            "knowledge_base_keywords": ["bash", "debian_cli"],
            "output_format": "json"
        }
        with open(roles_dir_main / "shellcommandgenerator.yaml", "w", encoding="utf-8") as f: yaml.dump(shell_generator_role_content, f)

        # Dummy KB files
        (kb_dir / "debian.txt").write_text("General Debian best practices document.")
        (kb_dir / "security.txt").write_text("Common security pitfalls on Linux.")
        (kb_dir / "bash.txt").write_text("Bash scripting best practices and common commands.")
        (kb_dir / "debian_cli.txt").write_text("Useful Debian command-line interface utilities.")
        logger.info(f"Created dummy roles and KB files for testing in {roles_dir_main} and {kb_dir}")

        logger.info("\n--- Testing Role Loading ---")
        for role_to_test in ["DebianAnalyst", "PythonCoder", "ShellCommandGenerator"]:
            loaded_config = load_role_config(role_to_test)
            assert loaded_config and loaded_config["role_name"] == role_to_test, f"Failed to load {role_to_test} role."
            logger.info(f"{role_to_test} role loaded successfully.")

        non_existent_role = load_role_config("NonExistentRole")
        assert non_existent_role is None, "NonExistentRole should not load."
        logger.info("Correctly returned None for 'NonExistentRole'.")

    except Exception as e: logger.error(f"Could not create test KB/Role files or test role loading: {e}", exc_info=True)

    test_model = config.DEFAULT_MODEL
    logger.info(f"\nUsing test model: {test_model} for subsequent tests.")

    # ... (original tests for analyze_system_item, generate_code_or_modification with roles can be kept or adapted)

    logger.info("\n--- Testing Shell Assistant (generate_shell_command) ---")
    shell_tasks = [
        "List all files in the /tmp directory including hidden ones.",
        "Install the package named 'htop' without asking for confirmation.",
        "Check the status of the 'ssh' service and suggest how to start it if it is inactive, and also how to enable it to start on boot."
    ]
    for i, task_desc in enumerate(shell_tasks):
        logger.info(f"\nShell Assistant Task {i+1}: {task_desc}")
        command_suggestion = generate_shell_command(task_desc) # Uses default "ShellCommandGenerator" role
        if command_suggestion and not command_suggestion.get("error"):
            logger.info(f"Suggestion:\n{json.dumps(command_suggestion, indent=2)}")
        else:
            logger.error(f"Failed to get suggestion for task '{task_desc}': {command_suggestion}")
main

    logger.info("\n--- End of OllamaInterface Test ---")
