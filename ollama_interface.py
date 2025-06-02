import json
import requests # type: ignore
import logging
import pathlib # Added for knowledge base path handling

# Relative imports for when this module is part of the package
try:
	@@ -17,6 +18,7 @@
    import logger_setup

# Initialize logger for this module
# This logger instance will be used by _load_knowledge_base_content
if __name__ == '__main__' and not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("OllamaInterface_direct")
	@@ -25,6 +27,22 @@
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
	@@ -133,9 +151,14 @@ def analyze_system_item(item_content, item_path, item_type, model_name=None):
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
	@@ -144,7 +167,7 @@ def analyze_system_item(item_content, item_path, item_type, model_name=None):
            "Provide necessary context (e.g., function name, relevant line numbers or patterns, variable names). "
        )
    else:  # Default to config file analysis
        prompt_intro_text = (
            f"You are an expert Debian system administrator. "
            f"Analyze the following Debian configuration file (from path: {item_path}):\n\n"
            f"```plaintext\n{item_content}\n```\n\n"
	@@ -153,8 +176,15 @@ def analyze_system_item(item_content, item_path, item_type, model_name=None):
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
	@@ -165,6 +195,9 @@ def analyze_system_item(item_content, item_path, item_type, model_name=None):
        "'new_code_snippet' (string, the proposed code/config change), 'language' (string, e.g., 'bash', 'ini'). "
        "Focus on enhancing stability, security, performance, and maintainability. If no issues or ideas, return empty lists for 'potential_issues' and 'enhancement_ideas'."
    )
    full_prompt_parts.append(json_format_instruction)

    prompt = "".join(full_prompt_parts)

    logger.info(f"Requesting analysis for {item_type} at {item_path} using model {model_name}.")
    return query_ollama(prompt, model_name, is_json_response_expected=True)
	@@ -178,13 +211,31 @@ def conceive_enhancement_strategy(system_snapshot, analysis_results_list, model_
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
	@@ -201,6 +252,9 @@ def conceive_enhancement_strategy(system_snapshot, analysis_results_list, model_
        "Prioritize changes that offer significant benefits (security, performance, stability, maintainability) with manageable risk. "
        "If no enhancements are deemed necessary, 'prioritized_enhancements' should be an empty list."
    )
    prompt_parts.append(prompt_instructions)

    prompt = "".join(prompt_parts)

    logger.info(f"Requesting enhancement strategy using model {model_name}.")
    return query_ollama(prompt, model_name, is_json_response_expected=True)
	@@ -290,44 +344,57 @@ def generate_code_or_modification(task_description, language, existing_code_cont
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
	@@ -364,3 +431,4 @@ def generate_code_or_modification(task_description, language, existing_code_cont
        logger.error("Failed to generate modified python code for Test 7.")

    logger.info("--- End of OllamaInterface Test ---")
