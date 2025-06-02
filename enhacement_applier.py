# ai_os_enhancer/enhancement_applier.py

import os
import shutil
import subprocess
import logging
import datetime
import pathlib
import shlex # For safely splitting command strings

# Relative imports for when this module is part of the package
try:
    from . import config
    from . import logger_setup
    from . import system_analyzer # For read_file_content
except ImportError:
    # Fallback for direct execution (e.g., if __name__ == '__main__')
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    import logger_setup
    import system_analyzer

# Initialize logger for this module
if __name__ == '__main__' and not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("EnhancementApplier_direct")
    logger.info("Running EnhancementApplier directly, basic logging configured.")
else:
    logger = logger_setup.setup_logger("EnhancementApplier")

# --- EnhancementApplier Module ---

def backup_file(file_path_str):
    """
    Creates a backup of the given file.
    Returns: String (path to backup file) or None on failure.
    """
    file_path = pathlib.Path(file_path_str)
    if not file_path.exists():
        logger.error(f"File not found, cannot backup: {file_path}")
        return None

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") # Added %f for microseconds
        abs_file_path = file_path.resolve()
        
        # Determine path relative to root or a known base if possible, to avoid overly deep backup structures
        # For simplicity, if path is absolute, take path components after anchor
        try:
            relative_path_parts = abs_file_path.relative_to(abs_file_path.anchor).parts
        except ValueError: # Handles cases like Windows paths if anchor logic differs or not under a common root for testing
            relative_path_parts = abs_file_path.parts[1:] if abs_file_path.is_absolute() else abs_file_path.parts

        backup_target_dir = config.BACKUP_BASE_PATH
        for part in relative_path_parts[:-1]: # All parts except filename
             backup_target_dir = backup_target_dir / part
        
        os.makedirs(backup_target_dir, exist_ok=True)
        
        backup_filename = f"{abs_file_path.name}.{timestamp}.bak"
        backup_path = backup_target_dir / backup_filename

        shutil.copy2(str(abs_file_path), str(backup_path))
        logger.info(f"Backed up '{abs_file_path}' to '{backup_path}'")
        return str(backup_path)
    except Exception as e:
        logger.error(f"Failed to backup file '{file_path}': {e}", exc_info=True)
        return None

def _write_file_content(file_path_str, content):
    """
    Helper to write content to a file.
    Returns: True on success, False on failure.
    """
    try:
        file_path = pathlib.Path(file_path_str)
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        logger.debug(f"Successfully wrote content to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write to file {file_path_str}: {e}", exc_info=True)
        return False

def apply_config_text_change(file_path_str, old_content_snippet, new_content, backup_path_provided):
    """
    Applies a text change to a configuration file.
    Returns: Boolean (success or failure).
    Modes: "APPEND_MODE", "PREPEND_MODE", "OVERWRITE_MODE", or replaces first occurrence.
    """
    logger.info(f"Attempting to apply config text change to: {file_path_str} (Mode/Snippet: '{old_content_snippet[:50]}{'...' if len(old_content_snippet)>50 else ''}')")
    
    # For OVERWRITE_MODE, we don't need to read current content if the file might not exist.
    # However, for other modes, reading is essential.
    if old_content_snippet != "OVERWRITE_MODE" and not os.path.exists(file_path_str):
        logger.error(f"File {file_path_str} does not exist for modification (non-OVERWRITE_MODE).")
        return False

    current_content = ""
    if old_content_snippet != "OVERWRITE_MODE": # Read only if needed
        current_content_read = system_analyzer.read_file_content(file_path_str)
        if current_content_read is None: # Check if read failed
            logger.error(f"Could not read current content of {file_path_str} to apply change.")
            return False
        current_content = current_content_read


    modified_content = None
    if old_content_snippet == "APPEND_MODE":
        modified_content = current_content + ("\n" if current_content else "") + new_content
        logger.debug("Applying change in APPEND_MODE.")
    elif old_content_snippet == "PREPEND_MODE":
        modified_content = new_content + ("\n" if current_content else "") + current_content
        logger.debug("Applying change in PREPEND_MODE.")
    elif old_content_snippet == "OVERWRITE_MODE":
        modified_content = new_content
        logger.debug("Applying change in OVERWRITE_MODE (replacing entire file).")
    elif old_content_snippet in current_content:
        modified_content = current_content.replace(old_content_snippet, new_content, 1)
        logger.debug("Replacing first occurrence of snippet.")
    else:
        logger.warning(f"Old content snippet not found in {file_path_str}. Config change not applied.")
        logger.debug(f"Snippet for comparison (first 100 chars): '{old_content_snippet[:100]}'")
        return False

    if _write_file_content(file_path_str, modified_content):
        logger.info(f"Applied text change to config {file_path_str}")
        return True
    else:
        logger.error(f"Failed to write changes to config {file_path_str}.")
        if backup_path_provided and os.path.exists(backup_path_provided):
            logger.info(f"Attempting rollback for {file_path_str} from {backup_path_provided}.")
            if not rollback_change(backup_path_provided, file_path_str):
                logger.critical(f"CRITICAL: Rollback failed for {file_path_str} after write error!")
        else:
            logger.warning(f"No backup path provided or backup does not exist at '{backup_path_provided}'. Cannot rollback {file_path_str}.")
        return False


def apply_script_modification(script_path_str, modification_details, ai_generated_code_block, backup_path_provided):
    """
    Applies a modification to a script file. More robust parsing/AST would be ideal.
    Returns: Boolean (success or failure).
    """
    mod_type = modification_details.get("type")
    logger.info(f"Attempting to apply script modification to: {script_path_str} with type: {mod_type}")
    
    current_script_content = system_analyzer.read_file_content(script_path_str)
    if current_script_content is None:
        logger.error(f"Cannot read script {script_path_str} for modification.")
        return False

    modified_content = None

    if mod_type == "replace_entire_script":
        modified_content = ai_generated_code_block
        logger.debug("Replacing entire script content.")
    elif mod_type == "append_to_script":
        modified_content = current_script_content + ("\n" if current_script_content else "") + ai_generated_code_block
        logger.debug("Appending code to script.")
    elif mod_type == "prepend_to_script":
        modified_content = ai_generated_code_block + ("\n" if current_script_content else "") + current_script_content
        logger.debug("Prepending code to script.")
    elif mod_type == "replace_bash_function" and modification_details.get("function_name"):
        import re
        func_name = modification_details["function_name"]
        # Improved regex to handle various Bash function definition styles more robustly
        # Matches: func_name() {body}, function func_name {body}, func_name (){body}, etc.
        # And accounts for functions possibly ending not at start of line for '}'
        pattern_str = r"(?:^|\s)(?:function\s+)?(" + re.escape(func_name) + r")\s*\(\s*\)\s*\{((?:[^{}]|\{[^{}]*\})*)\}"
        pattern = re.compile(pattern_str, re.MULTILINE | re.DOTALL)
        
        match = pattern.search(current_script_content)
        if match:
            # Replace the body of the function, keeping the declaration style somewhat similar if possible.
            # This is still regex based and not AST, so it has limitations.
            # The new code block should be the complete new function including declaration.
            modified_content = current_script_content[:match.start()] + ai_generated_code_block.strip() + current_script_content[match.end():]
            logger.debug(f"Attempted regex replacement of function '{func_name}'.")
        else:
            logger.warning(f"Could not find function '{func_name}' with regex for replacement in {script_path_str}.")
            return False
    else:
        logger.error(f"Unsupported script modification type or insufficient details: {mod_type}")
        return False

    if modified_content is not None:
        if _write_file_content(script_path_str, modified_content):
            logger.info(f"Successfully wrote script modification to {script_path_str}")
            language = modification_details.get("language", "bash").lower()
            syntax_check_passed = True
            if language == "bash":
                syntax_check_result = execute_command_or_script(f"bash -n {shlex.quote(script_path_str)}")
                if not syntax_check_result["success"] or syntax_check_result["exit_code"] != 0:
                    logger.error(f"Modified Bash script {script_path_str} has syntax errors! Output: '{syntax_check_result['output']}', Error: '{syntax_check_result['error']}'")
                    syntax_check_passed = False
                else:
                    logger.info(f"Bash script syntax check passed for {script_path_str}.")
            elif language == "python":
                # For Python, use py_compile to check syntax
                # python -m py_compile <filename> exits with 0 on success, non-zero on error
                syntax_check_result = execute_command_or_script(f"python -m py_compile {shlex.quote(script_path_str)}")
                if not syntax_check_result["success"] or syntax_check_result["exit_code"] != 0:
                    # py_compile prints errors to stderr
                    error_detail = syntax_check_result['error'] if syntax_check_result['error'] else syntax_check_result['output']
                    logger.error(f"Modified Python script {script_path_str} has syntax errors! Details: '{error_detail}'")
                    syntax_check_passed = False
                else:
                    logger.info(f"Python script syntax check passed for {script_path_str}.")
            # Add elif for other languages here if needed
            
            if not syntax_check_passed:
                logger.error(f"Rolling back script {script_path_str} due to syntax errors.")
                if backup_path_provided and os.path.exists(backup_path_provided):
                    if not rollback_change(backup_path_provided, script_path_str):
                         logger.critical(f"CRITICAL: Rollback FAILED for {script_path_str} after syntax error!")
                else:
                    logger.warning(f"No backup or backup does not exist at '{backup_path_provided}'. Cannot rollback syntax error in {script_path_str}.")
                return False
            return True
        else:
            logger.error(f"Failed to write modified script {script_path_str}")
            if backup_path_provided and os.path.exists(backup_path_provided):
                if not rollback_change(backup_path_provided, script_path_str):
                    logger.critical(f"CRITICAL: Rollback FAILED for {script_path_str} after write error!")
            else:
                logger.warning(f"No backup or backup does not exist at '{backup_path_provided}'. Cannot rollback write error for {script_path_str}.")
            return False
    else: # Should not happen if mod_type was valid and no error occurred during content processing
        logger.error(f"Script modification failed for {script_path_str}. Modified content was unexpectedly None.")
        return False


def execute_command_or_script(command_string, script_content=None, language=None, sandbox_level="HIGH"):
    """
    Executes a command or a script. Sandboxing is critical but placeholder.
    Returns: Dictionary {success (bool), output (str), error (str), exit_code (int)}
    """
    logger.warning(f"Executing (Sandbox Level: {sandbox_level} - CURRENTLY NOT ENFORCED): " +
                   (f"{language} script (content provided)" if script_content else command_string))
    
    if sandbox_level != "NONE":
        logger.critical("CRITICAL WARNING: True sandboxing is NOT IMPLEMENTED. Commands/scripts will run with the application's full privileges. Implementing effective sandboxing is complex, often requiring containerization (e.g., Docker), virtual machines, or advanced syscall filtering techniques to isolate processes. Without this, AI-generated or modified scripts could potentially perform unintended or harmful actions on the system. PROCEED WITH EXTREME CAUTION.")
        if sandbox_level == "HIGH" and script_content: # This check remains relevant
             logger.warning("High sandboxing specifically requested for AI-generated script content, but true isolation is not available. Proceeding with extreme caution and heightened awareness of potential risks.")

    exec_command_list = []
    temp_script_file = None

    if script_content and language:
        try:
            import tempfile
            suffix = f".{language}" if language else ".tmp"
            # Use NamedTemporaryFile correctly for this scenario
            fd, temp_script_file = tempfile.mkstemp(suffix=suffix, text=True)
            os.write(fd, script_content.encode('utf-8'))
            os.close(fd)
            
            interpreter_map = {"bash": "/bin/bash", "python": "/usr/bin/python3"} # Extendable
            interpreter = interpreter_map.get(language.lower())
            
            if not interpreter:
                logger.error(f"Unsupported script language for execution: {language}")
                if temp_script_file: os.remove(temp_script_file)
                return {"success": False, "output": "", "error": f"Unsupported language: {language}", "exit_code": -1}
            
            if language.lower() in ["bash", "sh"]: # Only make executable if it's a shell script
                os.chmod(temp_script_file, 0o755)

            exec_command_list = [interpreter, temp_script_file]
        except Exception as e:
            logger.error(f"Error preparing script for execution: {e}", exc_info=True)
            if temp_script_file and os.path.exists(temp_script_file): os.remove(temp_script_file)
            return {"success": False, "output": "", "error": str(e), "exit_code": -1}
    else:
        try:
            exec_command_list = shlex.split(command_string)
            if not exec_command_list: # Handle empty command string
                logger.error("Empty command string provided for execution.")
                return {"success": False, "output": "", "error": "Empty command string", "exit_code": -1}
        except Exception as e:
            logger.error(f"Error splitting command string '{command_string}': {e}")
            return {"success": False, "output": "", "error": f"Error splitting command: {e}", "exit_code": -1}


    try:
        logger.debug(f"Executing command list: {exec_command_list}")
        process = subprocess.run(exec_command_list, capture_output=True, text=True, check=False, timeout=60)
        
        success = process.returncode == 0
        output = process.stdout.strip() if process.stdout else ""
        error_output = process.stderr.strip() if process.stderr else ""

        log_output = output[:200] + ('...' if len(output) > 200 else '')
        log_error = error_output[:200] + ('...' if len(error_output) > 200 else '')

        if output: logger.debug(f"Command stdout: {log_output}")
        if error_output: logger.warning(f"Command stderr: {log_error}") # stderr isn't always an error
        logger.info(f"Execution result for '{' '.join(exec_command_list)}': Success={success}, ExitCode={process.returncode}")
        
        return {"success": success, "output": output, "error": error_output, "exit_code": process.returncode}
    
    except FileNotFoundError:
        cmd_not_found = exec_command_list[0] if exec_command_list else "Unknown"
        logger.error(f"Command not found: {cmd_not_found}")
        return {"success": False, "output": "", "error": f"Command not found: {cmd_not_found}", "exit_code": -127} # Common exit code for command not found
    except subprocess.TimeoutExpired:
        cmd_str = ' '.join(exec_command_list)
        logger.error(f"Command timed out after 60s: {cmd_str}")
        return {"success": False, "output": "", "error": f"Command execution timed out: {cmd_str}", "exit_code": -1} # Custom exit code for timeout
    except Exception as e:
        cmd_str = ' '.join(exec_command_list)
        logger.error(f"Error executing command '{cmd_str}': {e}", exc_info=True)
        return {"success": False, "output": "", "error": str(e), "exit_code": -1} # Generic error
    finally:
        if temp_script_file and os.path.exists(temp_script_file):
            try:
                os.remove(temp_script_file)
                logger.debug(f"Removed temporary script file: {temp_script_file}")
            except OSError as e: # Catch potential error during removal
                logger.error(f"Error removing temporary script file {temp_script_file}: {e}")


def rollback_change(backup_file_path_str, original_file_path_str):
    """
    Restores a file from its backup.
    Returns: Boolean (success or failure).
    """
    backup_file_path = pathlib.Path(backup_file_path_str)
    original_file_path = pathlib.Path(original_file_path_str)

    if not backup_file_path.exists():
        logger.error(f"Backup file not found, cannot rollback: {backup_file_path}")
        return False
    
    try:
        original_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Use copy2 to preserve metadata, and ensure original is overwritten
        shutil.copy2(str(backup_file_path), str(original_file_path))
        logger.info(f"Rolled back '{original_file_path}' from '{backup_file_path}'")
        return True
    except Exception as e:
        logger.error(f"Failed to restore '{original_file_path}' from backup '{backup_file_path}': {e}", exc_info=True)
        return False

def create_new_file(file_path_str, content, make_executable=False):
    """
    Creates a new file with the given content.
    Returns: True on success, False on failure.
    """
    logger.info(f"Attempting to create new file: {file_path_str}")
    file_path = pathlib.Path(file_path_str)

    try:
        if _write_file_content(str(file_path), content): # _write_file_content already handles parent dir creation
            logger.info(f"Successfully created new file: {file_path}")
            if make_executable:
                try:
                    os.chmod(file_path, 0o755) 
                    logger.info(f"Made file executable: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to make file executable {file_path}: {e}", exc_info=True)
                    # Depending on requirements, this might or might not be a fatal error for create_new_file
                    # For now, let's say if we want it executable, it must become executable.
                    # Consider removing the file if chmod fails and it's critical?
                    return False 
            return True
        else:
            # _write_file_content already logs the error
            logger.error(f"Failed to write content during creation of new file: {file_path}")
            return False
            
    except Exception as e: # Catch any other unexpected errors during path handling etc.
        logger.error(f"Error creating new file {file_path}: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    # Ensure logger_setup.setup_logger is called for the __main__ block.
    if not logger.hasHandlers() or isinstance(logger, logging.RootLogger):
        logger = logger_setup.setup_logger("EnhancementApplier_main", level=logging.DEBUG)
        logger.info("Logger re-initialized for __main__ block with DEBUG level.")
    else:
        logger.setLevel(logging.DEBUG)
        logger.info("Logger already initialized. Set level to DEBUG for __main__ block.")

    logger.info("--- EnhancementApplier Test ---")

    test_dir_name = "enhancement_applier_test_area"
    test_dir = config.PROJECT_ROOT / test_dir_name
    
    # Define backup base for test area to avoid polluting main backup area if path logic is tricky
    original_backup_base_path = config.BACKUP_BASE_PATH
    config.BACKUP_BASE_PATH = config.PROJECT_ROOT / "temp_backups_for_test" / test_dir_name
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir) 
    if os.path.exists(config.BACKUP_BASE_PATH): # Clean specific test backup area
        shutil.rmtree(config.BACKUP_BASE_PATH)

    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(config.BACKUP_BASE_PATH, exist_ok=True) # Ensure test backup root exists

    logger.info(f"Test area: {test_dir}")
    logger.info(f"Test backup area: {config.BACKUP_BASE_PATH}")


    dummy_config_name = "test_config.conf"
    dummy_script_name = "test_script.sh"
    dummy_config_path = test_dir / dummy_config_name
    dummy_script_path = test_dir / dummy_script_name

    config_content_v1 = "SettingA=1\nSettingB=2\n#Comment\nSettingC=3"
    script_content_v1 = "#!/bin/bash\n\necho \"Original Script Version 1\"\n\nmy_function() {\n    echo \"Inside my_function original\"\n}\nmy_function\n\nexit 0"

    _write_file_content(str(dummy_config_path), config_content_v1)
    _write_file_content(str(dummy_script_path), script_content_v1)
    os.chmod(dummy_script_path, 0o755)

    logger.info(f"Created dummy config: {dummy_config_path} with content:\n{config_content_v1}")
    logger.info(f"Created dummy script: {dummy_script_path} with content:\n{script_content_v1}")

    logger.info("\n--- Testing backup_file ---")
    config_backup_path = backup_file(str(dummy_config_path))
    assert config_backup_path and os.path.exists(config_backup_path), "Config backup failed or backup file not found."
    logger.info(f"Config backup created at: {config_backup_path}")
    
    script_backup_path = backup_file(str(dummy_script_path))
    assert script_backup_path and os.path.exists(script_backup_path), "Script backup failed or backup file not found."
    logger.info(f"Script backup created at: {script_backup_path}")


    logger.info("\n--- Testing apply_config_text_change ---")
    success_replace = apply_config_text_change(str(dummy_config_path), "SettingB=2", "SettingB=newValue", config_backup_path)
    assert success_replace and "SettingB=newValue" in system_analyzer.read_file_content(str(dummy_config_path)), "Config replace failed."
    logger.info(f"apply_config_text_change (replace SettingB): Success, content verified.")
    
    success_append = apply_config_text_change(str(dummy_config_path), "APPEND_MODE", "AppendedSetting=XYZ", config_backup_path)
    assert success_append and "AppendedSetting=XYZ" in system_analyzer.read_file_content(str(dummy_config_path)), "Config append failed."
    logger.info(f"apply_config_text_change (append): Success, content verified.")
    
    success_prepend = apply_config_text_change(str(dummy_config_path), "PREPEND_MODE", "PrependedSetting=ABC", config_backup_path)
    assert success_prepend and "PrependedSetting=ABC" in system_analyzer.read_file_content(str(dummy_config_path)), "Config prepend failed."
    logger.info(f"apply_config_text_change (prepend): Success, content verified.")
        
    success_overwrite = apply_config_text_change(str(dummy_config_path), "OVERWRITE_MODE", "ENTIRELY NEW CONTENT", config_backup_path)
    assert success_overwrite and system_analyzer.read_file_content(str(dummy_config_path)) == "ENTIRELY NEW CONTENT", "Config overwrite failed."
    logger.info(f"apply_config_text_change (overwrite): Success, content verified.")

    logger.info("\n--- Testing rollback_change (config) ---")
    success_rollback = rollback_change(config_backup_path, str(dummy_config_path))
    assert success_rollback and system_analyzer.read_file_content(str(dummy_config_path)) == config_content_v1, "Config rollback failed."
    logger.info(f"rollback_change (config): Success, content verified to original.")


    logger.info("\n--- Testing apply_script_modification ---")
    mod_details_replace_all = {"type": "replace_entire_script", "language": "bash"}
    new_script_content = "#!/bin/bash\necho 'Entirely new script content'\nexit 1" # exit 1 to check exit code later
    success_script_mod_all = apply_script_modification(str(dummy_script_path), mod_details_replace_all, new_script_content, script_backup_path)
    assert success_script_mod_all and system_analyzer.read_file_content(str(dummy_script_path)) == new_script_content, "Script replace_entire_script failed."
    logger.info(f"apply_script_modification (replace_entire_script): Success, content verified.")
    
    assert rollback_change(script_backup_path, str(dummy_script_path)), "Failed to rollback script for function test."
    logger.info(f"Rolled back script to V1 for function replacement test. Content:\n{system_analyzer.read_file_content(str(dummy_script_path))}")
    
    mod_details_replace_func = {"type": "replace_bash_function", "function_name": "my_function", "language": "bash"}
    new_func_code = "my_function() {\n    echo \"Inside my_function MODIFIED\"\n    echo \"Another line\"\n}"
    success_script_mod_func = apply_script_modification(str(dummy_script_path), mod_details_replace_func, new_func_code, script_backup_path)
    assert success_script_mod_func, "Replace bash function failed."
    content_after_func_replace = system_analyzer.read_file_content(str(dummy_script_path))
    assert "Inside my_function MODIFIED" in content_after_func_replace and "Inside my_function original" not in content_after_func_replace, "Function replacement content incorrect."
    logger.info(f"apply_script_modification (replace_bash_function 'my_function'): Success, content verified.")

    logger.info("\n--- Testing script modification with syntax error & rollback ---")
    assert rollback_change(script_backup_path, str(dummy_script_path)), "Failed to rollback script for bash syntax error test."
    logger.info(f"Rolled back bash script to V1 for syntax error test.")
        
    mod_details_bash_syntax_error = {"type": "append_to_script", "language": "bash"}
    bash_code_with_syntax_error = "\necho 'Valid line before error'\nif then else fi # obvious bash syntax error"
    success_bash_syntax_error = apply_script_modification(str(dummy_script_path), mod_details_bash_syntax_error, bash_code_with_syntax_error, script_backup_path)
    assert not success_bash_syntax_error, "Bash script modification with syntax error should have failed."
    content_after_failed_bash_mod = system_analyzer.read_file_content(str(dummy_script_path))
    assert content_after_failed_bash_mod == script_content_v1, "Bash script content not rolled back after syntax error."
    logger.info("apply_script_modification (bash with syntax error): Failed as expected, and content rolled back to original.")

    # --- Test Python Script Modification & Syntax Check ---
    logger.info("\n--- Testing Python script modification with syntax error & rollback ---")
    dummy_python_script_name = "test_python_script.py"
    dummy_python_script_path = test_dir / dummy_python_script_name
    python_script_v1_content = "#!/usr/bin/env python3\n\nprint('Original Python script version 1')\n\ndef main():\n    print('Python main original')\n\nif __name__ == '__main__':\n    main()\n"
    _write_file_content(str(dummy_python_script_path), python_script_v1_content)
    os.chmod(dummy_python_script_path, 0o755)
    logger.info(f"Created dummy Python script: {dummy_python_script_path}")

    python_script_backup_path = backup_file(str(dummy_python_script_path))
    assert python_script_backup_path, "Backup for Python script failed."

    mod_details_python_syntax_error = {"type": "append_to_script", "language": "python"}
    python_code_with_syntax_error = "\nprint('Valid line')\ndef error_func(\n    print('Oops, indent error') # Syntax error here"
    
    success_python_syntax_error = apply_script_modification(str(dummy_python_script_path), mod_details_python_syntax_error, python_code_with_syntax_error, python_script_backup_path)
    assert not success_python_syntax_error, "Python script modification with syntax error should have failed."
    content_after_failed_python_mod = system_analyzer.read_file_content(str(dummy_python_script_path))
    assert content_after_failed_python_mod == python_script_v1_content, "Python script content not rolled back after syntax error."
    logger.info("apply_script_modification (python with syntax error): Failed as expected, and content rolled back.")

    logger.info("\n--- Testing valid Python script modification ---")
    mod_details_python_valid = {"type": "replace_entire_script", "language": "python"}
    valid_python_content_v2 = "#!/usr/bin/env python3\nprint('Valid Python script version 2')\n"
    success_python_valid_mod = apply_script_modification(str(dummy_python_script_path), mod_details_python_valid, valid_python_content_v2, python_script_backup_path)
    assert success_python_valid_mod, "Valid Python script modification failed."
    assert system_analyzer.read_file_content(str(dummy_python_script_path)) == valid_python_content_v2, "Valid Python script content not updated correctly."
    logger.info("apply_script_modification (valid python): Success, content updated.")
    # --- End of Python Script Test ---

    logger.info("\n--- Testing execute_command_or_script ---")
    exec_result_ls = execute_command_or_script(f"ls -a {shlex.quote(str(test_dir))}") # Use -a to see . and ..
    assert exec_result_ls["success"] and dummy_config_name in exec_result_ls["output"], f"ls command failed or output incorrect: {exec_result_ls}"
    logger.info(f"execute_command_or_script (ls -a): Success, output verified.")
    
    exec_result_script_file = execute_command_or_script(f"/bin/bash {shlex.quote(str(dummy_script_path))}") # Current content is V1
    assert exec_result_script_file["success"] and "Original Script Version 1" in exec_result_script_file["output"] and "Inside my_function original" in exec_result_script_file["output"], f"Execution of script V1 failed or output incorrect: {exec_result_script_file}"
    logger.info(f"execute_command_or_script (dummy script file V1): Success, output verified.")

    bash_script_content_direct = "#!/bin/bash\necho 'Hello from direct bash execution'\nexit 0"
    exec_result_bash_direct = execute_command_or_script(command_string=None, script_content=bash_script_content_direct, language="bash")
    assert exec_result_bash_direct["success"] and "Hello from direct bash execution" in exec_result_bash_direct["output"], f"Direct bash script execution failed: {exec_result_bash_direct}"
    logger.info(f"execute_command_or_script (direct bash): Success, output verified.")

    python_script_content_direct = "import sys\nprint('Hello from direct python execution')\nsys.exit(0)"
    exec_result_python_direct = execute_command_or_script(command_string=None, script_content=python_script_content_direct, language="python")
    assert exec_result_python_direct["success"] and "Hello from direct python execution" in exec_result_python_direct["output"], f"Direct python script execution failed: {exec_result_python_direct}"
    logger.info(f"execute_command_or_script (direct python): Success, output verified.")
    
    exec_result_fail = execute_command_or_script("command_does_not_exist_sfsdfsdf")
    assert not exec_result_fail["success"] and exec_result_fail["exit_code"] == -127, f"Non-existent command test failed: {exec_result_fail}"
    logger.info(f"execute_command_or_script (non-existent command): Failed as expected.")


    logger.info("\n--- Testing create_new_file ---")
    new_file_path = test_dir / "brand_new_file.txt"
    new_file_content = "This is a brand new file.\nWith multiple lines of text."
    assert create_new_file(str(new_file_path), new_file_content), "create_new_file failed for text file."
    assert os.path.exists(new_file_path) and system_analyzer.read_file_content(str(new_file_path)) == new_file_content, "New text file content incorrect."
    logger.info(f"create_new_file (text file): Success, content verified.")

    new_script_exe_path = test_dir / "brand_new_script.sh"
    new_script_exe_content = "#!/bin/bash\necho 'Output from newly created executable script'\nexit 0"
    assert create_new_file(str(new_script_exe_path), new_script_exe_content, make_executable=True), "create_new_file failed for executable script."
    assert os.path.exists(new_script_exe_path) and os.access(new_script_exe_path, os.X_OK), "New script not executable or not found."
    logger.info(f"create_new_file (executable script): Success, executable permission verified.")
    
    exec_result_new_script = execute_command_or_script(f"/bin/bash {shlex.quote(str(new_script_exe_path))}")
    assert exec_result_new_script["success"] and "Output from newly created executable script" in exec_result_new_script["output"], f"Execution of newly created script failed: {exec_result_new_script}"
    logger.info(f"execute_command_or_script (newly created script): Success, output verified.")

    # Restore original backup path
    config.BACKUP_BASE_PATH = original_backup_base_path
    logger.info(f"\nTest files and backups are in: {test_dir} and {config.PROJECT_ROOT / 'temp_backups_for_test' / test_dir_name}")
    logger.info("Consider cleaning up test directories manually if needed.")
    logger.info("--- End of EnhancementApplier Test ---")

