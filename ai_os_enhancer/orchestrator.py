import logging
import json
import os
import difflib
import time # For time.sleep in the run loop

# Relative imports
try:
    from . import config
    from . import logger_setup
    from . import system_analyzer
    from . import ollama_interface
    from . import enhancement_applier
except ImportError:
    import sys
    # Add project root to path to allow direct execution and importing ai_os_enhancer modules
    # This assumes orchestrator.py is in ai_os_enhancer directory, so parent is project root
    sys.path.insert(0, str(system_analyzer.pathlib.Path(__file__).resolve().parent.parent))
    from ai_os_enhancer import config
    from ai_os_enhancer import logger_setup
    from ai_os_enhancer import system_analyzer
    from ai_os_enhancer import ollama_interface
    from ai_os_enhancer import enhancement_applier

# Initialize logger for this module
if __name__ == '__main__':
    # This block specifically configures logging when orchestrator.py is run directly.
    # It ensures that even without a higher-level entry point calling logger_setup first,
    # logs from this direct execution will be captured and formatted.
    # The logger obtained here will be the 'root' logger for this specific execution context if not named.
    # Giving it a specific name for clarity if needed, or configuring the root logger.
    if not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # This logger instance is primarily for the __main__ block itself.
        # The Orchestrator class will get its own instance via logger_setup.setup_logger.
        logger = logging.getLogger("Orchestrator_DirectMainSetup")
    else:
        # If handlers are already present, assume logger_setup has been run by an import.
        # Get a logger for the main block, possibly configuring its level.
        logger = logger_setup.setup_logger("Orchestrator_MainBlock", log_level=logging.DEBUG)
else:
    # When imported as a module, Orchestrator class will get its logger via __init__.
    # This module-level logger can also be set up if needed for module-level functions.
    logger = logger_setup.setup_logger("Orchestrator_Module")


class Orchestrator:
    def __init__(self):
        self.logger = logger_setup.setup_logger("Orchestrator_Instance", log_level=logging.DEBUG) # Ensure instance logger is also DEBUG for tests
        self.logger.info("Orchestrator instance created.")
        self.logger.info("Shell Assistant capability is available via ollama_interface.generate_shell_command. Orchestrator will use this for 'shell_task' type enhancements.")
        self.initialize_system()

    def _log_user_alert(self, message, level="INFO"):
        log_message = f"[USER_ALERT] {message}"
        level_upper = level.upper()
        if level_upper == "CRITICAL": self.logger.critical(log_message)
        elif level_upper == "WARNING": self.logger.warning(log_message)
        elif level_upper == "ERROR": self.logger.error(log_message)
        else: self.logger.info(log_message)
        print(log_message)

    def initialize_system(self):
        self.logger.info("Orchestrator initializing system...")
        self.project_root = config.PROJECT_ROOT
        self.db_path = config.CONFIG_DATABASE_PATH
        self.log_file = config.LOG_FILE_PATH
        self.backup_path = config.BACKUP_BASE_PATH
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.log_file.parent, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
        self.logger.info(f"Project Root: {self.project_root}")
        self.system_stability_score = 100.0
        self.human_intervention_required = False
        self.enhancement_history = []
        self._load_persistent_state()
        self.logger.info("System initialization complete.")

    def _load_persistent_state(self): self.logger.debug("Loading persistent state (placeholder).")
    def _save_persistent_state(self): self.logger.debug("Saving persistent state (placeholder).")

    def _is_approval_required(self, enhancement_proposal: dict) -> bool:
        risk_assessment_data = enhancement_proposal.get("risk_assessment", "HIGH")
        risk = "HIGH" # Default
        if isinstance(risk_assessment_data, dict):
            risk = risk_assessment_data.get("risk_level", "HIGH").upper()
        elif isinstance(risk_assessment_data, str):
            risk = risk_assessment_data.upper()

        impact = enhancement_proposal.get("impact_level", "SIGNIFICANT").upper()
        approval_threshold = str(config.HUMAN_APPROVAL_THRESHOLD).upper()
        self.logger.debug(f"Checking approval: Risk='{risk}', Impact='{impact}', Threshold='{approval_threshold}'")

        if approval_threshold == "LOW": return not (risk == "LOW" and impact == "MINIMAL")
        elif approval_threshold == "MEDIUM": return risk in ["MEDIUM", "HIGH"] or impact == "SIGNIFICANT"
        elif approval_threshold == "HIGH": return risk == "HIGH" or (risk == "MEDIUM" and impact == "SIGNIFICANT")
        self.logger.warning(f"Unknown HUMAN_APPROVAL_THRESHOLD: {approval_threshold}. Requiring approval.")
        return True

    def _display_shell_task_details(self, shell_details: dict) -> None:
        self._log_user_alert("--- Proposed Shell Task Details ---", "INFO")
        if shell_details.get("error"):
            self._log_user_alert(f"Error in shell task details: {shell_details['error']}", "ERROR")
            if shell_details.get("details"): self._log_user_alert(f"Further Details: {shell_details['details']}", "ERROR")
            return

        self._log_user_alert(f"  Task Description: {shell_details.get('task_description', 'N/A')}", "INFO")
        # Use print for potentially multi-line commands for better readability in console
        print(f"  [USER_ALERT] Generated Command:\n    {shell_details.get('generated_command', 'N/A')}")

        risk_info = shell_details.get("risk_assessment")
        if isinstance(risk_info, dict):
            self._log_user_alert("  Risk Assessment:", "INFO")
            self._log_user_alert(f"    Level: {risk_info.get('risk_level', 'N/A')}", "INFO")
            self._log_user_alert(f"    Operation Type: {risk_info.get('operation_type', 'N/A')}", "INFO")
            self._log_user_alert(f"    Requires Privileges: {risk_info.get('requires_privileges', 'N/A')}", "INFO")
        else: self._log_user_alert(f"  Risk Assessment: {risk_info if risk_info else 'N/A (Format Error)'}", "INFO")

        self._log_user_alert(f"  Prerequisites: {'; '.join(shell_details.get('prerequisites', [])) if shell_details.get('prerequisites') else 'None'}", "INFO")
        if shell_details.get("setup_commands"): self._log_user_alert(f"  Setup Commands: {shell_details.get('setup_commands')}", "INFO")
        self._log_user_alert("  Safety Notes:", "INFO")
        for note in shell_details.get("safety_notes", ["None"]): self._log_user_alert(f"    - {note}", "INFO")
        if shell_details.get("clarifications_needed"):
            self._log_user_alert("  Clarifications Needed by LLM:", "WARNING")
            for clar in shell_details.get("clarifications_needed"): self._log_user_alert(f"    - {clar}", "WARNING")
        self._log_user_alert(f"  Alternatives: {'; '.join(shell_details.get('alternatives', [])) if shell_details.get('alternatives') else 'None'}", "INFO")
        self._log_user_alert("---------------------------------", "INFO")


    def _display_proposal_diff(self, enhancement_proposal: dict) -> None:
        item_path = enhancement_proposal.get("item_path")
        item_type = enhancement_proposal.get("item_type")
        proposed_details = enhancement_proposal.get("proposed_change_details", {})
        change_type = str(proposed_details.get("type", enhancement_proposal.get("change_type", ""))).lower()
        is_new_file = "create_new" in change_type or "new_script_creation" in change_type

        if is_new_file:
            self._log_user_alert("\nThis proposal is for creating a NEW FILE.", "INFO")
            new_content_desc = proposed_details.get("task_description", enhancement_proposal.get("enhancement_description", "Content to be generated."))
            self._log_user_alert(f"Content will be based on: {new_content_desc}", "INFO")
            if proposed_details.get("language", enhancement.get("language")): # check both places
                self._log_user_alert(f"Language: {proposed_details.get('language', enhancement.get('language'))}", "INFO")
            pre_defined_content = proposed_details.get("new_code_snippet") or proposed_details.get("new_content") or enhancement_proposal.get("proposed_change_snippet")
            if pre_defined_content:
                self._log_user_alert("\n--- Proposed New File Content ---", "INFO"); print(pre_defined_content); self._log_user_alert("-----------------------------", "INFO")
            return

        if not item_path or item_path == "N/A":
            self._log_user_alert("\nCannot generate diff: Item path is 'N/A' or missing for file-based change.", "WARNING"); return

        current_content = system_analyzer.read_file_content(item_path)
        if current_content is None:
            self._log_user_alert(f"\nCould not read current content of {item_path} to generate diff.", "WARNING"); return

        proposed_final_content = enhancement_proposal.get("proposed_change_snippet") # Default to this if available

        # This logic might need refinement based on how proposed_change_details is structured by LLM for non-generated content
        if not proposed_final_content and proposed_details:
            pre_defined_content = proposed_details.get("new_code_snippet") or proposed_details.get("new_content") or proposed_details.get("new_line_content") or proposed_details.get("block_content")
            if pre_defined_content is not None:
                if item_type == "config":
                    target_marker = proposed_details.get("target_marker_or_snippet") or proposed_details.get("target_pattern") or proposed_details.get("target_line_pattern")
                    if "append" in change_type: proposed_final_content = current_content + ("\n" if current_content.strip() else "") + pre_defined_content
                    elif "prepend" in change_type: proposed_final_content = pre_defined_content + ("\n" if current_content.strip() else "") + current_content
                    elif "overwrite" in change_type or "replace_entire" in change_type: proposed_final_content = pre_defined_content
                    elif target_marker and target_marker in current_content: proposed_final_content = current_content.replace(target_marker, pre_defined_content, 1)
                    else: proposed_final_content = pre_defined_content # Fallback for diff view
                elif item_type == "script": proposed_final_content = pre_defined_content # Simplified for script diff view

        if proposed_final_content is not None:
            self._log_user_alert("\n--- Proposed Change Diff ---", "INFO")
            diff_lines = list(difflib.unified_diff(current_content.splitlines(keepends=True), proposed_final_content.splitlines(keepends=True), fromfile=f"a/{item_path}", tofile=f"b/{item_path}", lineterm=""))
            if diff_lines: [print(line, end='') for line in diff_lines]; print()
            else: self._log_user_alert("No textual changes proposed (or content is identical).", "INFO")
            self._log_user_alert("--------------------------", "INFO")
        else: self._log_user_alert("\nCould not determine or generate a diff preview for the file-based change.", "WARNING")

    def _display_proposal_details(self, enhancement_proposal: dict) -> None:
        self._log_user_alert("--- Enhancement Proposal (Summary from Strategist) ---", "INFO")
        try:
            temp_proposal_for_print = {k: v for k, v in enhancement_proposal.items() if k != "generated_shell_command_details"}
            print(json.dumps(temp_proposal_for_print, indent=2, sort_keys=True))
        except Exception as e: print(f"Error formatting JSON details: {e}\nRaw: {enhancement_proposal}")
        self._log_user_alert("----------------------------------------------------", "INFO")

        if enhancement_proposal.get("change_type") == "shell_task" and "generated_shell_command_details" in enhancement_proposal:
            self._display_shell_task_details(enhancement_proposal["generated_shell_command_details"])
        elif enhancement_proposal.get("change_type") != "shell_task":
            self._display_proposal_diff(enhancement_proposal)

    def _get_user_approval_input(self, proposal_summary: str, enhancement_proposal: dict) -> bool:
        item_path = enhancement_proposal.get("item_path", "Unknown item")
        self._log_user_alert(f"\n{proposal_summary}", "WARNING")
        try:
            while True:
                print("Approve this change? (yes/no/details): ", end=''); response = input().strip().lower()
                if response == "yes": self.logger.info(f"Enhancement APPROVED by human: {item_path}"); return True
                elif response == "no": self.logger.info(f"Enhancement REJECTED by human: {item_path}"); return False
                elif response == "details": self._display_proposal_details(enhancement_proposal); self._log_user_alert(f"\n{proposal_summary}", "WARNING")
                else: self._log_user_alert("Invalid input. Please enter 'yes', 'no', or 'details'.", "WARNING")
        except KeyboardInterrupt: self.logger.warning(f"Approval for '{item_path}' CANCELED by user (Ctrl+C)."); self._log_user_alert(f"Approval CANCELED.", "WARNING"); return False
        except EOFError: self.logger.warning(f"EOFError for '{item_path}'. Assuming rejection."); self._log_user_alert(f"Approval CANCELED (EOF).", "WARNING"); return False
        return False

    def request_human_approval_if_needed(self, enhancement_proposal: dict) -> bool:
        item_path = enhancement_proposal.get("item_path", "Unknown item")
        risk_data = enhancement_proposal.get("risk_assessment", "HIGH") # Default to high if missing
        effective_risk = "HIGH"
        if isinstance(risk_data, dict): effective_risk = risk_data.get("risk_level", "HIGH").upper()
        elif isinstance(risk_data, str): effective_risk = risk_data.upper()

        temp_proposal_for_check = enhancement_proposal.copy()
        temp_proposal_for_check["risk_assessment"] = effective_risk

        if not self._is_approval_required(temp_proposal_for_check):
            impact_level_log = enhancement_proposal.get("impact_level", "N/A").upper()
            if isinstance(risk_data, dict): impact_level_log = risk_data.get("operation_type", "N/A")
            self.logger.info(f"Enhancement auto-approved (Risk: {effective_risk}, Impact-proxy: {impact_level_log}) for '{item_path}'.")
            return True

        justification = enhancement_proposal.get("justification", "No justification provided.")
        desc = enhancement_proposal.get('enhancement_description', item_path)
        proposal_summary = f"Approval for '{desc}'. Assessed Risk: {effective_risk}. Justification: {justification}"
        if enhancement_proposal.get("change_type") == "shell_task" and isinstance(risk_data, dict): # More specific summary for shell tasks
            proposal_summary = f"Approval for SHELL TASK '{desc}'. Assessed Risk: {effective_risk} (Type: {risk_data.get('operation_type', 'N/A')}). Justification: {justification}"
        return self._get_user_approval_input(proposal_summary, enhancement_proposal)

    def monitor_system_health(self):
        self.logger.debug("Monitoring system health...")
        critical_services = getattr(config, 'CRITICAL_SERVICES_MONITOR', ["cron", "ssh"])
        penalty = sum(10 for s_name in critical_services if not system_analyzer.is_service_active(s_name))
        if penalty > 0:
            self.system_stability_score = max(0, self.system_stability_score - penalty)
            self.logger.warning(f"System stability score reduced to {self.system_stability_score:.2f} due to service issues.")
        elif self.system_stability_score < 100:
            self.system_stability_score = min(100, self.system_stability_score + 1)
        if self.system_stability_score < 50:
            self._log_user_alert(f"System stability critically low ({self.system_stability_score:.2f})!", "CRITICAL"); self.human_intervention_required = True
        self.logger.info(f"Current system stability score: {self.system_stability_score:.2f}")
        return self.system_stability_score

    def _format_item_for_analysis(self, item_path_str, item_type):
        content = system_analyzer.read_file_content(item_path_str)
        if content is not None: return {"path": item_path_str, "content": content, "type": item_type}
        self.logger.warning(f"Could not read {item_path_str}, skipping analysis."); return None

    def main_enhancement_cycle(self):
        self.logger.info(f"--- Starting Enhancement Cycle. Stability: {self.system_stability_score:.2f} ---")
        if self.human_intervention_required: self._log_user_alert("Human intervention required. System paused.", "CRITICAL"); return False

        specs = config.MONITORED_SCRIPTS_PATHS
        if not (isinstance(specs, list) and all(isinstance(i, dict) for i in specs)):
            if isinstance(specs, list) and all(isinstance(p, str) for p in specs): specs = [{"path": p, "type": "script"} for p in specs]
            else: specs = []
        if not specs: self.logger.info("MONITORED_SCRIPTS_PATHS is effectively empty.")

        tasks = [t for spec in specs if (p:=spec.get("path")) and system_analyzer.pathlib.Path(p).exists() and (t:=self._format_item_for_analysis(p, spec.get("type","script")))]
        if not tasks: self.logger.info("No existing items found/readable to analyze."); return True

        results = []
        max_items = config.MAX_CONCURRENT_ANALYSES
        self.logger.info(f"Analyzing up to {max_items} items.")
        for i, task_item in enumerate(tasks[:max_items]):
            self.logger.info(f"Analyzing item {i+1}: ({task_item['type']}) {task_item['path']}")
            analysis = ollama_interface.analyze_system_item(item_content=task_item['content'], item_path=task_item['path'], item_type=task_item['type'], role_name="GenericSystemItemAnalyzer")
            if analysis and not analysis.get("error"): analysis["item_path"], analysis["item_type"] = task_item['path'], task_item['type']; results.append(analysis)
            else: self.logger.error(f"Failed to analyze {task_item['path']}. Error: {analysis.get('error', 'Unknown') if analysis else 'No response'}"); self.system_stability_score -=1

        if not results: self.logger.info("No successful analysis results."); return True

        self.logger.info("Conceiving enhancement strategy...")
        snapshot = system_analyzer.get_system_snapshot()
        strategy = ollama_interface.conceive_enhancement_strategy(snapshot, results, role_name="EnhancementStrategist")
        if not (strategy and not strategy.get("error")):
            self.logger.error(f"Failed to conceive strategy. Error: {strategy.get('error', 'Unknown') if strategy else 'No response'}"); self.system_stability_score -=5; return True

        enhancements = strategy.get("prioritized_enhancements", [])
        if not enhancements: self.logger.info("No enhancements proposed."); return True

        for enh in enhancements:
            if self.human_intervention_required or self.system_stability_score < config.MIN_STABILITY_FOR_ENHANCEMENT:
                self._log_user_alert("Halting enhancements due to instability or prior issue.", "WARNING"); break

            c_type = str(enh.get("change_type", "")).lower()
            i_path = enh.get("item_path", "N/A")
            self.logger.info(f"Considering enhancement for '{i_path}' (Type: {c_type}): {enh.get('enhancement_description', 'N/A')}")

            if c_type == "shell_task":
                nl_task = enh.get("natural_language_task")
                if nl_task:
                    self.logger.info(f"Processing shell_task: '{nl_task}'")
                    gen_details = ollama_interface.generate_shell_command(nl_task)
                    if gen_details and not gen_details.get("error"):
                        enh["generated_shell_command_details"] = gen_details
                        if "risk_assessment" in gen_details: enh["risk_assessment"] = gen_details["risk_assessment"]
                        self.logger.info(f"Successfully generated shell command details for: '{nl_task}'")
                    else:
                        self.logger.error(f"Failed to generate shell command for task: '{nl_task}'. Details: {gen_details}")
                        enh["generated_shell_command_details"] = {"error": "Failed to generate command", "details": gen_details or "No response"}
                else:
                    self.logger.error("Shell task missing 'natural_language_task'."); enh["generated_shell_command_details"] = {"error": "Missing natural_language_task"}

            if not self.request_human_approval_if_needed(enh): self.logger.info(f"Enhancement skipped (human disapproval): {i_path}"); continue

            apply_ok = False
            if c_type == "shell_task":
                cmd_details = enh.get("generated_shell_command_details")
                if cmd_details and not cmd_details.get("error") and (cmd_to_run := cmd_details.get("generated_command")):
                    self.logger.info(f"Executing shell command for: {cmd_details.get('task_description')}. Command: {cmd_to_run}")
                    self.logger.warning("Safety Notes for execution:"); [self.logger.warning(f"  - {n}") for n in cmd_details.get("safety_notes",[])]
                    exec_res = enhancement_applier.execute_command_or_script(command_string=cmd_to_run, sandbox_level="HIGH") # Consider config for sandbox level
                    if exec_res["success"]: self.logger.info(f"Shell task OK. Output: {exec_res['output']}"); self.system_stability_score = min(100, self.system_stability_score + 2); apply_ok = True
                    else: self.logger.error(f"Shell task FAILED. Code: {exec_res['exit_code']}. Err: {exec_res['error']}. Out: {exec_res['output']}"); self.system_stability_score -= 20; apply_ok = False
                else: self.logger.error(f"Cannot execute shell_task for '{i_path}': command generation failed or empty. Details: {cmd_details}"); apply_ok = False
            else: # File-based modification
                is_new = "create_new" in c_type or "new_script_creation" in c_type
                bk_path = None
                if not is_new:
                    if not system_analyzer.pathlib.Path(i_path).exists(): self.logger.error(f"Target '{i_path}' not found. Skipping."); continue
                    if not (bk_path := enhancement_applier.backup_file(i_path)): self.logger.error(f"Backup failed for '{i_path}'. Skipping."); continue

                content = enh.get("proposed_change_snippet")
                if not content and enh.get("requires_code_generation", False):
                    self.logger.info(f"Generating content for '{i_path}'. Task: {enh.get('enhancement_description')}")
                    content = ollama_interface.generate_code_or_modification(enh.get('enhancement_description'), enh.get("language", "text"), enh.get("current_relevant_content_snippet"))
                    if not content: self.logger.error(f"Content generation failed for: {i_path}"); continue
                    enh["proposed_change_snippet"] = content

                if is_new: apply_ok = enhancement_applier.create_new_file(i_path, content, enh.get("item_type") == "script") if content else False
                elif enh.get("item_type") == "script": apply_ok = enhancement_applier.apply_script_modification(i_path, enh, content, bk_path) if content else False
                elif enh.get("item_type") == "config": apply_ok = enhancement_applier.apply_config_text_change(i_path, enh.get("target_criteria"), content, bk_path) if content and enh.get("target_criteria") else False

            if apply_ok: self.logger.info(f"Enhancement APPLIED for: {i_path}"); self.system_stability_score += 5
            else: self.logger.error(f"Failed to apply enhancement for: {i_path}"); self.system_stability_score -= 15
            self.enhancement_history.append({"enhancement": enh, "status": "success" if apply_ok else "failure", "backup": bk_path if not (c_type == "shell_task" or is_new) else None})

            self.monitor_system_health()
            if self.human_intervention_required: self._log_user_alert(f"System instability after change for {i_path}. Intervention needed.", "CRITICAL"); break

        self._save_persistent_state()
        self.logger.info(f"--- Enhancement cycle completed. Stability: {self.system_stability_score:.2f} ---")
        return True

    def run(self):
        self.logger.info(f"AI OS Enhancer starting up. PID: {os.getpid()}")
        self.logger.info(f"Human Approval Threshold: {config.HUMAN_APPROVAL_THRESHOLD}")
        self.logger.info(f"Monitored Paths: {config.MONITORED_SCRIPTS_PATHS}")
        self.logger.info(f"Ollama Endpoint: {config.OLLAMA_API_ENDPOINT}, Default Model: {config.DEFAULT_MODEL}")
        if not system_analyzer.check_ollama_service_availability(config.OLLAMA_API_ENDPOINT):
            self._log_user_alert(f"Ollama service not available at {config.OLLAMA_API_ENDPOINT}. Exiting.", "CRITICAL"); return
        try:
            count = 0
            test_run = os.getenv("AIOS_TEST_QUICK_CYCLE", "false").lower() == "true"
            while True:
                count += 1; self.logger.info(f"--- Orchestrator Cycle {count} ---")
                if not self.main_enhancement_cycle(): break
                if self.human_intervention_required: self.logger.critical("Human intervention required. Pausing."); break
                if test_run: self.logger.info("AIOS_TEST_QUICK_CYCLE=true. Exiting after one cycle."); break
                self.logger.info(f"Waiting for {config.CYCLE_INTERVAL_SECONDS}s..."); time.sleep(config.CYCLE_INTERVAL_SECONDS)
        except KeyboardInterrupt: self.logger.info("Orchestrator run interrupted by user.")
        except Exception as e: self.logger.critical(f"Uncaught exception in run loop: {e}", exc_info=True)
        finally: self._save_persistent_state(); self.logger.info("AI OS Enhancer shutting down.")

if __name__ == '__main__':
    main_logger = logger_setup.setup_logger("Orchestrator_DirectRun", log_level=logging.DEBUG)
    main_logger.info("Starting Orchestrator directly...")
    main_logger.info(f"Config - Min Stability for Enhancement: {config.MIN_STABILITY_FOR_ENHANCEMENT}")
    main_logger.info(f"Config - Cycle Interval Seconds: {config.CYCLE_INTERVAL_SECONDS}")

    orchestrator = Orchestrator()

    if os.getenv("AIOS_TEST_SHELL_TASK_WORKFLOW") == "true":
        main_logger.info("--- Initiating Test: Shell Task Workflow ---")
        test_shell_task_proposal = {
            "item_path": "N/A",
            "item_type": "system_task",
            "enhancement_description": "Test: List files in /etc like an expert",
            "justification": "Testing the shell_task direct workflow via __main__.",
            "estimated_impact": "Low",
            "change_type": "shell_task",
            "natural_language_task": "List all files and directories in /etc using 'ls -la', including their sizes in human-readable format.",
            "verification_steps": "Observe output for file listing.",
            "rollback_steps": "N/A for read-only ls command."
        }
        main_logger.info(f"Test natural_language_task: {test_shell_task_proposal['natural_language_task']}")

        # Simulate part of main_enhancement_cycle for this specific task
        generated_details = ollama_interface.generate_shell_command(test_shell_task_proposal['natural_language_task'])

        if generated_details and not generated_details.get("error"):
            test_shell_task_proposal["generated_shell_command_details"] = generated_details
            if "risk_assessment" in generated_details: # Update top-level risk for approval check
                test_shell_task_proposal["risk_assessment"] = generated_details["risk_assessment"]
            main_logger.info("Successfully generated shell command details for test task.")
        else:
            main_logger.error(f"Failed to generate shell command for test task. Details: {generated_details}")
            test_shell_task_proposal["generated_shell_command_details"] = {"error": "Failed to generate command during test", "details": generated_details}

        # Request approval (this will print details)
        approved = orchestrator.request_human_approval_if_needed(test_shell_task_proposal)

        if approved:
            main_logger.info("Test shell_task approved by user.")
            cmd_details_to_exec = test_shell_task_proposal.get("generated_shell_command_details", {})
            cmd_to_run = cmd_details_to_exec.get("generated_command")
            if cmd_to_run:
                main_logger.info(f"Executing command for test: {cmd_to_run}")
                execution_result = enhancement_applier.execute_command_or_script(
                    command_string=cmd_to_run,
                    sandbox_level=os.getenv("AIOS_TEST_SANDBOX_LEVEL", "DOCKER") # Allow test override
                )
                main_logger.info(f"Test execution result: {json.dumps(execution_result, indent=2)}")
            else:
                main_logger.error("No command was generated for the test task, cannot execute.")
        else:
            main_logger.info("Test shell_task was not approved by user.")
        main_logger.info("--- Shell Task Workflow Test Finished ---")
    else:
        orchestrator.run()

    main_logger.info("Orchestrator direct run finished or test completed.")
