# ai_os_enhancer/orchestrator.py

import time
import logging
import json # For formatting text data if needed by OllamaInterface's helper
import os # For getpid and makedirs in init
import difflib # Added for diff generation

# Relative imports for when this module is part of the package
try:
    from . import config
    from . import logger_setup
    from . import system_analyzer
    from . import ollama_interface
    from . import enhancement_applier
except ImportError:
    # Fallback for direct execution (e.g., if __name__ == '__main__')
    # This allows running orchestrator.py directly for some tests IF other modules are accessible
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add project root
    from ai_os_enhancer import config
    from ai_os_enhancer import logger_setup
    from ai_os_enhancer import system_analyzer
    from ai_os_enhancer import ollama_interface
    from ai_os_enhancer import enhancement_applier

# Initialize logger for this module
if __name__ == '__main__' and not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Orchestrator_direct")
    logger.info("Running Orchestrator directly, basic logging configured.")
else:
    logger = logger_setup.setup_logger("Orchestrator")

# --- Orchestrator Module ---

class Orchestrator:
    def __init__(self):
        self.current_enhancements_in_progress = []
        self.system_stability_score = 100.0
        self.human_intervention_required = False
        self.is_initialized = False

    def _log_user_alert(self, message, level="INFO"):
        """ Helper to log messages intended for user visibility. """
        level_upper = level.upper()
        log_message = f"USER_ALERT ({level_upper}): {message}"

        if level_upper == "CRITICAL":
            logger.critical(log_message)
        elif level_upper == "WARNING":
            logger.warning(log_message)
        elif level_upper == "ERROR":
            logger.error(log_message)
        else:
            logger.info(log_message)

        print(log_message)


    def initialize_system(self):
        """ Initializes the AI OS Enhancer system. """
        logger.info("AI OS Enhancer Initializing...")

        try:
            paths_to_check = {
                "Config DB Path": config.CONFIG_DATABASE_PATH,
                "Log Dir": config.LOG_FILE_PATH.parent,
                "Backup Base Path": config.BACKUP_BASE_PATH
            }
            for name, path_obj in paths_to_check.items():
                if not path_obj.exists():
                    logger.warning(f"{name} at {path_obj} not found, attempting to create.")
                    os.makedirs(path_obj, exist_ok=True)

            logger.info(f"Using database path: {config.CONFIG_DATABASE_PATH}")
            logger.info(f"Log file configured at: {config.LOG_FILE_PATH}")
            logger.info(f"Backups will be stored under: {config.BACKUP_BASE_PATH}")
        except Exception as e:
            logger.critical(f"Failed to ensure essential directories: {e}", exc_info=True)
            self.human_intervention_required = True
            self._log_user_alert(f"Critical error during directory setup: {e}. Cannot continue.", "CRITICAL")
            return

        debian_version = system_analyzer.get_debian_version()
        if debian_version:
            logger.info(f"System Analyzer detected OS: {debian_version}")
        else:
            logger.warning("Could not determine Debian version. Functionality may be limited.")

        logger.info("System initialized.")
        self.is_initialized = True


    def request_human_approval_if_needed(self, enhancement_proposal):
        """
        Requests human approval for an enhancement based on risk and impact.
        Returns: Boolean (approved or not).
        """
        risk = enhancement_proposal.get("risk_assessment", "MEDIUM").upper()
        impact = enhancement_proposal.get("impact_level", "MODERATE").upper()
        justification = enhancement_proposal.get("justification", "No justification provided.")
        item_path_str = enhancement_proposal.get("item_path", "Unknown item") # Renamed to avoid conflict

        needs_approval = False
        approval_threshold = str(config.HUMAN_APPROVAL_THRESHOLD).upper()

        if approval_threshold == "LOW":
            needs_approval = True
        elif approval_threshold == "MEDIUM":
            if risk in ["MEDIUM", "HIGH"] or impact == "SIGNIFICANT":
                needs_approval = True
        elif approval_threshold == "HIGH":
            if risk == "HIGH" or (risk == "MEDIUM" and impact == "SIGNIFICANT"):
                needs_approval = True

        if needs_approval:
            proposal_summary = f"Approval for '{item_path_str}'. Risk: {risk}, Impact: {impact}. Justification: {justification}"
            self._log_user_alert(proposal_summary, "WARNING")
            try:
                while True:
                    response = input("Approve this change? (yes/no/details): ").strip().lower()
                    if response == "yes":
                        logger.info(f"Enhancement APPROVED by human: {item_path_str}")
                        return True
                    elif response == "no":
                        logger.info(f"Enhancement REJECTED by human: {item_path_str}")
                        return False
                    elif response == "details":
                        print("--- Enhancement Proposal JSON Details ---")
                        try:
                            print(json.dumps(enhancement_proposal, indent=2, sort_keys=True))
                        except Exception as e:
                            print(f"Error formatting JSON details: {e}\nRaw: {enhancement_proposal}")
                        print("-------------------------------------")

                        # Diff generation logic starts here
                        proposed_details = enhancement_proposal.get("proposed_change_details", {})
                        item_type = enhancement_proposal.get("item_type")
                        requires_generation = proposed_details.get("requires_code_generation", False)
                        change_type_local = proposed_details.get("type", "").lower() # Renamed to avoid conflict
                        is_new_file_creation = "create_new" in change_type_local or "new_script_creation" in change_type_local

                        if is_new_file_creation:
                            print("\nThis proposal is for creating a new file. No diff against an existing file is applicable.")
                            new_content_task_desc = proposed_details.get("task_description", "Content to be generated by AI.")
                            print(f"Content will be: {new_content_task_desc}")
                            if proposed_details.get("language"):
                                print(f"Language: {proposed_details.get('language')}")

                        elif requires_generation:
                            print("\nThis proposal requires new code/content to be generated by the LLM.")
                            print("A diff of the final proposed change against the current file is not yet available.")
                            print(f"Task for LLM: {proposed_details.get('task_description', justification)}")
                            current_snippet = enhancement_proposal.get("current_relevant_content_snippet")
                            if current_snippet:
                                print("\n--- Current Relevant Snippet ---")
                                print(current_snippet)
                                print("------------------------------")
                        elif not item_path_str: # Check renamed item_path_str
                            print("\nCannot generate diff: Item path is missing in the proposal.")
                        else:
                            current_content = system_analyzer.read_file_content(item_path_str)
                            if current_content is None:
                                print(f"\nCould not read current content of {item_path_str} to generate diff.")
                            else:
                                proposed_final_content = None
                                pre_defined_content = proposed_details.get("new_code_snippet") or \
                                                      proposed_details.get("new_content") or \
                                                      proposed_details.get("new_line_content") or \
                                                      proposed_details.get("block_content")

                                if item_type == "config":
                                    config_change_type = str(proposed_details.get("type", "")).lower() # Use a local var
                                    old_snippet_marker = proposed_details.get("target_marker_or_snippet") or \
                                                         proposed_details.get("target_pattern") or \
                                                         proposed_details.get("target_line_pattern")

                                    if "append" in config_change_type and pre_defined_content is not None:
                                        proposed_final_content = current_content + ("\n" if current_content else "") + pre_defined_content
                                    elif "prepend" in config_change_type and pre_defined_content is not None:
                                        proposed_final_content = pre_defined_content + ("\n" if current_content else "") + current_content
                                    elif ("overwrite" in config_change_type or "replace_entire" in config_change_type) and pre_defined_content is not None:
                                        proposed_final_content = pre_defined_content
                                    elif old_snippet_marker and pre_defined_content is not None and old_snippet_marker in current_content:
                                        proposed_final_content = current_content.replace(old_snippet_marker, pre_defined_content, 1)
                                    elif pre_defined_content is not None:
                                         logger.warning(f"Diff display: Unclear config change type for {item_path_str} with pre-defined content and no clear markers. Assuming overwrite for diff purposes.")
                                         proposed_final_content = pre_defined_content
                                    else:
                                        print(f"\nCould not determine proposed content for config diff from details: {proposed_details}")

                                elif item_type == "script":
                                    mod_type = proposed_details.get("type")
                                    if mod_type == "replace_entire_script" and pre_defined_content is not None:
                                        proposed_final_content = pre_defined_content
                                    elif mod_type == "append_to_script" and pre_defined_content is not None:
                                        proposed_final_content = current_content + ("\n" if current_content else "") + pre_defined_content
                                    elif mod_type == "prepend_to_script" and pre_defined_content is not None:
                                        proposed_final_content = pre_defined_content + ("\n" if current_content else "") + current_content
                                    elif mod_type == "replace_bash_function" and proposed_details.get("function_name") and pre_defined_content:
                                        print(f"\nProposed new code for function '{proposed_details.get('function_name')}':")
                                        print("--- Proposed New Function Code ---")
                                        print(pre_defined_content)
                                        print("----------------------------------")
                                        proposed_final_content = None
                                    else:
                                         print(f"\nCould not determine proposed content for script diff from details: {proposed_details}")

                                if proposed_final_content is not None:
                                    print("\n--- Proposed Change Diff ---")
                                    current_content_lines = current_content.splitlines(keepends=True)
                                    proposed_content_lines = proposed_final_content.splitlines(keepends=True)

                                    diff_lines = list(difflib.unified_diff(
                                        current_content_lines, proposed_content_lines,
                                        fromfile=f"a/{item_path_str}", tofile=f"b/{item_path_str}", lineterm=""
                                    ))
                                    if diff_lines:
                                        for line_diff_item in diff_lines: # Renamed loop variable
                                            print(line_diff_item, end='')
                                        if diff_lines and not diff_lines[-1].endswith('\n'):
                                            print()
                                    else:
                                        print("No textual changes proposed (or content is identical).")
                                    print("--------------------------")
                                elif not (item_type == "script" and proposed_details.get("type") == "replace_bash_function" and pre_defined_content):
                                    print("\nCould not generate a diff for the proposed change with available details.")
                        print("---------------------------") # End of details section
                    else:
                        print("Invalid input. Please enter 'yes', 'no', or 'details'.")
            except KeyboardInterrupt:
                logger.warning(f"Approval process for '{item_path_str}' interrupted by user (Ctrl+C). Assuming rejection.")
                return False
            except EOFError:
                logger.warning(f"EOFError during input for '{item_path_str}'. Assuming rejection in non-interactive mode.")
                return False
        else:
            logger.info(f"Enhancement auto-approved (Risk: {risk}, Impact: {impact}) for '{item_path_str}'.")
            return True


    def monitor_system_health(self):
        """ Placeholder for system health monitoring. """
        logger.debug("Monitoring system health (basic check)...")

        critical_services_to_check = getattr(config, 'CRITICAL_SERVICES_MONITOR', ["cron", "ssh"])
        current_stability_penalty = 0

        for service_name in critical_services_to_check:
            status = system_analyzer.get_service_status(service_name)
            if status is None or status not in ["active", "running", "activating"]:
                logger.warning(f"Monitored service '{service_name}' is NOT active/running. Status: {status}")
                current_stability_penalty += 15
            else:
                logger.debug(f"Monitored service '{service_name}' is {status}.")

        if current_stability_penalty > 0:
            self.system_stability_score = max(0, self.system_stability_score - current_stability_penalty)
            logger.warning(f"System stability score reduced to {self.system_stability_score:.2f} due to service issues.")
        elif self.system_stability_score < 100:
            self.system_stability_score = min(100, self.system_stability_score + 1)
            logger.debug(f"System stability score slightly recovered to {self.system_stability_score:.2f}")

        if self.system_stability_score < 50:
            alert_msg = f"System stability critically low ({self.system_stability_score:.2f})!"
            logger.critical(alert_msg)
            self._log_user_alert(alert_msg, "CRITICAL")
            self.human_intervention_required = True

        logger.info(f"Current system stability score: {self.system_stability_score:.2f}")
        return self.system_stability_score


    def _format_item_for_analysis(self, item_path_str, item_type):
        """ Helper to read item content and prepare for analysis. """
        item_content = system_analyzer.read_file_content(item_path_str)
        if item_content is not None:
            return {
                "path": item_path_str,
                "content": item_content,
                "type": item_type
            }
        logger.warning(f"Could not read content for {item_type} at {item_path_str}")
        return None

    def main_enhancement_cycle(self):
        """ Main cycle for analyzing, conceiving, and applying enhancements. """
        if not self.is_initialized:
            logger.error("Orchestrator not initialized. Skipping enhancement cycle.")
            return
        if self.human_intervention_required:
             logger.warning("Human intervention flag is active. Skipping new enhancement cycle until cleared.")
             return

        logger.info("--- Starting new enhancement cycle ---")

        logger.info("Phase 1: Analyzing System State...")
        items_to_analyze_specs = system_analyzer.list_key_config_and_script_areas()

        analysis_tasks_queue = []
        for item_area in items_to_analyze_specs:
            area_type = item_area["type"]
            for path_str in item_area["paths"]: # Renamed to avoid conflict
                if system_analyzer.pathlib.Path(path_str).exists():
                    analysis_task = self._format_item_for_analysis(path_str, area_type)
                    if analysis_task: analysis_tasks_queue.append(analysis_task)
                else:
                    logger.debug(f"Path {path_str} from spec does not exist, skipping analysis.")

        if not analysis_tasks_queue:
            logger.info("No existing items found to analyze in this cycle based on current configuration.")
            logger.info("--- Enhancement cycle ended (no items to analyze) ---")
            return

        analysis_results_list = []
        max_analyses = config.MAX_CONCURRENT_ANALYSES if hasattr(config, 'MAX_CONCURRENT_ANALYSES') else 1
        logger.info(f"Analyzing up to {max_analyses} items sequentially (concurrency not implemented).")

        for i, task in enumerate(analysis_tasks_queue[:max_analyses]):
            logger.info(f"Analyzing item {i+1}/{len(analysis_tasks_queue[:max_analyses])}: ({task['type']}) {task['path']}")
            ollama_analysis = ollama_interface.analyze_system_item(
                item_content=task['content'], item_path=task['path'], item_type=task['type']
            )
            if ollama_analysis:
                analysis_results_list.append({
                    "item_path": task['path'], "item_type": task['type'], "analysis": ollama_analysis
                })
            else:
                logger.warning(f"Failed to get analysis from Ollama for {task['path']}.")
                self.system_stability_score = max(0, self.system_stability_score - 2)

        if not analysis_results_list:
            logger.info("No system items were successfully analyzed by Ollama.")
            logger.info("--- Enhancement cycle ended (no analysis results) ---")
            return

        logger.info("Phase 2: Conceiving Enhancement Strategy...")
        system_snapshot = {
            "debian_version": system_analyzer.get_debian_version(),
            "installed_packages_count": len(system_analyzer.get_installed_packages()),
            "system_stability_score": self.system_stability_score,
            "monitored_scripts_count": len(config.MONITORED_SCRIPTS_PATHS),
        }
        logger.debug(f"System snapshot for strategy conception: {system_snapshot}")
        enhancement_plan = ollama_interface.conceive_enhancement_strategy(system_snapshot, analysis_results_list)

        if not enhancement_plan or not enhancement_plan.get("prioritized_enhancements"):
            logger.info("No enhancements conceived by Ollama in this cycle.")
            logger.info("--- Enhancement cycle ended (no plan from Ollama) ---")
            return

        logger.info(f"Overall AI Strategy: {enhancement_plan.get('overall_strategy_summary', 'Not provided.')}")

        logger.info("Phase 3: Applying Enhancements...")
        for enhancement in enhancement_plan.get("prioritized_enhancements", []):
            if self.human_intervention_required:
                self._log_user_alert("Halting further enhancements in this cycle due to prior critical issue or instability.", "WARNING")
                break

            item_path_enh = enhancement.get("item_path") # Renamed to avoid conflict
            item_type_enh = enhancement.get("item_type")
            proposed_details_enh = enhancement.get("proposed_change_details", {})
            change_type_enh = str(proposed_details_enh.get("type", "")).lower()

            log_msg_enh_cycle = f"Considering enhancement for '{item_path_enh}' ({item_type_enh}, type: {change_type_enh}): {enhancement.get('justification', 'N/A')}"
            logger.info(log_msg_enh_cycle)
            logger.info(f"Risk: {enhancement.get('risk_assessment', 'N/A')}, Impact: {enhancement.get('impact_level', 'N/A')}")
            logger.debug(f"Full proposed change details: {json.dumps(proposed_details_enh, indent=2)}")

            if not all([item_path_enh, item_type_enh, proposed_details_enh, change_type_enh]):
                logger.error(f"Skipping enhancement due to missing critical fields. Details: {enhancement}")
                self.system_stability_score = max(0, self.system_stability_score - 5)
                continue

            if not self.request_human_approval_if_needed(enhancement):
                logger.info(f"Enhancement skipped (human disapproval or non-interactive rejection): {item_path_enh}")
                continue

            is_new_file = "create_new" in change_type_enh or "new_script_creation" in change_type_enh
            backup_file_path = None
            if not is_new_file:
                if not system_analyzer.pathlib.Path(item_path_enh).exists():
                    logger.error(f"Target item '{item_path_enh}' does not exist for modification. Skipping.")
                    self.system_stability_score = max(0, self.system_stability_score - 3)
                    continue
                backup_file_path = enhancement_applier.backup_file(item_path_enh)
                if not backup_file_path:
                    logger.error(f"Backup failed for '{item_path_enh}'. Skipping enhancement.")
                    self.system_stability_score = max(0, self.system_stability_score - 10)
                    continue

            apply_success = False
            content_for_change = None
            task_desc_for_llm = proposed_details_enh.get('task_description', enhancement.get('justification', 'Apply system enhancement.'))
            language_for_llm = proposed_details_enh.get("language", "bash" if item_type_enh == "script" else "text")

            if proposed_details_enh.get("requires_code_generation", False):
                logger.info(f"Requesting LLM to generate content for '{item_path_enh}'. Task: {task_desc_for_llm}")
                content_for_change = ollama_interface.generate_code_or_modification(
                    task_description=task_desc_for_llm,
                    language=language_for_llm,
                    existing_code_context=enhancement.get("current_relevant_content_snippet"),
                    modification_target_details=proposed_details_enh
                )
                if not content_for_change:
                    logger.error(f"Failed to generate content from Ollama for: {item_path_enh}")
                    self.system_stability_score = max(0, self.system_stability_score - 5)
                    continue
            else:
                content_keys = ["code_to_insert_or_replace", "new_code_snippet", "new_content", "new_line_content", "block_content"]
                for key in content_keys:
                    if key in proposed_details_enh:
                        content_for_change = proposed_details_enh[key]
                        break
                if content_for_change is None and not is_new_file:
                    logger.warning(f"No code generation requested, and no pre-defined content found in proposed_change_details for existing item '{item_path_enh}'.")

            if is_new_file:
                if content_for_change is None:
                     logger.info(f"Content for new file '{item_path_enh}' not pre-defined, generating now.")
                     content_for_change = ollama_interface.generate_code_or_modification(task_desc_for_llm, language_for_llm)

                if content_for_change is not None:
                    make_executable = item_type_enh == "script"
                    apply_success = enhancement_applier.create_new_file(item_path_enh, content_for_change, make_executable)
                else:
                    logger.error(f"Failed to obtain content for new {item_type_enh}: {item_path_enh}")

            elif item_type_enh == "script":
                if content_for_change is not None:
                    apply_success = enhancement_applier.apply_script_modification(item_path_enh, proposed_details_enh, content_for_change, backup_file_path)
                else:
                    logger.error(f"No content (generated or pre-defined) for script modification: {item_path_enh}")

            elif item_type_enh == "config":
                old_snippet_keys_cfg = ["target_marker_or_snippet", "target_pattern", "target_line_pattern"] # Renamed
                old_snippet_val_cfg = next((proposed_details_enh[k] for k in old_snippet_keys_cfg if k in proposed_details_enh), None) # Renamed

                # Handle special modes like APPEND_MODE, PREPEND_MODE, OVERWRITE_MODE if specified in proposed_details.type
                config_change_type_apply = str(proposed_details_enh.get("type", "")).lower() # Renamed
                if "append" in config_change_type_apply: old_snippet_val_cfg = "APPEND_MODE"
                elif "prepend" in config_change_type_apply: old_snippet_val_cfg = "PREPEND_MODE"
                elif "overwrite" in config_change_type_apply or "replace_entire" in config_change_type_apply : old_snippet_val_cfg = "OVERWRITE_MODE"

                if content_for_change is not None and old_snippet_val_cfg is not None:
                    apply_success = enhancement_applier.apply_config_text_change(
                        item_path_enh, old_snippet_val_cfg, content_for_change, backup_file_path
                    )
                elif old_snippet_val_cfg is None:
                     logger.error(f"Cannot apply config change to {item_path_enh}: missing target snippet/pattern and not append/prepend/overwrite.")
                else:
                     logger.warning(f"Config change for {item_path_enh} specified target but no new content; 'delete' type not fully supported.")
            else:
                logger.warning(f"Unknown enhancement item_type: {item_type_enh} for path {item_path_enh}")

            if apply_success:
                logger.info(f"Enhancement APPLIED successfully for: {item_path_enh}")
                self.system_stability_score = min(100, self.system_stability_score + 5)
            else:
                logger.error(f"Failed to apply enhancement for: {item_path_enh}")
                self.system_stability_score = max(0, self.system_stability_score - 15)

            self.monitor_system_health()

            if self.human_intervention_required:
                self._log_user_alert(f"System instability after attempting change to {item_path_enh}. Intervention needed.", "CRITICAL")
                if not is_new_file and backup_file_path and system_analyzer.pathlib.Path(backup_file_path).exists():
                    logger.info(f"Attempting general rollback for {item_path_enh} due to instability.")
                    if enhancement_applier.rollback_change(backup_file_path, item_path_enh):
                        logger.info(f"Successfully rolled back {item_path_enh}.")
                        self.system_stability_score = max(0, self.system_stability_score - 20)
                    else:
                        logger.critical(f"CRITICAL: Failed to rollback {item_path_enh} despite instability!")
                elif is_new_file and system_analyzer.pathlib.Path(item_path_enh).exists():
                    logger.info(f"Attempting to delete newly created file {item_path_enh} due to instability.")
                    try: system_analyzer.pathlib.Path(item_path_enh).unlink(); logger.info(f"Deleted {item_path_enh}.")
                    except Exception as e_del: logger.error(f"Failed to delete {item_path_enh}: {e_del}")
                break

        logger.info(f"--- Enhancement cycle completed. Current Stability: {self.system_stability_score:.2f} ---")


    def run(self):
        """ Main execution loop for the Orchestrator. """
        global logger
        if not logger.hasHandlers() or isinstance(logger, logging.RootLogger) or logger.name == "Orchestrator_direct":
            logger = logger_setup.setup_logger("Orchestrator", log_level=logging.INFO)
            logger.info("Orchestrator logger re-initialized for run.")

        logger.info(f"AI OS Enhancer starting up. PID: {os.getpid()}")
        logger.info(f"Default Ollama Model: {config.DEFAULT_MODEL}")
        logger.info(f"Human Approval Threshold: {config.HUMAN_APPROVAL_THRESHOLD}")
        logger.info(f"Monitored script paths: {config.MONITORED_SCRIPTS_PATHS}")

        self.initialize_system()

        if self.human_intervention_required:
            self._log_user_alert("Halting due to critical errors during initialization.", "CRITICAL")
            return

        try:
            cycle_count = 0
            is_test_run = os.getenv("AIOS_TEST_QUICK_CYCLE", "false").lower() == "true"

            while True:
                cycle_count += 1
                logger.info(f"====== Starting Orchestrator Cycle #{cycle_count} (Stability: {self.system_stability_score:.2f}) ======")

                min_stability_threshold = getattr(config, 'MIN_STABILITY_FOR_CYCLE', 30)
                if self.system_stability_score < min_stability_threshold:
                    logger.error(f"System stability ({self.system_stability_score:.2f}) is below threshold ({min_stability_threshold}). Skipping enhancement cycle. Requiring human intervention.")
                    self.human_intervention_required = True
                else:
                    self.main_enhancement_cycle()

                self.monitor_system_health()

                if self.human_intervention_required:
                    self._log_user_alert("System requires human intervention. Pausing autonomous operations.", "CRITICAL")
                    try:
                        user_input = input("Human intervention required. Type 'continue' to resume after fixing, or 'stop' to exit: ").strip().lower()
                        if user_input == 'stop':
                            logger.info("Orchestrator stopped by user command.")
                            break
                        elif user_input == 'continue':
                            self.human_intervention_required = False
                            self.system_stability_score = max(50, self.system_stability_score)
                            logger.info("Human intervention flag 'cleared' by user. Resuming operations.")
                        else:
                            logger.warning("Invalid command. Assuming pause continues.")
                    except (KeyboardInterrupt, EOFError):
                        logger.info("Orchestrator stopped by user (Ctrl+C or EOF) during human intervention pause.")
                        break

                sleep_interval_seconds = 30 if is_test_run else getattr(config, 'CYCLE_SLEEP_INTERVAL_SECONDS', 3600)
                if self.system_stability_score < 60 and not is_test_run : sleep_interval_seconds *= 2

                logger.info(f"Cycle #{cycle_count} finished. Next cycle in {sleep_interval_seconds}s.")
                time.sleep(sleep_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Orchestrator run loop interrupted by user (Ctrl+C). Shutting down.")
        except Exception as e:
            logger.critical(f"Orchestrator encountered a critical unhandled exception: {e}", exc_info=True)
            self._log_user_alert(f"Orchestrator CRASHED: {e}", "CRITICAL")
        finally:
            logger.info("AI OS Enhancer shutting down.")

if __name__ == '__main__':
