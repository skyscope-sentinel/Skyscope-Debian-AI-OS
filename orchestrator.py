# ai_os_enhancer/orchestrator.py

import time
import logging
import json # For formatting text data if needed by OllamaInterface's helper
import os # For getpid and makedirs in init

# Relative imports for when this module is part of the package
try:
    from . import config
    from . import logger_setup
    from . import system_analyzer
    from . import ollama_interface
    from . import enhancement_applier
except ImportError:
    # Fallback for direct execution (e.g., if __name__ == '__main__')
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    import logger_setup
    import system_analyzer
    import ollama_interface
    import enhancement_applier

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
        elif level_upper == "ERROR": # Added ERROR level
            logger.error(log_message)
        else: # Default to INFO
            logger.info(log_message)
        
        print(log_message) # Always print for CLI visibility


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
        item_path = enhancement_proposal.get("item_path", "Unknown item")

        needs_approval = False
        # Ensure HUMAN_APPROVAL_THRESHOLD is uppercase for reliable comparison
        approval_threshold = str(config.HUMAN_APPROVAL_THRESHOLD).upper()

        if approval_threshold == "LOW":
            needs_approval = True 
        elif approval_threshold == "MEDIUM":
            if risk in ["MEDIUM", "HIGH"] or impact == "SIGNIFICANT":
                needs_approval = True
        elif approval_threshold == "HIGH":
            if risk == "HIGH" or (risk == "MEDIUM" and impact == "SIGNIFICANT"): # Adjusted HIGH slightly
                needs_approval = True
        
        if needs_approval:
            proposal_summary = f"Approval for '{item_path}'. Risk: {risk}, Impact: {impact}. Justification: {justification}"
            self._log_user_alert(proposal_summary, "WARNING")
            try:
                while True:
                    response = input("Approve this change? (yes/no/details): ").strip().lower()
                    if response == "yes":
                        logger.info(f"Enhancement APPROVED by human: {item_path}")
                        return True
                    elif response == "no":
                        logger.info(f"Enhancement REJECTED by human: {item_path}")
                        return False
                    elif response == "details":
                        print("--- Enhancement Details ---")
                        try:
                            print(json.dumps(enhancement_proposal, indent=2, sort_keys=True))
                        except Exception as e:
                            print(f"Error formatting details: {e}\nRaw: {enhancement_proposal}")
                        print("---------------------------")
                    else:
                        print("Invalid input. Please enter 'yes', 'no', or 'details'.")
            except KeyboardInterrupt:
                logger.warning(f"Approval process for '{item_path}' interrupted by user (Ctrl+C). Assuming rejection.")
                return False
            except EOFError: # Handle if input stream closes (e.g. in automated non-interactive test)
                logger.warning(f"EOFError during input for '{item_path}'. Assuming rejection in non-interactive mode.")
                return False
        else:
            logger.info(f"Enhancement auto-approved (Risk: {risk}, Impact: {impact}) for '{item_path}'.")
            return True


    def monitor_system_health(self):
        """ Placeholder for system health monitoring. """
        logger.debug("Monitoring system health (basic check)...")
        
        # Example: Check a critical service (make this configurable)
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
        elif self.system_stability_score < 100: # No new issues, and not at max
            self.system_stability_score = min(100, self.system_stability_score + 1) 
            logger.debug(f"System stability score slightly recovered to {self.system_stability_score:.2f}")

        if self.system_stability_score < 50: 
            alert_msg = f"System stability critically low ({self.system_stability_score:.2f})!"
            logger.critical(alert_msg)
            self._log_user_alert(alert_msg, "CRITICAL")
            self.human_intervention_required = True # This requires human to reset
        
        logger.info(f"Current system stability score: {self.system_stability_score:.2f}")
        return self.system_stability_score


    def _format_item_for_analysis(self, item_path_str, item_type):
        """ Helper to read item content and prepare for analysis. """
        item_content = system_analyzer.read_file_content(item_path_str)
        if item_content is not None: # Check for None, as empty string is valid content
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
            for item_path_str in item_area["paths"]:
                if system_analyzer.pathlib.Path(item_path_str).exists():
                    analysis_task = self._format_item_for_analysis(item_path_str, area_type)
                    if analysis_task: analysis_tasks_queue.append(analysis_task)
                else:
                    logger.debug(f"Path {item_path_str} from spec does not exist, skipping analysis.")
        
        if not analysis_tasks_queue:
            logger.info("No existing items found to analyze in this cycle based on current configuration.")
            logger.info("--- Enhancement cycle ended (no items to analyze) ---")
            return

        analysis_results_list = []
        max_analyses = config.MAX_CONCURRENT_ANALYSES if hasattr(config, 'MAX_CONCURRENT_ANALYSES') else 1
        logger.info(f"Analyzing up to {max_analyses} items sequentially (concurrency not implemented).")
        
        for i, task in enumerate(analysis_tasks_queue[:max_analyses]): # Simple sequential for now
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

            item_path = enhancement.get("item_path")
            item_type = enhancement.get("item_type") # 'config' or 'script'
            proposed_details = enhancement.get("proposed_change_details", {})
            change_type = str(proposed_details.get("type", "")).lower()
            
            log_msg_enh = f"Considering enhancement for '{item_path}' ({item_type}, type: {change_type}): {enhancement.get('justification', 'N/A')}"
            logger.info(log_msg_enh)
            logger.info(f"Risk: {enhancement.get('risk_assessment', 'N/A')}, Impact: {enhancement.get('impact_level', 'N/A')}")
            logger.debug(f"Full proposed change details: {json.dumps(proposed_details, indent=2)}")

            if not all([item_path, item_type, proposed_details, change_type]):
                logger.error(f"Skipping enhancement due to missing critical fields. Details: {enhancement}")
                self.system_stability_score = max(0, self.system_stability_score - 5)
                continue

            if not self.request_human_approval_if_needed(enhancement):
                logger.info(f"Enhancement skipped (human disapproval or non-interactive rejection): {item_path}")
                continue
            
            is_new_file = "create_new" in change_type or "new_script_creation" in change_type
            backup_file_path = None
            if not is_new_file:
                if not system_analyzer.pathlib.Path(item_path).exists():
                    logger.error(f"Target item '{item_path}' does not exist for modification. Skipping.")
                    self.system_stability_score = max(0, self.system_stability_score - 3)
                    continue
                backup_file_path = enhancement_applier.backup_file(item_path)
                if not backup_file_path:
                    logger.error(f"Backup failed for '{item_path}'. Skipping enhancement.")
                    self.system_stability_score = max(0, self.system_stability_score - 10)
                    continue
            
            apply_success = False
            content_for_change = None
            task_desc_for_llm = proposed_details.get('task_description', enhancement.get('justification', 'Apply system enhancement.'))
            language_for_llm = proposed_details.get("language", "bash" if item_type == "script" else "text")

            if proposed_details.get("requires_code_generation", False):
                logger.info(f"Requesting LLM to generate content for '{item_path}'. Task: {task_desc_for_llm}")
                content_for_change = ollama_interface.generate_code_or_modification(
                    task_description=task_desc_for_llm,
                    language=language_for_llm,
                    existing_code_context=enhancement.get("current_relevant_content_snippet"),
                    modification_target_details=proposed_details 
                )
                if not content_for_change:
                    logger.error(f"Failed to generate content from Ollama for: {item_path}")
                    self.system_stability_score = max(0, self.system_stability_score - 5)
                    continue 
            else: # Content should be in proposed_details
                content_keys = ["code_to_insert_or_replace", "new_code_snippet", "new_content", "new_line_content", "block_content"]
                for key in content_keys:
                    if key in proposed_details:
                        content_for_change = proposed_details[key]
                        break
                if content_for_change is None and not is_new_file: # New file might generate content later
                    logger.warning(f"No code generation requested, and no pre-defined content found in proposed_change_details for existing item '{item_path}'. Trying to proceed if change type allows.")
            
            # Apply the change
            if is_new_file:
                if content_for_change is None: # If not generated/provided earlier, generate now
                     logger.info(f"Content for new file '{item_path}' not pre-defined, generating now.")
                     content_for_change = ollama_interface.generate_code_or_modification(task_desc_for_llm, language_for_llm)
                
                if content_for_change is not None:
                    make_executable = item_type == "script" # and language_for_llm in ["bash", "sh"]
                    apply_success = enhancement_applier.create_new_file(item_path, content_for_change, make_executable)
                else:
                    logger.error(f"Failed to obtain content for new {item_type}: {item_path}")
            
            elif item_type == "script":
                if content_for_change is not None:
                    apply_success = enhancement_applier.apply_script_modification(item_path, proposed_details, content_for_change, backup_file_path)
                else:
                    logger.error(f"No content (generated or pre-defined) for script modification: {item_path}")
            
            elif item_type == "config":
                old_snippet_keys = ["target_marker_or_snippet", "target_pattern", "target_line_pattern"]
                old_snippet_val = next((proposed_details[k] for k in old_snippet_keys if k in proposed_details), None)

                if "append" in change_type: old_snippet_val = "APPEND_MODE"
                elif "prepend" in change_type: old_snippet_val = "PREPEND_MODE"
                elif "overwrite" in change_type or "replace_entire" in change_type: old_snippet_val = "OVERWRITE_MODE"
                
                if content_for_change is not None and old_snippet_val is not None:
                    apply_success = enhancement_applier.apply_config_text_change(item_path, old_snippet_val, content_for_change, backup_file_path)
                elif old_snippet_val is None:
                     logger.error(f"Cannot apply config change to {item_path}: missing target snippet/pattern and not append/prepend/overwrite.")
                else: # content_for_change is None
                     logger.warning(f"Config change for {item_path} specified target but no new content; 'delete' type not fully supported.")
            else:
                logger.warning(f"Unknown enhancement item_type: {item_type} for path {item_path}")

            # Post-Change
            if apply_success:
                logger.info(f"Enhancement APPLIED successfully for: {item_path}")
                self.system_stability_score = min(100, self.system_stability_score + 5)
            else:
                logger.error(f"Failed to apply enhancement for: {item_path}")
                self.system_stability_score = max(0, self.system_stability_score - 15)
            
            self.monitor_system_health() # Check health after each attempt

            if self.human_intervention_required:
                self._log_user_alert(f"System instability after attempting change to {item_path}. Intervention needed.", "CRITICAL")
                if not is_new_file and backup_file_path and system_analyzer.pathlib.Path(backup_file_path).exists():
                    logger.info(f"Attempting general rollback for {item_path} due to instability.")
                    if enhancement_applier.rollback_change(backup_file_path, item_path):
                        logger.info(f"Successfully rolled back {item_path}.")
                        self.system_stability_score = max(0, self.system_stability_score - 20)
                    else:
                        logger.critical(f"CRITICAL: Failed to rollback {item_path} despite instability!")
                elif is_new_file and system_analyzer.pathlib.Path(item_path).exists(): # Cleanup created file
                    logger.info(f"Attempting to delete newly created file {item_path} due to instability.")
                    try: system_analyzer.pathlib.Path(item_path).unlink(); logger.info(f"Deleted {item_path}.")
                    except Exception as e_del: logger.error(f"Failed to delete {item_path}: {e_del}")
                break 

        logger.info(f"--- Enhancement cycle completed. Current Stability: {self.system_stability_score:.2f} ---")


    def run(self):
        """ Main execution loop for the Orchestrator. """
        # Ensure logger_setup.setup_logger is called for the main run if not already configured.
        global logger # To potentially re-assign if it's the basicConfig one
        if not logger.hasHandlers() or isinstance(logger, logging.RootLogger) or logger.name == "Orchestrator_direct":
            logger = logger_setup.setup_logger("Orchestrator", log_level=logging.INFO) # Default to INFO for run
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
            # Use a very short cycle for testing if a specific environment variable is set
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
    # This allows running the orchestrator directly for testing.
    sample_scripts_dir = config.PROJECT_ROOT / "sample_scripts_for_orchestrator"
    os.makedirs(sample_scripts_dir, exist_ok=True)
    test_script_path = sample_scripts_dir / "orchestrator_test_script.sh"
    with open(test_script_path, "w") as f:
        f.write("#!/bin/bash\n#This is a test script for the AI OS Enhancer.\necho 'Orchestrator test script running'\n# A small inefficiency for the AI to potentially find:\nfor i in 1 2 3 4 5; do\n    echo \"Loop iteration: $i\"\n    sleep 0.1 # Simulate work\ndone\n\necho 'Test script finished.'\nexit 0\n")
    os.chmod(test_script_path, 0o755)
    
    original_monitored_paths = list(config.MONITORED_SCRIPTS_PATHS) 
    if str(test_script_path) not in config.MONITORED_SCRIPTS_PATHS:
        config.MONITORED_SCRIPTS_PATHS.append(str(test_script_path))
    
    # For testing, you might want to set HUMAN_APPROVAL_THRESHOLD to "LOW" or "MEDIUM"
    # in config.py to interactively approve/reject changes.
    # You can also set AIOS_TEST_QUICK_CYCLE=true as an environment variable for faster cycles.
    print(f"Orchestrator test setup: '{test_script_path}' is configured for monitoring.")
    print(f"Default model: {config.DEFAULT_MODEL}. Approval: {config.HUMAN_APPROVAL_THRESHOLD}.")
    print("Ensure Ollama is running for full test. The test will run cycles based on config or env var.")
    print("Use Ctrl+C to stop.")

    orchestrator_instance = Orchestrator()
    orchestrator_instance.run()

    # Cleanup (optional)
    # config.MONITORED_SCRIPTS_PATHS = original_monitored_paths
    # try: os.remove(test_script_path) except OSError: pass
    # try: os.rmdir(sample_scripts_dir) except OSError: pass
