import logging
import json # For formatting text data if needed by OllamaInterface's helper
import os # For getpid and makedirs in init
import difflib # Added for diff generation

# Relative imports for when this module is part of the package
try:
    # If Orchestrator is part of a package, use relative imports
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
# The global 'logger' will be initialized by logger_setup.setup_logger()
# However, for direct execution (__name__ == '__main__'), we might need to initialize it
# if it hasn't been, or re-initialize it with specific settings for the main orchestrator.
if __name__ == '__main__':
    # If this script is run directly, ensure a basic logger is set up if not already.
    # This might be redundant if logger_setup.setup_logger is called early in Orchestrator.run()
    if not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("Orchestrator_direct_main_setup") # Unique name for this specific setup
    else:
        # If logger might be already configured by a previous import of logger_setup
        logger = logger_setup.setup_logger("Orchestrator_main_block", log_level=logging.INFO)
else:
    # When imported as a module, use the standard logger setup.
    logger = logger_setup.setup_logger("Orchestrator")


class Orchestrator:
    def __init__(self):
        """ Initializes the Orchestrator. """
        # Logger setup: ensure self.logger is consistently used within the class.
        # If the global logger is already configured by logger_setup, this might just re-fetch it.
        # It's important that logger_setup.setup_logger can be called multiple times without issue
        # (e.g., it retrieves an existing logger if already configured).
        self.logger = logger_setup.setup_logger("Orchestrator_instance") # Or use the global 'logger' directly
        self.logger.info("Orchestrator instance created.")
        self.initialize_system()

    def _log_user_alert(self, message, level="INFO"):
        """ Logs a message and prints it to the console for user visibility. """
        # Ensure self.logger is used, or the global logger if preferred.
        log_message = f"[USER_ALERT] {message}"
        level_upper = level.upper()
        
        if level_upper == "CRITICAL":
            self.logger.critical(log_message)
        elif level_upper == "WARNING":
            self.logger.warning(log_message)
        elif level_upper == "ERROR": 
            self.logger.error(log_message)
        else: 
            self.logger.info(log_message)

        print(log_message) # Always print for user visibility


    def initialize_system(self):
        """ Initializes system state, paths, and configurations. """
        self.logger.info("Orchestrator initializing system...")
        self.project_root = config.PROJECT_ROOT 
        self.db_path = config.CONFIG_DATABASE_PATH
        self.log_file = config.LOG_FILE_PATH
        self.backup_path = config.BACKUP_BASE_PATH

        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.log_file.parent, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)

        self.logger.info(f"Project Root: {self.project_root}")
        self.logger.info(f"DB Path: {self.db_path}")
        self.logger.info(f"Log File: {self.log_file}")
        self.logger.info(f"Backup Path: {self.backup_path}")
        
        self.system_stability_score = 100.0
        self.human_intervention_required = False
        self.enhancement_history = []
        self._load_persistent_state()
        self.logger.info("System initialization complete.")

    def _load_persistent_state(self):
        self.logger.debug("Loading persistent state (placeholder).")

    def _save_persistent_state(self):
        self.logger.debug("Saving persistent state (placeholder).")

    def _is_approval_required(self, enhancement_proposal: dict) -> bool:
        """Checks if human approval is required based on risk/impact and configuration."""
        risk = enhancement_proposal.get("risk_assessment", "HIGH").upper()
        impact = enhancement_proposal.get("impact_level", "SIGNIFICANT").upper()
        # Ensure config.HUMAN_APPROVAL_THRESHOLD is accessed correctly
        approval_threshold = str(config.HUMAN_APPROVAL_THRESHOLD).upper() 

        self.logger.debug(f"Checking approval: Risk='{risk}', Impact='{impact}', Threshold='{approval_threshold}'")

        if approval_threshold == "LOW":
            return not (risk == "LOW" and impact == "MINIMAL")
        elif approval_threshold == "MEDIUM":
            return risk in ["MEDIUM", "HIGH"] or impact == "SIGNIFICANT"
        elif approval_threshold == "HIGH":
            # Approval only for HIGH risk, or MEDIUM risk if impact is SIGNIFICANT
            return risk == "HIGH" or (risk == "MEDIUM" and impact == "SIGNIFICANT")
        
        self.logger.warning(f"Unknown HUMAN_APPROVAL_THRESHOLD value: {approval_threshold}. Defaulting to requiring approval.")
        return True

    def _display_proposal_diff(self, enhancement_proposal: dict) -> None:
        """Generates and prints a diff for the proposed change."""
        item_path = enhancement_proposal.get("item_path")
        item_type = enhancement_proposal.get("item_type") # Used for context
        proposed_details = enhancement_proposal.get("proposed_change_details", {})
        change_type = str(proposed_details.get("type", "")).lower()
        is_new_file = "create_new" in change_type or "new_script_creation" in change_type

        if is_new_file:
            self._log_user_alert("\nThis proposal is for creating a NEW FILE.", "INFO")
            new_content_desc = proposed_details.get("task_description", "Content to be generated by AI.")
            self._log_user_alert(f"Content will be based on: {new_content_desc}", "INFO")
            if proposed_details.get("language"):
                self._log_user_alert(f"Language: {proposed_details.get('language')}", "INFO")
            
            pre_defined_content = proposed_details.get("new_code_snippet") or proposed_details.get("new_content")
            if pre_defined_content:
                self._log_user_alert("\n--- Proposed New File Content ---", "INFO")
                print(pre_defined_content) # Direct print
                self._log_user_alert("-----------------------------", "INFO")
            return

        if not item_path:
            self._log_user_alert("\nCannot generate diff: Item path is missing in the proposal.", "WARNING")
            return

        # Ensure system_analyzer is accessible, e.g. via self or direct import
        current_content = system_analyzer.read_file_content(item_path)
        if current_content is None:
            self._log_user_alert(f"\nCould not read current content of {item_path} to generate diff. File might be new or inaccessible.", "WARNING")
            # If it's a new file but not marked as 'create_new', it's an issue.
            # However, if content is None, we can't diff.
            # If there's pre-defined content, we can show that as the "new" content.
            pre_defined_content_for_empty = proposed_details.get("new_code_snippet") or proposed_details.get("new_content")
            if pre_defined_content_for_empty:
                self._log_user_alert(f"\n--- Proposed Content for {item_path} (current content inaccessible/empty) ---", "INFO")
                print(pre_defined_content_for_empty)
                self._log_user_alert("--------------------------------------------------------------------", "INFO")
            return


        proposed_final_content = None
        pre_defined_content = proposed_details.get("new_code_snippet") or \
                              proposed_details.get("new_content") or \
                              proposed_details.get("new_line_content") or \
                              proposed_details.get("block_content")

        if proposed_details.get("requires_code_generation", False) and not pre_defined_content:
            self._log_user_alert("\nThis proposal requires new code/content to be generated by the LLM.", "INFO")
            self._log_user_alert("A diff of the final proposed change against the current file cannot be shown until generation.", "INFO")
            self._log_user_alert(f"Task for LLM: {proposed_details.get('task_description', enhancement_proposal.get('justification', 'N/A'))}", "INFO")
            current_snippet = enhancement_proposal.get("current_relevant_content_snippet")
            if current_snippet:
                self._log_user_alert("\n--- Current Relevant Snippet (context for LLM) ---", "INFO")
                print(current_snippet)
                self._log_user_alert("------------------------------------------------", "INFO")
            return

        # Constructing proposed_final_content based on change type for diff purposes
        if item_type == "config":
            target_marker = proposed_details.get("target_marker_or_snippet") or \
                            proposed_details.get("target_pattern") or \
                            proposed_details.get("target_line_pattern")

            if "append" in change_type and pre_defined_content is not None:
                proposed_final_content = current_content + ("\n" if current_content.strip() else "") + pre_defined_content
            elif "prepend" in change_type and pre_defined_content is not None:
                proposed_final_content = pre_defined_content + ("\n" if current_content.strip() else "") + current_content
            elif ("overwrite" in change_type or "replace_entire" in change_type) and pre_defined_content is not None:
                proposed_final_content = pre_defined_content
            elif target_marker and pre_defined_content is not None and target_marker in current_content:
                proposed_final_content = current_content.replace(target_marker, pre_defined_content, 1)
            elif pre_defined_content is not None:
                self.logger.debug(f"Diff display: Config change type for {item_path} is '{change_type}'. Assuming overwrite for diff purposes as other conditions not met.")
                proposed_final_content = pre_defined_content
            else:
                self._log_user_alert(f"\nCould not determine proposed content for config diff. Details: {proposed_details}", "WARNING")
        
        elif item_type == "script": # Simplified logic for script diff
            if pre_defined_content is not None:
                 # For scripts, if pre_defined_content exists, assume it's the new full script or relevant new block
                 # More specific diffs (like function replacement) are hard to show generically here.
                 # The actual application logic in enhancement_applier will handle precise changes.
                 # For user diff, showing the pre_defined_content against old is the best guess.
                if "replace_entire_script" in change_type:
                    proposed_final_content = pre_defined_content
                elif "append_to_script" in change_type:
                     proposed_final_content = current_content + ("\n" if current_content.strip() else "") + pre_defined_content
                elif "prepend_to_script" in change_type:
                     proposed_final_content = pre_defined_content + ("\n" if current_content.strip() else "") + current_content
                else: # For other script changes with pre-defined content, show it as replacing current
                    self.logger.debug(f"Diff display: Script change type for {item_path} is '{change_type}'. Showing diff of pre-defined content against current.")
                    proposed_final_content = pre_defined_content # This might be a full replacement or a targeted one.
            else:
                self._log_user_alert(f"\nNo pre-defined content for script modification, cannot generate simple diff. Details: {proposed_details}", "WARNING")

        if proposed_final_content is not None:
            self._log_user_alert("\n--- Proposed Change Diff ---", "INFO")
            current_content_lines = current_content.splitlines(keepends=True)
            proposed_content_lines = proposed_final_content.splitlines(keepends=True)

            diff_lines = list(difflib.unified_diff(
                current_content_lines, proposed_content_lines,
                fromfile=f"a/{item_path}", tofile=f"b/{item_path}", lineterm=""
            ))
            if diff_lines:
                for line_diff_item in diff_lines: # Print diff lines directly
                    print(line_diff_item, end='') 
                if diff_lines and not diff_lines[-1].endswith('\n'): # Ensure last line has a newline
                    print()
            else:
                self._log_user_alert("No textual changes proposed (or content is identical).", "INFO")
            self._log_user_alert("--------------------------", "INFO")
        else:
             self._log_user_alert("\nCould not generate a preview diff for the proposed change with available details.", "WARNING")

    def _display_proposal_details(self, enhancement_proposal: dict) -> None:
        """Prints JSON details of the proposal and calls to display the diff."""
        self._log_user_alert("--- Enhancement Proposal JSON Details ---", "INFO")
        try:
            print(json.dumps(enhancement_proposal, indent=2, sort_keys=True))
        except Exception as e:
            print(f"Error formatting JSON details: {e}\nRaw: {enhancement_proposal}")
        self._log_user_alert("-------------------------------------", "INFO")
        self._display_proposal_diff(enhancement_proposal)

    def _get_user_approval_input(self, proposal_summary: str, enhancement_proposal: dict) -> bool:
        """Handles the user input loop for approving or rejecting a proposal."""
        item_path = enhancement_proposal.get("item_path", "Unknown item")
        self._log_user_alert(f"\n{proposal_summary}", "WARNING") # Initial summary to user
        try:
            while True:
                # Ensure prompt is printed to console, not just logger
                print("Approve this change? (yes/no/details): ", end='') 
                response = input().strip().lower() # Read from stdin

                if response == "yes":
                    self.logger.info(f"Enhancement APPROVED by human: {item_path}")
                    return True
                elif response == "no":
                    self.logger.info(f"Enhancement REJECTED by human: {item_path}")
                    return False
                elif response == "details":
                    self._display_proposal_details(enhancement_proposal)
                    # Re-prompt after showing details
                    self._log_user_alert(f"\n{proposal_summary}", "WARNING") 
                else:
                    self._log_user_alert("Invalid input. Please enter 'yes', 'no', or 'details'.", "WARNING")
        except KeyboardInterrupt:
            self.logger.warning(f"Approval process for '{item_path}' interrupted by user (Ctrl+C). Assuming rejection.")
            self._log_user_alert(f"Approval for '{item_path}' CANCELED by user.", "WARNING")
            return False
        except EOFError: 
            self.logger.warning(f"EOFError during input for '{item_path}'. Assuming rejection in non-interactive mode.")
            self._log_user_alert(f"Approval for '{item_path}' CANCELED due to EOF.", "WARNING")
            return False
        return False 

    def request_human_approval_if_needed(self, enhancement_proposal: dict) -> bool:
        """
        Requests human approval for an enhancement based on risk and impact.
        Uses helper methods for logic separation.
        """
        item_path = enhancement_proposal.get("item_path", "Unknown item")
        
        if not self._is_approval_required(enhancement_proposal):
            risk = enhancement_proposal.get("risk_assessment", "N/A").upper() # For logging
            impact = enhancement_proposal.get("impact_level", "N/A").upper() # For logging
            self.logger.info(f"Enhancement auto-approved (Risk: {risk}, Impact: {impact}) for '{item_path}'.")
            return True

        # Construct summary for _get_user_approval_input if approval is needed
        risk = enhancement_proposal.get("risk_assessment", "HIGH").upper()
        impact = enhancement_proposal.get("impact_level", "SIGNIFICANT").upper()
        justification = enhancement_proposal.get("justification", "No justification provided.")
        proposal_summary = f"Approval for '{item_path}'. Risk: {risk}, Impact: {impact}. Justification: {justification}"
        
        return self._get_user_approval_input(proposal_summary, enhancement_proposal)

    def monitor_system_health(self):
        """ Placeholder for system health monitoring. """
        self.logger.debug("Monitoring system health (basic check)...")

        critical_services_to_check = getattr(config, 'CRITICAL_SERVICES_MONITOR', ["cron", "ssh"])
        current_stability_penalty = 0

        for service_name in critical_services_to_check:
            if not system_analyzer.is_service_active(service_name): # Assuming system_analyzer is imported
                self.logger.warning(f"Critical service '{service_name}' is not active.")
                current_stability_penalty += 10 
            else:
                self.logger.debug(f"Service '{service_name}' is active.")
        
        if current_stability_penalty > 0:
            self.system_stability_score = max(0, self.system_stability_score - current_stability_penalty)
            self.logger.warning(f"System stability score reduced to {self.system_stability_score:.2f} due to service issues.")
        elif self.system_stability_score < 100: 
            self.system_stability_score = min(100, self.system_stability_score + 1) 
            self.logger.debug(f"System stability score slightly recovered to {self.system_stability_score:.2f}")

        if self.system_stability_score < 50: 
            alert_msg = f"System stability critically low ({self.system_stability_score:.2f})!"
            self.logger.critical(alert_msg)
            self._log_user_alert(alert_msg, "CRITICAL")
            self.human_intervention_required = True 

        self.logger.info(f"Current system stability score: {self.system_stability_score:.2f}")
        return self.system_stability_score

    def _format_item_for_analysis(self, item_path_str, item_type):
        """ Helper to read item content and prepare for analysis. """
        item_content = system_analyzer.read_file_content(item_path_str) # Assuming system_analyzer is imported
        if item_content is not None: 
            return {
                "path": item_path_str,
                "content": item_content, 
                "type": item_type
            }
        self.logger.warning(f"Could not read content for {item_path_str}, skipping analysis for this item.")
        return None # Return None if content can't be read

    def main_enhancement_cycle(self):
        """ Core logic for one cycle of analysis, strategy, and application. """
        self.logger.info(f"--- Starting Enhancement Cycle. Stability: {self.system_stability_score:.2f} ---")
        
        if self.human_intervention_required:
            self._log_user_alert("Human intervention required. System paused. Please resolve issues and restart.", "CRITICAL")
            return False # Indicate cycle cannot proceed

        # 1. Analyze System Items
        items_to_analyze_specs = config.MONITORED_SCRIPTS_PATHS # This needs to be structured correctly in config
        if not isinstance(items_to_analyze_specs, list) or not all(isinstance(item, dict) for item in items_to_analyze_specs):
            self.logger.error("MONITORED_SCRIPTS_PATHS in config.py is not a list of dicts. Cannot proceed with analysis.")
            # Example structure: MONITORED_SCRIPTS_PATHS = [{"path": "/path/to/script.sh", "type": "script"}, {"path": "/etc/config.conf", "type": "config"}]
            # For now, if it's a simple list of paths, adapt:
            if isinstance(items_to_analyze_specs, list) and all(isinstance(p, str) for p in items_to_analyze_specs):
                 self.logger.warning("MONITORED_SCRIPTS_PATHS is a simple list of paths. Assuming 'script' type for all.")
                 items_to_analyze_specs = [{"path": p, "type": "script"} for p in items_to_analyze_specs] # Convert to expected structure
            else: # If not even a list of strings, then it's unusable.
                 items_to_analyze_specs = []


        analysis_tasks_queue = []
        for item_spec in items_to_analyze_specs: # Changed from item_area to item_spec
            path_str = item_spec.get("path")
            area_type = item_spec.get("type", "script") # Default to script if type missing
            if not path_str:
                self.logger.warning(f"Skipping item spec due to missing 'path': {item_spec}")
                continue

            if system_analyzer.pathlib.Path(path_str).exists(): # Assuming system_analyzer is imported
                analysis_task = self._format_item_for_analysis(path_str, area_type)
                if analysis_task: analysis_tasks_queue.append(analysis_task)
            else:
                self.logger.debug(f"Path {path_str} from spec does not exist, skipping analysis.")

        if not analysis_tasks_queue:
            self.logger.info("No existing items found to analyze in this cycle based on current configuration.")
            # Potentially look for new items or opportunities even if pre-defined list is empty/non-existent
            # For now, we'll just end the cycle here if nothing to analyze.
            return True # Cycle completed, but did nothing.

        all_analysis_results = []
        # MAX_CONCURRENT_ANALYSES is now used as a sequential limit for simplicity
        max_analyses = config.MAX_CONCURRENT_ANALYSES if hasattr(config, 'MAX_CONCURRENT_ANALYSES') else 1
        self.logger.info(f"Analyzing up to {max_analyses} items sequentially.")

        for i, task in enumerate(analysis_tasks_queue[:max_analyses]): 
            self.logger.info(f"Analyzing item {i+1}/{len(analysis_tasks_queue[:max_analyses])}: ({task['type']}) {task['path']}")
            # Ensure ollama_interface is accessible
            ollama_analysis = ollama_interface.analyze_system_item(
                item_content=task['content'], item_path=task['path'], item_type=task['type']
            )
            if ollama_analysis and not ollama_analysis.get("error"):
                # Append item_path and item_type for context in strategy phase
                ollama_analysis["item_path"] = task['path'] 
                ollama_analysis["item_type"] = task['type']
                all_analysis_results.append(ollama_analysis)
                self.logger.debug(f"Analysis result for {task['path']}: {json.dumps(ollama_analysis, indent=2)}")
            else:
                self.logger.error(f"Failed to analyze {task['path']}. Error: {ollama_analysis.get('error', 'Unknown error') if ollama_analysis else 'No response'}")
                self.system_stability_score = max(0, self.system_stability_score - 1) # Small penalty for analysis failure

        if not all_analysis_results:
            self.logger.info("No successful analysis results obtained. Cannot proceed to strategy phase.")
            return True

        # 2. Conceive Enhancement Strategy
        self.logger.info("Conceiving enhancement strategy based on analysis results...")
        system_snapshot = system_analyzer.get_system_snapshot() # Assuming system_analyzer is imported
        enhancement_strategy_result = ollama_interface.conceive_enhancement_strategy(system_snapshot, all_analysis_results)

        if not enhancement_strategy_result or enhancement_strategy_result.get("error"):
            self.logger.error(f"Failed to conceive enhancement strategy. Error: {enhancement_strategy_result.get('error', 'Unknown error') if enhancement_strategy_result else 'No response'}")
            self.system_stability_score = max(0, self.system_stability_score - 5)
            return True
        
        self.logger.info(f"Enhancement Strategy Summary: {enhancement_strategy_result.get('overall_strategy_summary', 'N/A')}")
        prioritized_enhancements = enhancement_strategy_result.get("prioritized_enhancements", [])

        if not prioritized_enhancements:
            self.logger.info("No enhancements proposed in the current strategy.")
            return True

        # 3. Apply Enhancements (Iterate through prioritized list)
        for enhancement in prioritized_enhancements:
            if self.human_intervention_required or self.system_stability_score < config.MIN_STABILITY_FOR_ENHANCEMENT:
                self._log_user_alert("Halting further enhancements in this cycle due to prior critical issue or instability.", "WARNING")
                break

            item_path_enh = enhancement.get("item_path")
            item_type_enh = enhancement.get("item_type") 
            proposed_details_enh = enhancement.get("proposed_change_details", {}) # This was in original, ensure it's used or remove
            change_type_enh = str(enhancement.get("change_type", "")).lower() # Get change_type directly from enhancement ifflat

            # Use proposed_change_details if it exists and has 'type', otherwise use top-level 'change_type'
            if "type" in proposed_details_enh: # Check if detailed structure is present
                change_type_enh = str(proposed_details_enh.get("type", "")).lower()
            
            self.logger.info(f"Considering enhancement for '{item_path_enh}' ({item_type_enh}, type: {change_type_enh}): {enhancement.get('justification', 'N/A')}")
            self.logger.info(f"Risk: {enhancement.get('risk_assessment', 'N/A')}, Impact: {enhancement.get('impact_level', 'N/A')}")
            self.logger.debug(f"Full proposed enhancement details: {json.dumps(enhancement, indent=2)}")


            if not all([item_path_enh, item_type_enh, change_type_enh]): # Simplified check
                self.logger.error(f"Skipping enhancement due to missing critical fields (path, type, or change_type). Details: {enhancement}")
                self.system_stability_score = max(0, self.system_stability_score - 5)
                continue

            # Human approval request
            if not self.request_human_approval_if_needed(enhancement): # Pass the whole enhancement dict
                self.logger.info(f"Enhancement skipped (human disapproval or non-interactive rejection): {item_path_enh}")
                continue

            is_new_file = "create_new" in change_type_enh or "new_script_creation" in change_type_enh
            backup_file_path = None
            if not is_new_file:
                if not system_analyzer.pathlib.Path(item_path_enh).exists():
                    self.logger.error(f"Target item '{item_path_enh}' does not exist for modification. Skipping.")
                    self.system_stability_score = max(0, self.system_stability_score - 3)
                    continue
                backup_file_path = enhancement_applier.backup_file(item_path_enh) # Ensure enhancement_applier is imported
                if not backup_file_path:
                    self.logger.error(f"Backup failed for '{item_path_enh}'. Skipping enhancement.")
                    self.system_stability_score = max(0, self.system_stability_score - 10)
                    continue
            
            apply_success = False
            # Content for change might be directly in proposal or need generation
            content_for_change = enhancement.get("proposed_change_snippet") # Standardized key from strategy
            
            if not content_for_change and enhancement.get("requires_code_generation", False): # If snippet missing and gen required
                self.logger.info(f"Requesting LLM to generate content for '{item_path_enh}'. Task: {enhancement.get('enhancement_description', 'Apply system enhancement.')}")
                content_for_change = ollama_interface.generate_code_or_modification(
                    task_description=enhancement.get('enhancement_description', 'Apply system enhancement.'),
                    language=enhancement.get("language", "bash" if item_type_enh == "script" else "text"), # Assuming language is part of proposal
                    existing_code_context=enhancement.get("current_relevant_content_snippet") # If available in proposal
                )
                if not content_for_change:
                    self.logger.error(f"Failed to generate content from Ollama for: {item_path_enh}")
                    self.system_stability_score = max(0, self.system_stability_score - 5)
                    continue
                else: # Update the proposal with the generated content for consistent application
                    enhancement["proposed_change_snippet"] = content_for_change


            if is_new_file:
                if content_for_change is not None:
                    make_executable = item_type_enh == "script" 
                    apply_success = enhancement_applier.create_new_file(item_path_enh, content_for_change, make_executable)
                else:
                    self.logger.error(f"Failed to obtain content for new {item_type_enh}: {item_path_enh}")
            elif item_type_enh == "script":
                # Script modifications need careful handling based on 'change_type' and content
                # This part assumes enhancement_applier.apply_script_modification can handle various types
                if content_for_change is not None: # Requires content_for_change for most script mods
                    apply_success = enhancement_applier.apply_script_modification(
                        item_path_enh, 
                        enhancement, # Pass the whole enhancement dict which contains details
                        content_for_change, 
                        backup_file_path
                    )
                else:
                    self.logger.error(f"No content (generated or pre-defined) for script modification: {item_path_enh}")
            elif item_type_enh == "config":
                target_criteria = enhancement.get("target_criteria") # e.g. line, pattern, or special like APPEND_MODE
                if content_for_change is not None and target_criteria is not None:
                    apply_success = enhancement_applier.apply_config_text_change(
                        item_path_enh, target_criteria, content_for_change, backup_file_path
                    )
                elif target_criteria is None:
                     self.logger.error(f"Cannot apply config change to {item_path_enh}: missing target_criteria.")
                elif content_for_change is None and not (target_criteria == "DELETE_MODE" or "delete" in change_type_enh) : # Allow if it's a deletion
                     self.logger.error(f"Config change for {item_path_enh} specified target but no new content, and not a deletion.")
                else: # Handle deletion if supported by applier
                    if target_criteria == "DELETE_MODE" or "delete" in change_type_enh:
                        apply_success = enhancement_applier.apply_config_text_change(
                            item_path_enh, target_criteria, "", backup_file_path # Empty content for deletion
                        )
                    else:
                        self.logger.warning(f"Unhandled config change case for {item_path_enh}")
            else:
                self.logger.warning(f"Unknown enhancement item_type: {item_type_enh} for path {item_path_enh}")

            if apply_success:
                self.logger.info(f"Enhancement APPLIED successfully for: {item_path_enh}")
                self.system_stability_score = min(100, self.system_stability_score + 5) # Increase stability
                self.enhancement_history.append({"applied": enhancement, "status": "success", "backup": backup_file_path})
            else:
                self.logger.error(f"Failed to apply enhancement for: {item_path_enh}")
                self.system_stability_score = max(0, self.system_stability_score - 15) # Significant penalty
                self.enhancement_history.append({"applied": enhancement, "status": "failure", "backup": backup_file_path})


            self.monitor_system_health() 

            if self.human_intervention_required:
                self._log_user_alert(f"System instability after attempting change to {item_path_enh}. Intervention needed.", "CRITICAL")
                if not is_new_file and backup_file_path and system_analyzer.pathlib.Path(backup_file_path).exists():
                    self.logger.info(f"Attempting general rollback for {item_path_enh} due to instability.")
                    if enhancement_applier.rollback_change(backup_file_path, item_path_enh):
                        self.logger.info(f"Successfully rolled back {item_path_enh}.")
                        self.system_stability_score = max(0, self.system_stability_score - 20) # Further penalty for rollback
                    else:
                        self.logger.critical(f"CRITICAL: Failed to rollback {item_path_enh} despite instability!")
                elif is_new_file and system_analyzer.pathlib.Path(item_path_enh).exists(): 
                    self.logger.info(f"Attempting to delete newly created file {item_path_enh} due to instability.")
                    try: 
                        system_analyzer.pathlib.Path(item_path_enh).unlink()
                        self.logger.info(f"Deleted {item_path_enh}.")
                    except Exception as e_del: 
                        self.logger.error(f"Failed to delete {item_path_enh}: {e_del}")
                break # Stop further enhancements this cycle

        self._save_persistent_state() # Save state after cycle
        self.logger.info(f"--- Enhancement cycle completed. Current Stability: {self.system_stability_score:.2f} ---")
        return True # Cycle completed (even if some actions failed or were skipped)


    def run(self):
        """ Main execution loop for the Orchestrator. """
        # Ensure logger is correctly set up for the Orchestrator run
        # self.logger is already initialized in __init__
        self.logger.info(f"AI OS Enhancer starting up. PID: {os.getpid()}")
        self.logger.info(f"Using Human Approval Threshold: {config.HUMAN_APPROVAL_THRESHOLD}")
        self.logger.info(f"Monitored script paths: {config.MONITORED_SCRIPTS_PATHS}")
        self.logger.info(f"Ollama API Endpoint: {config.OLLAMA_API_ENDPOINT}")
        self.logger.info(f"Default Model: {config.DEFAULT_MODEL}")

        if not system_analyzer.check_ollama_service_availability(config.OLLAMA_API_ENDPOINT):
            self.logger.critical(f"Ollama service not available at {config.OLLAMA_API_ENDPOINT}. Please ensure Ollama is running.")
            self._log_user_alert(f"Ollama service not available at {config.OLLAMA_API_ENDPOINT}. Exiting.", "CRITICAL")
            return

        try:
            cycle_count = 0
            is_test_run = os.getenv("AIOS_TEST_QUICK_CYCLE", "false").lower() == "true"

            while True:
                cycle_count += 1
                self.logger.info(f"--- Orchestrator Cycle {cycle_count} ---")
                
                if not self.main_enhancement_cycle(): # If cycle indicates a hard stop (e.g. human intervention)
                    self.logger.warning("Main enhancement cycle indicated a stop. Orchestrator pausing.")
                    break 

                if self.human_intervention_required:
                    self.logger.critical("Human intervention required. Orchestrator pausing.")
                    break
                
                if is_test_run:
                    self.logger.info("AIOS_TEST_QUICK_CYCLE is true. Orchestrator will exit after one cycle.")
                    break

                self.logger.info(f"Waiting for {config.CYCLE_INTERVAL_SECONDS} seconds before next cycle...")
                # Use time.sleep for waiting
                import time
                time.sleep(config.CYCLE_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            self.logger.info("Orchestrator run interrupted by user (Ctrl+C).")
        except Exception as e:
            self.logger.critical(f"An uncaught exception occurred in Orchestrator run loop: {e}", exc_info=True)
        finally:
            self._save_persistent_state() # Attempt to save state on exit
            self.logger.info("AI OS Enhancer shutting down.")

if __name__ == '__main__':
    # Setup basic logger for direct execution if not already done by module-level check
    # The Orchestrator class itself will get/create its own logger instance via logger_setup.
    # This __main__ block logger is primarily for messages printed directly from this block.
    main_logger_name = "Orchestrator_DirectRun"
    clilogger = logger_setup.setup_logger(main_logger_name, log_level=logging.DEBUG) # Use a distinct name
    
    clilogger.info("Starting Orchestrator directly...")
    orchestrator_instance = Orchestrator()
    orchestrator_instance.run()
    clilogger.info("Orchestrator direct run finished.")
