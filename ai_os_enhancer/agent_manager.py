# ai_os_enhancer/agent_manager.py
import logging
import json
import os
import pathlib
import yaml
from typing import Dict, Optional, List, Any

# Relative imports
try:
    from .agent_context import AgentContext
    from . import ollama_interface
    from . import config
    from . import logger_setup
except ImportError:
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from ai_os_enhancer.agent_context import AgentContext
    from ai_os_enhancer import ollama_interface
    from ai_os_enhancer import config
    from ai_os_enhancer import logger_setup

# Setup logger for this module
logger = logger_setup.setup_logger("AgentManager")

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, AgentContext] = {}
        logger.info("AgentManager initialized.")

    def register_agent(self, agent_id: str, role_name: str, initial_objective: Optional[str] = None) -> AgentContext | None:
        if agent_id in self.agents:
            logger.warning(f"Agent with ID '{agent_id}' already registered. Returning existing instance.")
            return self.agents[agent_id]

        role_config = ollama_interface.load_role_config(role_name)
        if not role_config:
            logger.error(f"Failed to register agent '{agent_id}': Role '{role_name}' could not be loaded.")
            return None

        agent = AgentContext(agent_id, role_name, initial_objective)
        system_prompt_template = role_config.get("system_prompt")

        if system_prompt_template:
            prompt_context_for_registration = {
                "agent_id": agent.agent_id,
                "role_name": agent.role_name,
                "initial_objective": initial_objective or "Not specified"
                # Add other static context available at registration if roles use them
            }
            # Format the system prompt using available context.
            # _format_system_prompt handles missing keys gracefully.
            formatted_system_prompt = ollama_interface._format_system_prompt(system_prompt_template, prompt_context_for_registration)
            agent.add_message_to_history("system", formatted_system_prompt)

        self.agents[agent_id] = agent
        logger.info(f"Agent '{agent_id}' registered with role '{role_name}'. Objective: {initial_objective or 'None'}. System prompt (potentially formatted) added to history.")
        return agent

    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent '{agent_id}' unregistered.")
            return True
        logger.warning(f"Agent with ID '{agent_id}' not found for unregistering.")
        return False

    def get_agent_context(self, agent_id: str) -> Optional[AgentContext]:
        agent = self.agents.get(agent_id)
        if not agent:
            logger.warning(f"Agent '{agent_id}' not found in get_agent_context.")
        return agent

    def process_agent_turn(self, agent_id: str, input_content: str, llm_model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        agent = self.get_agent_context(agent_id)
        if not agent:
            logger.error(f"Agent '{agent_id}' not found. Cannot process turn.")
            return {"error": f"Agent '{agent_id}' not found."}

        role_config = ollama_interface.load_role_config(agent.role_name)
        if not role_config:
            logger.error(f"Role '{agent.role_name}' for agent '{agent_id}' not found. Cannot process turn.")
            return {"error": f"Role '{agent.role_name}' not found."}

        # Note: System prompt is set at registration. If per-turn system prompt modification
        # or dynamic placeholder filling with 'input_content' is needed, it would happen here
        # by modifying the 'messages_for_llm' list before calling query_ollama_chat.
        # For now, system prompt is static post-registration.
        # Example for per-turn system prompt formatting if needed:
        # current_system_prompt = agent.get_conversation_history()[0]['content'] # Assuming it's always first
        # turn_context = {"user_input": input_content, ...}
        # formatted_turn_system_prompt = ollama_interface._format_system_prompt(current_system_prompt, turn_context)
        # messages_for_llm = [{"role": "system", "content": formatted_turn_system_prompt}] + agent.get_conversation_history()[1:]
        # messages_for_llm.append({"role": "user", "content": input_content})
        # This is complex; current design keeps system prompt largely static after registration.

        agent.add_message_to_history("user", input_content)
        messages_for_llm = agent.get_conversation_history()


        effective_model_name = llm_model_override or role_config.get("model_name", config.DEFAULT_MODEL)
        output_format = role_config.get("output_format", "text")
        instruct_llm_to_return_json_content = (output_format == "json")

        response_data = ollama_interface.query_ollama_chat(
            model_name=effective_model_name,
            messages=messages_for_llm,
            instruct_llm_to_return_json_content=instruct_llm_to_return_json_content
        )

        if response_data:
            if isinstance(response_data, dict) and response_data.get("error"):
                logger.error(f"Error from Ollama chat for agent '{agent_id}': {response_data['error']}")
                return response_data

            llm_message_obj = response_data.get("message", {})
            llm_response_content_final_form = llm_message_obj.get("content") # This is already dict if parsing succeeded in query_ollama_chat

            history_content_to_add = ""
            if isinstance(llm_response_content_final_form, dict):
                history_content_to_add = json.dumps(llm_response_content_final_form)
            elif isinstance(llm_response_content_final_form, str):
                 history_content_to_add = llm_response_content_final_form
            elif llm_response_content_final_form is not None:
                history_content_to_add = str(llm_response_content_final_form)

            if history_content_to_add or llm_message_obj.get("role") == "assistant": # Add even if content is empty string
                agent.add_message_to_history("assistant", history_content_to_add)
                logger.debug(f"Agent '{agent_id}' received response. History length: {len(agent.conversation_history)}")

            if instruct_llm_to_return_json_content:
                content_to_validate = llm_response_content_final_form
                if isinstance(content_to_validate, dict):
                    valid, missing_keys = ollama_interface._validate_llm_response(content_to_validate, role_config, agent.role_name)
                    if not valid:
                        error_msg = f"LLM response content for agent '{agent_id}' (role '{agent.role_name}') missing expected keys: {missing_keys}."
                        logger.error(f"{error_msg} Full LLM message content: {content_to_validate}")
                        response_data["validation_error"] = error_msg
                        response_data["missing_keys_in_content"] = missing_keys
                else:
                    error_msg = f"LLM response content for agent '{agent_id}' (role '{agent.role_name}') was expected to be JSON, but received type {type(content_to_validate)} (parsing may have failed in query_ollama_chat or LLM did not comply)."
                    logger.error(error_msg + f" Raw content string received by AgentManager was: {history_content_to_add[:200]}")
                    response_data["validation_error"] = error_msg

            return response_data
        else:
            logger.error(f"No response data from Ollama chat for agent '{agent_id}'.")
            return {"error": "No response data from Ollama chat interface.", "message": {"role":"assistant", "content":""}}

    def list_agents(self) -> List[Dict[str, Any]]:
        return [{"agent_id": agent.agent_id, "role_name": agent.role_name, "objective": agent.current_objective}
                for agent in self.agents.values()]

if __name__ == '__main__':
    # Setup logger for direct execution of this script
    # Using the global 'logger' instance which is configured at the top of this file based on __name__
    logger.setLevel(logging.DEBUG) # Ensure debug messages are shown for test
    # Ensure a console handler for visibility when run directly
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if hasattr(logger_setup, 'LOG_FORMAT'): # Use project's standard format if available
            formatter = logging.Formatter(logger_setup.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("--- AgentManager Test Script ---")

    try:
        roles_dir = pathlib.Path(config.PROJECT_ROOT) / "ai_os_enhancer" / "roles"
        roles_dir.mkdir(parents=True, exist_ok=True)

        planner_role_content = {
            "role_name": "TestPlannerRole",
            "version": "1.0.0", "author": "AgentManagerTest", "tags": ["test", "planner"],
            "description": "A test role for planning tasks for an executor agent.",
            "system_prompt": "You are a planner. Your task is to receive a high-level goal: {initial_objective}. Break it down into a specific natural language instruction for an executor agent. The executor agent can generate shell commands. Respond with a JSON object containing the key \"natural_language_task_for_executor\".",
            "output_format": "json",
            "expected_llm_output_keys": ["natural_language_task_for_executor"]
        }
        with open(roles_dir / "testplannerrole.yaml", "w", encoding="utf-8") as f:
            yaml.dump(planner_role_content, f)

        executor_role_content = {
            "role_name": "TestExecutorRole",
            "version": "1.0.0", "author": "AgentManagerTest", "tags": ["test", "executor", "shell"],
            "description": "A test role for generating shell commands from a natural language task.",
            "system_prompt": "You are an executor. You receive a natural language task and must generate a shell command for it. The task is: {user_input}. Respond with a JSON object containing: \"task_description\", \"generated_command\", \"safety_notes\".",
            "output_format": "json",
            "expected_llm_output_keys": ["task_description", "generated_command", "safety_notes"]
        }
        # Note: The executor's system prompt using {user_input} might be unconventional.
        # Typically, the user_input is the content of the "user" message, not a placeholder in "system" message.
        # For testing, we'll see how this plays out or if the LLM handles it.
        # A more standard approach: System prompt is static, user_input is the task.
        # Reverting to a more standard approach for TestExecutorRole system prompt:
        executor_role_content["system_prompt"] = "You are an executor. You receive a natural language task and must generate a shell command for it. Respond with a JSON object containing: \"task_description\", \"generated_command\", \"safety_notes\"."


        with open(roles_dir / "testexecutorrole.yaml", "w", encoding="utf-8") as f:
            yaml.dump(executor_role_content, f)
        logger.info(f"Created dummy roles in {roles_dir}")
    except Exception as e:
        logger.error(f"Could not create dummy role files for test: {e}", exc_info=True)

    manager = AgentManager()

    if os.getenv("AIOS_TEST_AGENT_INTERACTION") == "true":
        clilogger = logger # Alias for clarity in this block
        clilogger.info("\n--- Two-Agent Simulation Test ---")
        agent_planner_id = "Planner" # Simpler ID for test
        agent_executor_id = "Executor"

        initial_goal = "Automate the installation of the 'htop' utility and verify its version."
        planner_agent = manager.register_agent(agent_planner_id, "TestPlannerRole", initial_goal)
        executor_agent = manager.register_agent(agent_executor_id, "TestExecutorRole") # No initial objective for executor

        if not planner_agent or not executor_agent:
            clilogger.error("Failed to register one or both agents. Aborting two-agent simulation.")
        else:
            clilogger.info(f"--- Turn 1: Planner ({planner_agent.agent_id}) - Objective: '{initial_goal}' ---")
            # The initial objective is now part of the planner's system prompt via placeholder.
            # The input_content here is the first user message to the planner.
            planner_user_input = "Please generate the task for the executor based on my objective."
            planner_response_full = manager.process_agent_turn(agent_id=agent_planner_id, input_content=planner_user_input)

            clilogger.info(f"Planner raw response object: {json.dumps(planner_response_full, indent=2) if planner_response_full else 'None'}")

            natural_language_task_for_executor = None
            if planner_response_full and not planner_response_full.get("error") and not planner_response_full.get("validation_error"):
                planner_message_content = planner_response_full.get("message", {}).get("content")
                if isinstance(planner_message_content, dict):
                    natural_language_task_for_executor = planner_message_content.get("natural_language_task_for_executor")
                else: clilogger.error(f"Planner response content was not a dict as expected by role: {planner_message_content}")

            if natural_language_task_for_executor:
                clilogger.info(f"--- Turn 2: Executor ({executor_agent.agent_id}) receives task: '{natural_language_task_for_executor}' ---")
                executor_response_full = manager.process_agent_turn(agent_id=agent_executor_id, input_content=natural_language_task_for_executor)

                clilogger.info(f"Executor raw response object: {json.dumps(executor_response_full, indent=2) if executor_response_full else 'None'}")

                if executor_response_full and not executor_response_full.get("error") and not executor_response_full.get("validation_error"):
                    executor_message_content = executor_response_full.get("message", {}).get("content")
                    if isinstance(executor_message_content, dict):
                        clilogger.info(f"Executor - Task Description: {executor_message_content.get('task_description')}")
                        clilogger.info(f"Executor - Generated Command: {executor_message_content.get('generated_command')}")
                        clilogger.info(f"Executor - Safety Notes: {executor_message_content.get('safety_notes')}")
                    else:
                        clilogger.warning(f"Executor response content was not a parsed dictionary: {executor_message_content}")
                else:
                    clilogger.error(f"Executor turn failed or produced error/validation error: {executor_response_full}")
            else:
                clilogger.error("Planner failed to provide a 'natural_language_task_for_executor'. Cannot proceed to executor turn.")

            clilogger.info("\n--- Planner Conversation History ---")
            if planner_ctx := manager.get_agent_context(agent_planner_id): # Python 3.8+ assignment expression
                for msg in planner_ctx.get_conversation_history(): clilogger.info(f"  {msg['role']}: {msg['content'][:250]}...")

            clilogger.info("\n--- Executor Conversation History ---")
            if executor_ctx := manager.get_agent_context(agent_executor_id):
                for msg in executor_ctx.get_conversation_history(): clilogger.info(f"  {msg['role']}: {msg['content'][:250]}...")

        clilogger.info("--- End of AgentManager Two-Agent Simulation Test ---")
    else:
        clilogger.info("Skipping Two-Agent Simulation Test. Set AIOS_TEST_AGENT_INTERACTION=true to run.")

    clilogger.info("NOTE: Agent interaction tests require a running Ollama instance with a model able to follow JSON structured prompts and role instructions.")
    clilogger.info("--- AgentManager Test Script Finished ---")
