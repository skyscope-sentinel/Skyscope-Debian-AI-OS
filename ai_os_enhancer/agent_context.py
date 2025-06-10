# ai_os_enhancer/agent_context.py
import datetime
from typing import List, Dict, Any, Optional

class AgentContext:
    def __init__(self, agent_id: str, role_name: str, initial_objective: Optional[str] = None):
        self.agent_id: str = agent_id
        self.role_name: str = role_name
        self.conversation_history: List[Dict[str, str]] = [] # List of {"role": "...", "content": "..."}
        self.current_objective: Optional[str] = initial_objective
        self.state: Dict[str, Any] = {} # For agent-specific state persistence within a session
        self.creation_timestamp: datetime.datetime = datetime.datetime.now()
        self.last_interaction_timestamp: Optional[datetime.datetime] = None
        # Role config (system_prompt, model etc.) will be loaded and used by AgentManager, not stored here.

    def add_message_to_history(self, role: str, content: str):
        """Adds a message to the agent's conversation history."""
        if role not in ["system", "user", "assistant"]:
            import logging
            logger_instance = logging.getLogger("AgentContext")
            # Ensure logger has handlers, especially if this class is used early or in isolation
            if not logger_instance.hasHandlers():
                 logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logger_instance.warning(f"Unconventional role '{role}' being added to history for agent '{self.agent_id}'. Valid roles are 'system', 'user', 'assistant'.")

        self.conversation_history.append({"role": role, "content": content})
        self.last_interaction_timestamp = datetime.datetime.now()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Returns a copy of the conversation history."""
        return list(self.conversation_history)

    def update_state(self, key: str, value: Any):
        """Updates a key-value pair in the agent's state."""
        self.state[key] = value
        self.last_interaction_timestamp = datetime.datetime.now()

    def get_state(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieves a value from the agent's state."""
        return self.state.get(key, default)

    def set_objective(self, objective: str):
        """Sets or updates the agent's current objective."""
        self.current_objective = objective
        self.last_interaction_timestamp = datetime.datetime.now()
        # Optionally, log this change or add a special message to history if needed for LLM context
        # self.add_message_to_history("system", f"Objective updated by agent to: {objective}")


    def __repr__(self) -> str:
        return (f"AgentContext(agent_id='{self.agent_id}', role_name='{self.role_name}', "
                f"objective='{self.current_objective}', history_len={len(self.conversation_history)}, "
                f"last_interaction='{self.last_interaction_timestamp}')")
