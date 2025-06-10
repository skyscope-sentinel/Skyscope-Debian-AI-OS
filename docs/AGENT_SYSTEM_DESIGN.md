# Skyscope Sentinel - Multi-Agent System Design (Initial)

This document outlines the initial design for the multi-agent capabilities within the Skyscope Sentinel AI OS Enhancer. The goal is to enable multiple specialized AI agents to collaborate or work on distinct tasks, managed by a central system but all utilizing a single (potentially local) Ollama LLM instance efficiently.

## Core Concepts

### 1. Agent Context (`AgentContext`)

Each AI agent operating within the system is represented by an `AgentContext`. This context encapsulates all information unique to that agent, ensuring its state and identity are maintained separately from other agents.

Key attributes of `AgentContext` (defined in `ai_os_enhancer/agent_context.py`):
- **`agent_id`**: A unique string identifier for the agent.
- **`role_name`**: The name of a configured AI Persona (Role) that defines the agent's expertise, system prompt, preferred model, and output expectations. (See "AI Personas (Roles)" in `README.md`).
- **`conversation_history`**: A list of messages (`{"role": "system/user/assistant", "content": "..."}`) that maintains the dialogue history between this specific agent and the LLM. The initial system message is derived from its assigned role.
- **`current_objective`**: A high-level task or goal assigned to the agent.
- **`state`**: A flexible dictionary for the agent to store any other arbitrary data, notes, or intermediate results relevant to its tasks.
- **Timestamps**: `creation_timestamp`, `last_interaction_timestamp`.

### 2. Agent Manager (`AgentManager`)

The `AgentManager` (defined in `ai_os_enhancer/agent_manager.py`) is responsible for creating, managing, and orchestrating the interaction turns for all registered `AgentContext` instances.

Key responsibilities:
- **Agent Lifecycle:** Registering new agents (associating them with a role and objective) and unregistering them.
- **Context Management:** Storing and retrieving `AgentContext` objects.
- **Turn Processing (`process_agent_turn`)**:
    - Takes an `agent_id` and new `input_content` for that agent.
    - Retrieves the agent's current role configuration and full conversation history.
    - Constructs the appropriate message list for the Ollama API (typically using the `/api/chat` endpoint).
    - Invokes the LLM via `ollama_interface.query_ollama_chat`.
    - Updates the specific agent's conversation history with the new user input and the LLM's response.
    - Returns the LLM's response.

### 3. Agent Multiplexing (Conceptual)

The current implementation of `AgentManager` processes agent turns sequentially. This is the foundational step towards the "Agent Multiplexing via Pipes" concept described in the Phase II directive.

The core idea is that a single, locally-run Ollama model instance can serve multiple distinct AI agents. The `AgentManager` ensures that each agent's interaction (its unique context, history, and role-defined persona) is "piped" to the LLM in a serialized, turn-based manner. The LLM's response is then piped back and applied only to the state of the agent for whom the request was made.

Strict context isolation is maintained by the `AgentContext` objects and the `AgentManager`'s targeted updates. There is no "context bleed" between agents sharing the same LLM instance because only one agent's full context is presented to the LLM at any given time for a specific request-response cycle.

Future iterations may explore concurrent processing of turns using asynchronous programming or other techniques if the Ollama instance can handle concurrent requests efficiently, while still maintaining this strict context isolation per request.

## Interaction with Roles System

The multi-agent system heavily relies on the "AI Personas (Roles)" feature. Each agent is assigned a `role_name` upon registration. This role dictates:
- The **system prompt** used to initialize the agent's conversation history, defining its core persona and instructions.
- The **Ollama model** to be used (can override the global default).
- **Knowledge base keywords** to provide relevant context.
- The **expected output format** (e.g., JSON) and keys for response validation.

This allows for creating a team of specialized agents, each with its own expertise and behavioral guidelines.
