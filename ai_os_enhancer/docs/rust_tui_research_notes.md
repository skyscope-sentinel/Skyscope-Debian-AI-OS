# Rust TUI Research Notes for Skyscope Sentinel

This document outlines initial research into `ratatui` and `crossterm` for building the futuristic TUI, and a potential project structure.

## Core Crates:

### 1. `ratatui`
*   **Purpose:** A Rust library for building Terminal User Interfaces (TUIs).
*   **Functionality:**
    *   Provides UI building blocks: Layout management, a rich set of pre-built widgets (Lists, Paragraphs, Charts, Tables, Blocks, etc.), styling (colors, text attributes).
    *   Supports various application structuring patterns (Elm, Component, Flux).
    *   Backend Agnostic: Relies on a separate crate (like `crossterm`) for direct terminal interaction.
*   **Website:** [https://ratatui.rs](https://ratatui.rs)

### 2. `crossterm`
*   **Purpose:** A cross-platform (Unix and Windows) terminal manipulation library in pure Rust.
*   **Functionality (Serves as a backend for `ratatui`):**
    *   **Cursor Control:** Moving, showing/hiding, positioning.
    *   **Styled Output:** Setting colors (16 base, 256 ANSI, RGB) and text attributes (bold, italic, etc.).
    *   **Terminal Management:** Clearing screen areas, scrolling, setting size/title, alternate screen mode, raw mode.
    *   **Input Handling:** Reading keyboard events (with modifiers) and mouse events.
    *   **Event System:** Can provide input events as a futures `Stream`.
*   **GitHub:** [https://github.com/crossterm-rs/crossterm](https://github.com/crossterm-rs/crossterm)

**How they work together:**
`ratatui` is used to define the TUI's structure, layout, and the widgets to be displayed. When `ratatui` needs to render the UI or capture input, it uses `crossterm`'s capabilities to directly manipulate the terminal (e.g., print styled text, move cursor, listen for events).

## TUI Functional Requirements (from Skyscope Sentinel Directive):

*   Live, high-level feed of agents' real-time thinking, current strategy, and actions.
*   Minimalist system process display (max 2 lines, showing underlying system processes like Git, Python, Docker, crossterm).
*   Animated micro-task progress bar:
    *   Progress linked to micro-task completion from an agent's objective.
    *   Visuals: Nine small, futuristic-style squares animation to the right of the bar.
    *   Real-time percentage display (e.g., `[██████----] futuristic_squares 60%`).

## Potential High-Level Rust TUI Project Structure:

A separate Rust project, let's call it `skyscope_sentinel_tui`:

```
skyscope_sentinel_tui/
├── Cargo.toml      // Defines Rust project, dependencies (ratatui, crossterm, communication crates)
├── src/
│   ├── main.rs     // Main entry point: initializes TUI, sets up terminal, runs main event loop.
│   ├── app.rs      // Holds the application state (data to be displayed), handles application logic.
│   ├── ui.rs       // Contains functions to draw the UI using ratatui based on `app.rs` state.
│   ├── event.rs    // Manages terminal input events (from crossterm) and custom application events.
│   ├── components/ // Optional directory for custom, reusable TUI components.
│   │   ├── progress_bar.rs // Specific logic for the animated progress bar & squares.
│   │   └── agent_display.rs // Logic for rendering an individual agent's status feed.
│   └── comms.rs    // Module for handling communication with the Python backend.
└── README.md       // README for the TUI project.
```

### Key Implementation Aspects:

*   **`main.rs` / `app.rs` Core Loop:**
    1.  Initialize terminal (raw mode, alternate screen via `crossterm`).
    2.  Loop:
        *   Poll for terminal input events (`crossterm`).
        *   Poll for data updates from the Python backend (via `comms.rs`).
        *   Update application state (`app.rs`) based on events and data.
        *   Call `ui.rs` functions to redraw the entire UI in a buffer.
        *   Render the buffer to the terminal using `ratatui`'s draw mechanism (which uses `crossterm`).
    3.  Restore terminal on exit.

*   **State Management (`app.rs`):**
    *   Store current text for each agent's "thinking/strategy/action" feed.
    *   Store the list of "system processes" being used by the backend.
    *   Store current progress percentage for the micro-task progress bar.
    *   Store the current animation state for the "nine futuristic squares".

*   **Drawing Logic (`ui.rs`):**
    *   Use `ratatui::layout` (e.g., `Layout`, `Constraint`, `Rect`) to divide the terminal screen.
    *   Use `ratatui::widgets::Paragraph` for displaying agent feeds and system processes.
    *   The animated progress bar and squares would likely be a custom widget. This might involve:
        *   Calculating character representations for the bar and squares.
        *   Using `ratatui::buffer::Buffer` for direct character manipulation if needed for fine-grained animation, or by rapidly redrawing a `Paragraph` or `Block` with updated text/styling.
        *   Animation would be driven by periodic updates to the `app.rs` state (e.g., via a timer event or when new progress data arrives).

*   **Communication (`comms.rs`):**
    *   This module needs to establish a connection with the Python AI OS Enhancer backend.
    *   Potential methods:
        *   **HTTP/REST API or WebSockets:** Python backend runs a lightweight server (e.g., using Flask, FastAPI, or `websockets` library). Rust TUI uses a crate like `reqwest` (for HTTP) or `tokio-tungstenite` (for WebSockets). This is a common and flexible approach.
        *   **gRPC:** For more structured RPC, but adds complexity.
        *   **ZeroMQ or NNG (Nanomsg Next Gen):** Message queueing systems, good for IPC.
        *   **Standard I/O Pipes:** If the TUI launches the Python backend as a child process (or vice-versa), but can be more complex to manage robustly.
    *   The data format would likely be JSON for status updates.

### Notes on User's Specific Requirements:
*   **Animated Squares & Progress Bar:** This will require careful state management and timely redraws. The "futuristic squares" animation might involve cycling through different Unicode characters or block styles.
*   **Minimalist System Process Display:** This should be straightforward using a `Paragraph` widget, updated with data from the backend.
*   **Real-time Feed:** Depends on the efficiency of the communication channel and the redraw rate of the TUI.

This conceptual outline provides a starting point for designing the Rust TUI component. Actual implementation will require significant Rust development effort.
```
