# ai_os_enhancer/main.py

import os
import sys
import logging # For main logger level setting if needed from config

# Ensure the package directory is in the Python path if running as a script from project root
# This is often needed if you run `python ai_os_enhancer/main.py` from the directory containing `ai_os_enhancer`
if __package__ is None and not hasattr(sys, 'frozen'):
    # direct execution (not via -m)
    # Resolve the path to the 'ai_os_enhancer' directory itself
    current_file_path = os.path.abspath(__file__)
    ai_os_enhancer_dir = os.path.dirname(current_file_path)

    # The parent of 'ai_os_enhancer' dir should be in sys.path for `from ai_os_enhancer import ...` to work
    project_root_dir = os.path.dirname(ai_os_enhancer_dir)

    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)

# Now that sys.path is potentially adjusted, try importing package components
from ai_os_enhancer import orchestrator
from ai_os_enhancer import logger_setup
from ai_os_enhancer import config

def main():
    """
    Main function to initialize and run the AI OS Enhancer.
    """
    # Setup basic logging as the first step.
    # The actual logging level for handlers within setup_logger might be further
    # controlled by logger_setup.py itself (e.g. using config.LOG_LEVEL)
    # For setup_basic_logging, it's usually a simple console logger.
    # Let's assume config.LOG_LEVEL could define the overall default level.
    log_level_from_config = getattr(config, 'GLOBAL_LOG_LEVEL', logging.INFO)
    logger_setup.setup_basic_logging(level=log_level_from_config)

    # Get a logger for main.py after basic setup.
    # Note: logger_setup.setup_logger() creates more sophisticated loggers (file+console)
    # and is used by other modules. setup_basic_logging() is for very early messages.
    # If Orchestrator's run() re-initializes with setup_logger, that will take precedence for module loggers.
    main_logger = logging.getLogger("AI_OS_Enhancer_Main") # Using a distinct name
    main_logger.info("Application starting from main.py...")
    main_logger.info(f"Initiated with log level: {logging.getLevelName(log_level_from_config)}")
    main_logger.info(f"Using configuration: OLLAMA_API_ENDPOINT={config.OLLAMA_API_ENDPOINT}, DEFAULT_MODEL={config.DEFAULT_MODEL}")
    main_logger.info(f"Full project root determined by config.py: {config.PROJECT_ROOT}")


    try:
        app_orchestrator = orchestrator.Orchestrator()
        app_orchestrator.run() # Orchestrator's run method will likely re-init its own logger
    except SystemExit as e: # Catch sys.exit to log it but allow exit
        main_logger.info(f"Application exited with code {e.code}.")
        raise # Re-raise to actually exit
    except KeyboardInterrupt:
        main_logger.info("Application interrupted by user (Ctrl+C). Shutting down.")
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        main_logger.critical(f"A critical unhandled exception reached main.py: {e}", exc_info=True)
        sys.exit(1) # Exit with a general error code
    finally:
        main_logger.info("Application main.py finished.")

if __name__ == '__main__':
    main()
