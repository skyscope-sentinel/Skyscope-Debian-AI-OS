# ai_os_enhancer/logger_setup.py
import logging
import os
import sys # Import sys for setup_basic_logging
# Assuming config.py is in the same directory or ai_os_enhancer is in PYTHONPATH
# For relative imports to work when running this script directly for testing,
# we might need to adjust the path or run as a module.
# from .config import LOG_FILE_PATH, PROJECT_ROOT
# For now, to make it runnable directly and importable, let's use a placeholder
# and assume config.py sets up paths correctly when imported by other modules.

# A robust way to get config if this script is run:
try:
    from .config import LOG_FILE_PATH, PROJECT_ROOT
except ImportError:
    # This fallback allows the script to be run directly for testing,
    # assuming it's in the ai_os_enhancer directory.
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from config import LOG_FILE_PATH, PROJECT_ROOT
    except ImportError:
        # Fallback if config.py is not found (e.g. during initial setup or testing in isolation)
        print("Warning: config.py not found. Using default log path.")
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "db", "logs", "ai_os_enhancer_fallback.log")


def setup_logger(name='ai_os_enhancer_logger', log_level=logging.INFO, log_file=None):
    """Sets up a logger with file and console handlers."""

    if log_file is None:
        log_file = LOG_FILE_PATH

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print(f"Error creating log directory {log_dir}: {e}")
            # Fallback to a temporary directory if creation fails
            import tempfile
            log_dir = tempfile.gettempdir()
            log_file = os.path.join(log_dir, os.path.basename(log_file))
            print(f"Logging to temporary file: {log_file}")


    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers if already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    try:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except IOError as e:
        print(f"Error setting up file handler for logger: {e}")
        # Optionally, log to console only if file handler fails
        # For now, we'll just print the error and continue with console handler

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Simpler for console
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    if not logger.handlers:
        # If no handlers could be added (e.g. file system error and console also failed, though unlikely for console)
        # Fallback to basicConfig to ensure logs are output somewhere
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.warning("Logger setup failed for file/console, using basicConfig.")


    return logger

def setup_basic_logging(level=logging.INFO):
    """
    Configures a basic logger to print to stdout.
    Useful for early log messages before more complex setup.
    """
    # Basic configuration, ensuring it's idempotent somewhat by checking root handlers
    if not logging.getLogger().handlers: # Check if root logger has no handlers
        logging.basicConfig(level=level,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])
        logging.getLogger().info(f"Basic logging configured with level {logging.getLevelName(level)}.")
    else:
        logging.getLogger().setLevel(level) # If handlers exist, just try to set level
        # logging.getLogger().debug(f"Basic logging already configured. Set root logger level to {logging.getLevelName(level)}.")


# Example usage:
if __name__ == '__main__':
    # This part is for testing the logger setup independently.
    import sys # Needed for StreamHandler(sys.stdout) in setup_basic_logging if called here

    setup_basic_logging(level=logging.DEBUG) # Test the new basic logger
    logging.getLogger("BasicTest").info("Testing basic_logging from logger_setup main.")

    print(f"Attempting to log to: {LOG_FILE_PATH}")
    # Ensure the directory from config is created for the test
    # config.py should handle this, but for direct testing, we can be explicit.
    log_dir_for_test = os.path.dirname(LOG_FILE_PATH)
    if not os.path.exists(log_dir_for_test):
        print(f"Test: Log directory {log_dir_for_test} does not exist. Creating it.")
        try:
            os.makedirs(log_dir_for_test, exist_ok=True)
        except Exception as e:
            print(f"Test: Failed to create log directory {log_dir_for_test}: {e}")


    # Test default logger
    logger_default = setup_logger()
    logger_default.info("This is an INFO message from the default logger.")

    # Test custom logger
    custom_log_file = os.path.join(os.path.dirname(LOG_FILE_PATH), "custom_test_logger.log")
    print(f"Attempting to log to custom file: {custom_log_file}")

    logger_custom = setup_logger(name='MyCustomTestLogger', log_level=logging.DEBUG, log_file=custom_log_file)
    logger_custom.debug("This is a DEBUG message from MyCustomTestLogger.")
    logger_custom.info("This is an INFO message from MyCustomTestLogger.")
    logger_custom.warning("This is a WARNING message from MyCustomTestLogger.")
    logger_custom.error("This is an ERROR message from MyCustomTestLogger.")
    logger_custom.critical("This is a CRITICAL message from MyCustomTestLogger.")

    print("\nTest logging complete.")
    print(f"Default log file should be at: {LOG_FILE_PATH}")
    print(f"Custom test log file should be at: {custom_log_file}")

    # Test import of config (if it works)
    try:
        from .config import DEFAULT_MODEL # Try to import something from config
        print(f"Successfully imported DEFAULT_MODEL from config: {DEFAULT_MODEL}")
    except ImportError as e:
        print(f"Could not import from .config directly: {e}")
        try:
            import config
            print(f"Successfully imported config using 'import config': {config.DEFAULT_MODEL}")
        except ImportError as ie:
            print(f"Could not import config module: {ie}")
    print(f"Project root (determined by logger_setup): {PROJECT_ROOT}")
