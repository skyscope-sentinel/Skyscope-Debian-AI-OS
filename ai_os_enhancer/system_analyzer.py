# ai_os_enhancer/system_analyzer.py

import subprocess
import os
import pathlib
import logging
import shlex # For safely splitting command strings
import glob # For path globbing

# Relative imports for when this module is part of the package
try:
    from . import config
    from . import logger_setup
except ImportError:
    # Fallback for direct execution (e.g., if __name__ == '__main__')
    # This assumes 'config.py' and 'logger_setup.py' are in the same directory
    # and the script is run from within the 'ai_os_enhancer' directory.
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    import logger_setup

# Initialize logger for this module
# If run directly, logger_setup might need to initialize basic logging if not already done.
if __name__ == '__main__' and not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
    # A simple basicConfig for direct script execution if no handlers are present
    # This is a fallback if logger_setup.setup_logger() or similar hasn't been called by an entry point
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("SystemStateAnalyzer_direct")
    logger.info("Running SystemStateAnalyzer directly, basic logging configured.")
else:
    logger = logger_setup.setup_logger("SystemStateAnalyzer")


# --- SystemStateAnalyzer Module ---

def _execute_command(command_string):
    """
    Helper function to execute a shell command and return its output.
    Returns a tuple: (stdout, stderr, return_code)
    """
    try:
        logger.debug(f"Executing command: {command_string}")
        # For commands constructed with f-strings, ensure variables are controlled or sanitized.
        # shlex.split is good for simple cases. For complex args, pass a list.
        if isinstance(command_string, list):
            cmd_list = command_string
        else:
            cmd_list = shlex.split(command_string)

        process = subprocess.run(cmd_list, capture_output=True, text=True, check=False)

        # Log stdout/stderr only if they are not excessively long
        stdout_log = process.stdout.strip()
        stderr_log = process.stderr.strip()
        if len(stdout_log) > 200: stdout_log = stdout_log[:200] + "... (truncated)"
        if len(stderr_log) > 200: stderr_log = stderr_log[:200] + "... (truncated)"

        logger.debug(f"Command stdout: {stdout_log}")
        if process.stderr:
            logger.debug(f"Command stderr: {stderr_log}")
        logger.debug(f"Command return code: {process.returncode}")
        return process.stdout.strip(), process.stderr.strip(), process.returncode
    except FileNotFoundError:
        cmd_name = command_string.split()[0] if isinstance(command_string, str) else command_string[0]
        logger.error(f"Command not found: {cmd_name}")
        return "", f"Command not found: {cmd_name}", -1 # Consistent error code
    except Exception as e:
        logger.error(f"Error executing command '{command_string}': {e}")
        return "", str(e), -1 # Consistent error code

def get_debian_version():
    """
    Returns: String (e.g., "Debian GNU/Linux 13 (Trixie)") or None on error.
    """
    stdout, stderr, return_code = _execute_command("lsb_release -ds")
    if return_code == 0 and stdout:
        return stdout
    else:
        logger.debug("lsb_release -ds failed or returned empty. Trying lsb_release -a.")
        stdout_all, _, rc_all = _execute_command("lsb_release -a")
        if rc_all == 0:
            for line in stdout_all.splitlines():
                if line.startswith("Description:"):
                    return line.split(":", 1)[1].strip()
        logger.warning("Could not determine Debian version via lsb_release. Reading /etc/os-release.")
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
            logger.warning("PRETTY_NAME not found in /etc/os-release.")
        except FileNotFoundError:
            logger.error("/etc/os-release not found.")
        except Exception as e:
            logger.error(f"Error reading /etc/os-release: {e}")
        return None


def get_installed_packages():
    """
    Returns: List of strings (package names) or empty list on error.
    """
    stdout, stderr, return_code = _execute_command("dpkg-query -W -f='${Package}\n'")
    if return_code == 0 and stdout:
        return [pkg for pkg in stdout.splitlines() if pkg] # Filter out empty lines
    else:
        logger.warning("dpkg-query failed. Attempting 'apt list --installed'.")
        stdout_apt, stderr_apt, rc_apt = _execute_command("apt list --installed")
        if rc_apt == 0 and stdout_apt:
            packages = []
            # Skip the header line like "Listing..."
            lines = stdout_apt.splitlines()
            if lines and lines[0].startswith("Listing..."):
                lines = lines[1:]

            for line in lines:
                if not line.strip(): # Skip empty lines
                    continue
                # Example line: acl/stable,now 2.3.1-3 amd64 [installed]
                # Or: accountsservice/now 22.08.8-1 amd64 [installed,automatic]
                parts = line.split('/')
                if len(parts) > 0:
                    packages.append(parts[0])
            return packages
        logger.error("Failed to get installed packages using both dpkg-query and apt list.")
        return []

def get_service_status(service_name):
    """
    Returns: String (e.g., "active", "inactive", "failed", "unknown") or None on error for bad name.
    """
    # Basic sanitization for service_name. More robust validation might be needed
    # if service_name can come from untrusted sources.
    if not service_name or not all(c.isalnum() or c in ['-', '_', '.'] for c in service_name):
        logger.error(f"Invalid service name format: {service_name}")
        return None # Or raise ValueError

    stdout, stderr, return_code = _execute_command(f"systemctl is-active {service_name}")

    # systemctl is-active:
    # - If active: stdout is "active", RC is 0.
    # - If inactive: stdout is "inactive", RC is non-zero (usually 3).
    # - If failed: stdout is "failed", RC is non-zero (usually 3).
    # - If service not found: stdout is empty (or "unknown"), stderr may have "Unit SERVICE_NAME not found.", RC non-zero.
    # - If status is "activating", "deactivating": stdout is that status, RC 0.

    if stdout: # stdout often contains the direct status
        return stdout.strip()
    else:
        # If stdout is empty, it often means the service is not found or there was an error.
        # stderr might contain more info.
        logger.warning(f"Service '{service_name}': 'is-active' returned empty stdout. RC: {return_code}. Stderr: {stderr}")
        # We could try 'systemctl status' for more details but it's heavier.
        # For now, if 'is-active' fails to give a clear status on stdout, we mark as 'unknown'.
        if "not-found" in stderr.lower() or "no such file or directory" in stderr.lower(): # Heuristic
            return "not-found"
        return "unknown"


def read_file_content(file_path_str):
    """
    Returns: String (content of the file) or None on error.
    """
    try:
        file_path = pathlib.Path(file_path_str)
        if not file_path.is_file(): # Checks if it's a regular file and exists
            logger.warning(f"File not found or not a regular file: {file_path}")
            return None
        return file_path.read_text(encoding='utf-8') # Specify encoding
    except FileNotFoundError: # Should be caught by is_file, but good for robustness
        logger.error(f"File not found: {file_path}")
        return None
    except PermissionError:
        logger.error(f"Permission denied reading file: {file_path}")
        return None
    except Exception as e: # Catch other potential errors like decoding errors
        logger.error(f"Failed to read file {file_path}: {e}")
        return None

def get_file_permissions(file_path_str):
    """
    Returns: String (e.g., "0755") or None on error.
    """
    try:
        file_path = pathlib.Path(file_path_str)
        if not file_path.exists(): # Check general existence (file, dir, link)
            logger.warning(f"Path does not exist: {file_path}")
            return None
        mode = file_path.stat().st_mode
        return oct(mode & 0o7777)[-4:] # Ensure we only get permission bits, formatted as 4 octal digits
    except FileNotFoundError: # Should be caught by exists()
        logger.error(f"File not found for permissions check: {file_path}")
        return None
    except PermissionError:
        logger.error(f"Permission denied for stat on: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error getting permissions for {file_path}: {e}")
        return None

def get_file_owner(file_path_str):
    """
    Returns: String (e.g., "root:root") or None on error.
    """
    try:
        file_path = pathlib.Path(file_path_str)
        if not file_path.exists():
            logger.warning(f"Path does not exist for owner check: {file_path}")
            return None

        stat_info = file_path.stat() # Follows symlinks by default
        uid = stat_info.st_uid
        gid = stat_info.st_gid

        import pwd  # POSIX specific, should be fine on Debian
        import grp  # POSIX specific

        try:
            user_name = pwd.getpwuid(uid).pw_name
        except KeyError: # If UID is not in the system's user database
            logger.warning(f"Could not find username for UID {uid} (path: {file_path})")
            user_name = str(uid)

        try:
            group_name = grp.getgrgid(gid).gr_name
        except KeyError: # If GID is not in the system's group database
            logger.warning(f"Could not find group name for GID {gid} (path: {file_path})")
            group_name = str(gid)

        return f"{user_name}:{group_name}"
    except FileNotFoundError: # Should be caught by exists()
        logger.error(f"File not found for owner check: {file_path}")
        return None
    except PermissionError:
        logger.error(f"Permission denied for stat (owner check): {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error getting owner for {file_path}: {e}")
        return None

def list_key_config_and_script_areas():
    """
    Returns: List of dictionaries {name, type, paths_to_check, heuristic}
    Paths are expanded from glob patterns.
    """
    # Define base configuration areas with potential glob patterns
    base_configs_definitions = [
        {"name": "NetworkInterfaces", "type": "config", "paths_patterns": ["/etc/network/interfaces", "/etc/netplan/*.yaml"], "heuristic": "Network performance, security"},
        {"name": "SystemControl", "type": "config", "paths_patterns": ["/etc/sysctl.conf", "/etc/sysctl.d/*.conf"], "heuristic": "Kernel parameters, performance, security"},
        {"name": "FirewallRules", "type": "config", "paths_patterns": ["/etc/ufw/user.rules", "/etc/iptables/rules.v4", "/etc/nftables.conf"], "heuristic": "Security, network access"},
        {"name": "ScheduledTasks", "type": "config", "paths_patterns": ["/etc/crontab", "/var/spool/cron/crontabs/*", "/etc/cron.d/*", "/etc/cron.hourly/*", "/etc/cron.daily/*", "/etc/cron.weekly/*", "/etc/cron.monthly/*"], "heuristic": "Automation, system maintenance"},
        {"name": "BootLoader", "type": "config", "paths_patterns": ["/boot/grub/grub.cfg", "/etc/default/grub"], "heuristic": "Boot process, kernel options"},
        {"name": "SSHConfig", "type": "config", "paths_patterns": ["/etc/ssh/sshd_config"], "heuristic": "Remote access security"},
        {"name": "PAM", "type": "config", "paths_patterns": ["/etc/pam.d/*"], "heuristic": "Authentication modules"},
        {"name": "SystemdServicesUser", "type": "config", "paths_patterns": ["/etc/systemd/system/*", "/etc/systemd/user/*"], "heuristic": "Service management (user overrides)"},
        {"name": "SystemdServicesSystem", "type": "config", "paths_patterns": ["/usr/lib/systemd/system/*", "/usr/lib/systemd/user/*" ], "heuristic": "Service management (default units)"},
        {"name": "LogRotation", "type": "config", "paths_patterns": ["/etc/logrotate.conf", "/etc/logrotate.d/*"], "heuristic": "Log management"},
        {"name": "UserManagement", "type": "config", "paths_patterns": ["/etc/passwd", "/etc/shadow", "/etc/group", "/etc/sudoers", "/etc/sudoers.d/*"], "heuristic": "User accounts and privileges"},
    ]

    expanded_areas = []
    for area_def in base_configs_definitions:
        current_paths = []
        for pattern in area_def["paths_patterns"]:
            try:
                # glob.glob handles non-glob patterns gracefully (returns them as is if they exist)
                # Using iglob for potentially large directories, though for these system paths, it's minor.
                found_paths = list(glob.iglob(pattern, recursive=False)) # Set recursive based on need
                if found_paths:
                    current_paths.extend(found_paths)
                elif "*" not in pattern and "?" not in pattern and "[" not in pattern:
                    # If it's a specific path (not a glob) and not found, still add it.
                    # The AI might want to know about its absence or create it.
                    current_paths.append(pattern)
                    logger.debug(f"Specific path {pattern} not found, but added to check list for area {area_def['name']}.")
                else:
                    logger.debug(f"Glob pattern {pattern} in area {area_def['name']} yielded no results.")
            except Exception as e:
                logger.error(f"Error during globbing pattern {pattern} for area {area_def['name']}: {e}")

        if current_paths:
            expanded_areas.append({
                "name": area_def["name"],
                "type": area_def["type"],
                "paths": sorted(list(set(current_paths))), # Unique, sorted paths
                "heuristic": area_def["heuristic"]
            })

    # Add user-defined monitored scripts
    user_scripts_definitions = []
    if hasattr(config, 'MONITORED_SCRIPTS_PATHS') and config.MONITORED_SCRIPTS_PATHS:
        for script_path_str in config.MONITORED_SCRIPTS_PATHS:
            script_path = pathlib.Path(script_path_str)
            # We check existence here. If a script is meant to be analyzed but doesn't exist,
            # it's a configuration issue or the AI needs to note its absence.
            if script_path.exists() and script_path.is_file():
                user_scripts_definitions.append({
                    "name": script_path.name,
                    "type": "script",
                    "paths": [str(script_path)], # Path must be a list of strings
                    "heuristic": "User-defined automation, custom system tasks, potential for AI refactoring"
                })
            else:
                logger.warning(f"Monitored script path from config not found or not a file: {script_path_str}. It will not be added to key areas.")

    return expanded_areas + user_scripts_definitions


if __name__ == '__main__':
    # Setup logger for direct execution (if not already configured by the import fallbacks)
    # This ensures that if logger_setup.setup_logger was not called due to import choices,
    # we still get output.
    if not logger.hasHandlers(): # Check if our module logger specifically has handlers
        logger_setup.setup_logger("SystemStateAnalyzer_main", level=logging.DEBUG) # Re-get or setup
        logger.info("Logger re-initialized for __main__ block with DEBUG level.")
    else:
        logger.setLevel(logging.DEBUG) # Ensure level is appropriate for testing
        logger.info("Logger already initialized. Set level to DEBUG for __main__ block.")

    logger.info("--- SystemStateAnalyzer Test ---")

    debian_version = get_debian_version()
    logger.info(f"Debian Version: {debian_version if debian_version else 'Not found or error'}")

    installed_packages = get_installed_packages()
    if installed_packages:
        logger.info(f"Installed Packages (first 5): {installed_packages[:5]}")
        logger.info(f"Total Installed Packages: {len(installed_packages)}")
    else:
        logger.warning("No installed packages found or error retrieving them.")

    common_services = ["cron", "ssh", "systemd-journald", "nonexistentservice"]
    for service in common_services:
        status = get_service_status(service)
        logger.info(f"Service '{service}' status: {status if status else 'Error or invalid name'}")

    # Dummy file for testing file operations (within project, should be writable)
    # Ensure PROJECT_ROOT is correctly determined
    project_root_for_test = config.PROJECT_ROOT if hasattr(config, 'PROJECT_ROOT') else pathlib.Path(__file__).parent.resolve()
    dummy_file_path = project_root_for_test / "dummy_test_file.txt"
    logger.info(f"Using project root for dummy file: {project_root_for_test}")
    logger.info(f"Dummy file will be at: {dummy_file_path}")

    try:
        # Ensure parent directory for dummy_file_path exists if it's nested
        dummy_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dummy_file_path, "w", encoding="utf-8") as f:
            f.write("Hello, AI OS Enhancer!\nThis is a test file.\nLine 3.")
        logger.info(f"Created dummy file: {dummy_file_path}")

        content = read_file_content(str(dummy_file_path))
        if content is not None: # Check for None explicitly
            # Prepare the content for logging to avoid backslash issues in f-string
            log_content = content[:60].replace('\n', '\\n') # Replace newline with literal \n for logging
            logger.info(f"Dummy file content (first 60 chars): '{log_content}'")
        else:
            logger.error(f"Failed to read dummy file content.")

        permissions = get_file_permissions(str(dummy_file_path))
        logger.info(f"Dummy file permissions: {permissions if permissions else 'Error'}")

        owner = get_file_owner(str(dummy_file_path))
        logger.info(f"Dummy file owner: {owner if owner else 'Error'}")

    except Exception as e:
        logger.error(f"Error in dummy file operations test: {e}", exc_info=True)
    finally:
        if dummy_file_path.exists():
            try:
                os.remove(dummy_file_path)
                logger.info(f"Removed dummy file: {dummy_file_path}")
            except Exception as e:
                logger.error(f"Error removing dummy file {dummy_file_path}: {e}")

    key_areas = list_key_config_and_script_areas()
    logger.info(f"Found {len(key_areas)} key config/script areas.")
    for i, area in enumerate(key_areas):
        if i < 5: # Log first 5 areas for a bit more detail
             logger.info(f"Area: {area['name']}, Type: {area['type']}, Paths (up to 2): {area['paths'][:2]}..., Heuristic: {area['heuristic']}")
        elif i == 5:
             logger.info("... and potentially more areas (logging limited for brevity).")
    if not key_areas:
        logger.info("No key config/script areas were identified (check glob patterns and system paths).")


    logger.info(f"Monitored script paths from config: {config.MONITORED_SCRIPTS_PATHS if hasattr(config, 'MONITORED_SCRIPTS_PATHS') else 'Not configured'}")
    # Example: Create a dummy monitored script for testing this part
    sample_script_dir = project_root_for_test / "sample_scripts_for_test"
    sample_script_path = None
    if hasattr(config, 'MONITORED_SCRIPTS_PATHS') and isinstance(config.MONITORED_SCRIPTS_PATHS, list):
        # For testing, let's assume the first monitored script path might be this dummy one
        # Or, more reliably, add a specific test script path to config for testing.
        # For now, let's create one if MONITORED_SCRIPTS_PATHS is empty or not set for testing.
        if not config.MONITORED_SCRIPTS_PATHS:
            logger.info("MONITORED_SCRIPTS_PATHS is empty in config. Creating a dummy one for test.")
            sample_script_dir.mkdir(exist_ok=True)
            sample_script_path = sample_script_dir / "test_monitor_script.sh"
            with open(sample_script_path, "w") as sf:
                sf.write("#!/bin/bash\necho 'Monitored script test output'")
            config.MONITORED_SCRIPTS_PATHS.append(str(sample_script_path)) # Modify for this run
            logger.info(f"Added dummy monitored script: {sample_script_path}")
            # Re-run list_key_config_and_script_areas to include it
            key_areas = list_key_config_and_script_areas()


    scripts_in_key_areas = [area for area in key_areas if area['type'] == 'script']
    if scripts_in_key_areas:
        logger.info(f"Monitored scripts found in key areas: {[s['name'] for s in scripts_in_key_areas]}")
        for script_area in scripts_in_key_areas:
             logger.debug(f"  Script Area Details: {script_area}")
    else:
        logger.info("No monitored scripts found or configured in key areas for this test run.")

    # Cleanup dummy script if created
    if sample_script_path and sample_script_path.exists():
        os.remove(sample_script_path)
        logger.info(f"Removed dummy monitored script: {sample_script_path}")
        # Remove directory if empty
        try:
            sample_script_dir.rmdir() # Fails if not empty
            logger.info(f"Removed dummy script directory: {sample_script_dir}")
        except OSError:
            logger.debug(f"Dummy script directory {sample_script_dir} not empty or other error on rmdir.")


    logger.info("--- End of SystemStateAnalyzer Test ---")
