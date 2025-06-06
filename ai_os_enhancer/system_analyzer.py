# ai_os_enhancer/system_analyzer.py

import subprocess
import os
import pathlib
import logging
import shlex # For safely splitting command strings
import glob # For path globbing
import requests # For check_ollama_service_availability
import shutil # For get_system_snapshot (disk_usage)
from urllib.parse import urlparse # For check_ollama_service_availability

# Relative imports for when this module is part of the package
try:
    from . import config
    from . import logger_setup
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Add parent dir to path
    import config
    import logger_setup

# Initialize logger for this module
if __name__ == '__main__' and not logging.getLogger(logger_setup.LOGGER_NAME if hasattr(logger_setup, 'LOGGER_NAME') else 'ai_os_enhancer_logger').hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("SystemStateAnalyzer_direct_main")
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
        if isinstance(command_string, list):
            cmd_list = command_string
        else:
            cmd_list = shlex.split(command_string)

        process = subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=30) # Added timeout

        stdout_log = process.stdout.strip()
        stderr_log = process.stderr.strip()
        if len(stdout_log) > 200: stdout_log = stdout_log[:200] + "... (truncated)"
        if len(stderr_log) > 200: stderr_log = stderr_log[:200] + "... (truncated)"

        logger.debug(f"Command stdout: {stdout_log}")
        if process.stderr:
            logger.debug(f"Command stderr: {stderr_log}") # Changed to debug, as stderr isn't always an error
        logger.debug(f"Command return code: {process.returncode}")
        return process.stdout.strip(), process.stderr.strip(), process.returncode
    except FileNotFoundError:
        cmd_name = command_string.split()[0] if isinstance(command_string, str) else command_string[0]
        logger.error(f"Command not found: {cmd_name}")
        return "", f"Command not found: {cmd_name}", 127 # Common exit code for command not found
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {command_string}")
        return "", "Command timed out", -9 # Common for timeout
    except Exception as e:
        logger.error(f"Error executing command '{command_string}': {e}", exc_info=True)
        return "", str(e), 1 # Generic error code

def get_debian_version():
    """
    Returns: String (e.g., "Debian GNU/Linux 13 (Trixie)") or None on error.
    """
    stdout, stderr, return_code = _execute_command("lsb_release -ds")
    if return_code == 0 and stdout:
        return stdout
    else:
        logger.debug("lsb_release -ds failed or returned empty. Trying /etc/os-release.")
        try:
            content = read_file_content("/etc/os-release")
            if content:
                for line in content.splitlines():
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
            logger.warning("PRETTY_NAME not found in /etc/os-release.")
        except Exception as e: # Catch errors from read_file_content if any
            logger.error(f"Error reading /etc/os-release: {e}")
        return "Unknown Debian-based OS"


def get_installed_packages():
    """
    Returns: List of strings (package names) or empty list on error.
    """
    stdout, stderr, return_code = _execute_command("dpkg-query -W -f='${Package}\n'")
    if return_code == 0 and stdout:
        return [pkg for pkg in stdout.splitlines() if pkg]
    else:
        logger.warning("dpkg-query failed. Attempting 'apt list --installed'. This might be slower.")
        stdout_apt, stderr_apt, rc_apt = _execute_command("apt list --installed")
        if rc_apt == 0 and stdout_apt:
            packages = []
            lines = stdout_apt.splitlines()
            if lines and (lines[0].startswith("Listing...") or lines[0].startswith("Auflistung…")): # Handle different locales
                lines = lines[1:]
            for line in lines:
                if not line.strip(): continue
                parts = line.split('/')
                if len(parts) > 0:
                    packages.append(parts[0])
            return packages
        logger.error("Failed to get installed packages using both dpkg-query and apt list.")
        return []

def get_service_status(service_name: str) -> str | None:
    """
    Returns: String (e.g., "active", "inactive", "failed", "not-found", "unknown") or None on error for bad name.
    """
    if not service_name or not all(c.isalnum() or c in ['-', '_', '.', '@'] for c in service_name): # Allow '@' for template instances
        logger.error(f"Invalid service name format: {service_name}")
        return None

    stdout, stderr, return_code = _execute_command(f"systemctl is-active {shlex.quote(service_name)}")

    if stdout:
        return stdout.strip()
    else:
        logger.debug(f"Service '{service_name}': 'is-active' returned empty stdout. RC: {return_code}. Stderr: {stderr}")
        # Try 'systemctl is-enabled' as a fallback for non-active but existing services
        stdout_enabled, _, _ = _execute_command(f"systemctl is-enabled {shlex.quote(service_name)}")
        if stdout_enabled == "enabled" or stdout_enabled == "disabled":
             return "inactive" # If it's enabled/disabled but not active, treat as inactive for simplicity

        if "not-found" in stderr.lower() or "no such file or directory" in stderr.lower() or "Failed to get unit file state" in stderr:
            return "not-found"
        return "unknown"

def is_service_active(service_name: str) -> bool:
    """
    Checks if a systemd service is currently active.
    Args:
        service_name (str): The name of the service (e.g., "cron", "ssh").
    Returns:
        bool: True if the service status is "active", False otherwise.
    """
    status = get_service_status(service_name)
    if status is None:
        return False
    return status.strip().lower() == "active"

def read_file_content(file_path_str: str) -> str | None:
    """
    Returns: String (content of the file) or None on error.
    """
    try:
        file_path = pathlib.Path(file_path_str)
        if not file_path.is_file():
            logger.debug(f"File not found or not a regular file: {file_path}")
            return None
        return file_path.read_text(encoding='utf-8')
    except PermissionError:
        logger.error(f"Permission denied reading file: {file_path_str}")
        return None
    except Exception as e:
        logger.error(f"Failed to read file {file_path_str}: {e}", exc_info=True)
        return None

def get_file_permissions(file_path_str: str) -> str | None:
    """
    Returns: String (e.g., "0755") or None on error.
    """
    try:
        file_path = pathlib.Path(file_path_str)
        if not file_path.exists():
            logger.debug(f"Path does not exist for permission check: {file_path_str}")
            return None
        mode = file_path.stat().st_mode
        return oct(mode & 0o7777)[-4:]
    except PermissionError:
        logger.error(f"Permission denied for stat on: {file_path_str}")
        return None
    except Exception as e:
        logger.error(f"Error getting permissions for {file_path_str}: {e}", exc_info=True)
        return None

def get_file_owner(file_path_str: str) -> str | None:
    """
    Returns: String (e.g., "root:root") or None on error.
    """
    try:
        file_path = pathlib.Path(file_path_str)
        if not file_path.exists():
            logger.debug(f"Path does not exist for owner check: {file_path_str}")
            return None

        stat_info = file_path.stat()
        uid = stat_info.st_uid
        gid = stat_info.st_gid

        import pwd
        import grp

        try:
            user_name = pwd.getpwuid(uid).pw_name
        except KeyError:
            logger.debug(f"Could not find username for UID {uid} (path: {file_path_str})")
            user_name = str(uid)

        try:
            group_name = grp.getgrgid(gid).gr_name
        except KeyError:
            logger.debug(f"Could not find group name for GID {gid} (path: {file_path_str})")
            group_name = str(gid)

        return f"{user_name}:{group_name}"
    except PermissionError:
        logger.error(f"Permission denied for stat (owner check): {file_path_str}")
        return None
    except Exception as e:
        logger.error(f"Error getting owner for {file_path_str}: {e}", exc_info=True)
        return None

def list_key_config_and_script_areas():
    """
    Returns: List of dictionaries {name, type, paths_to_check, heuristic}
    Paths are expanded from glob patterns.
    """
    base_configs_definitions = [
        {"name": "NetworkInterfaces", "type": "config", "paths_patterns": ["/etc/network/interfaces", "/etc/netplan/*.yaml"], "heuristic": "Network performance, security"},
        {"name": "SystemControl", "type": "config", "paths_patterns": ["/etc/sysctl.conf", "/etc/sysctl.d/*.conf"], "heuristic": "Kernel parameters, performance, security"},
        {"name": "FirewallRules", "type": "config", "paths_patterns": ["/etc/ufw/user.rules", "/etc/iptables/rules.v4", "/etc/nftables.conf"], "heuristic": "Security, network access"},
        {"name": "ScheduledTasks", "type": "config", "paths_patterns": ["/etc/crontab", "/var/spool/cron/crontabs/*", "/etc/cron.d/*", "/etc/cron.hourly/*", "/etc/cron.daily/*", "/etc/cron.weekly/*", "/etc/cron.monthly/*"], "heuristic": "Automation, system maintenance"},
        {"name": "BootLoader", "type": "config", "paths_patterns": ["/boot/grub/grub.cfg", "/etc/default/grub"], "heuristic": "Boot process, kernel options"},
        {"name": "SSHConfig", "type": "config", "paths_patterns": ["/etc/ssh/sshd_config", "/etc/ssh/sshd_config.d/*.conf"], "heuristic": "Remote access security"},
        {"name": "PAM", "type": "config", "paths_patterns": ["/etc/pam.d/*"], "heuristic": "Authentication modules"},
        {"name": "SystemdServicesUser", "type": "config", "paths_patterns": ["/etc/systemd/system/*.service", "/etc/systemd/user/*.service"], "heuristic": "Service management (user overrides)"},
        {"name": "SystemdServicesSystem", "type": "config", "paths_patterns": ["/usr/lib/systemd/system/*.service", "/lib/systemd/system/*.service"], "heuristic": "Service management (default units)"},
        {"name": "LogRotation", "type": "config", "paths_patterns": ["/etc/logrotate.conf", "/etc/logrotate.d/*"], "heuristic": "Log management"},
        {"name": "UserManagement", "type": "config", "paths_patterns": ["/etc/passwd", "/etc/shadow", "/etc/group", "/etc/sudoers", "/etc/sudoers.d/*"], "heuristic": "User accounts and privileges"},
    ]

    expanded_areas = []
    for area_def in base_configs_definitions:
        current_paths = []
        for pattern in area_def["paths_patterns"]:
            try:
                found_paths = list(glob.iglob(pattern, recursive=False))
                if found_paths:
                    current_paths.extend(found_paths)
                elif not any(c in pattern for c in "*?[]"): # Not a glob pattern
                    # AI might want to know about its absence or create it.
                    # Only add if explicitly configured and not found by glob (which means it doesn't exist)
                    # current_paths.append(pattern)
                    logger.debug(f"Specific path {pattern} not found for area {area_def['name']}. Not adding to list.")
                else:
                    logger.debug(f"Glob pattern {pattern} in area {area_def['name']} yielded no results.")
            except Exception as e:
                logger.error(f"Error during globbing pattern {pattern} for area {area_def['name']}: {e}", exc_info=True)

        if current_paths:
            expanded_areas.append({
                "name": area_def["name"],
                "type": area_def["type"],
                "paths": sorted(list(set(current_paths))),
                "heuristic": area_def["heuristic"]
            })

    user_scripts_definitions = []
    if hasattr(config, 'MONITORED_SCRIPTS_PATHS') and config.MONITORED_SCRIPTS_PATHS:
        # Assuming MONITORED_SCRIPTS_PATHS is a list of dicts like [{"path": "/path/to/script.sh", "type": "script"}, ...]
        # or a simple list of paths. Adapting for both.
        for item in config.MONITORED_SCRIPTS_PATHS:
            script_path_str = None
            script_type = "script" # Default type
            if isinstance(item, dict):
                script_path_str = item.get("path")
                script_type = item.get("type", "script")
            elif isinstance(item, str):
                script_path_str = item

            if not script_path_str:
                logger.warning(f"Invalid item in MONITORED_SCRIPTS_PATHS: {item}")
                continue

            script_path = pathlib.Path(script_path_str)
            if script_path.exists() and script_path.is_file():
                user_scripts_definitions.append({
                    "name": script_path.name,
                    "type": script_type, # Use provided type or default
                    "paths": [str(script_path)],
                    "heuristic": "User-defined script/config, potential for AI refactoring or analysis"
                })
            else:
                logger.warning(f"Monitored path from config not found or not a file: {script_path_str}. It will not be added.")

    return expanded_areas + user_scripts_definitions

def check_ollama_service_availability(api_endpoint: str) -> bool:
    """
    Checks if the Ollama service is available by making a GET request to its base URL.
    Args:
        api_endpoint (str): The full API endpoint URL (e.g., "http://localhost:11434/api/generate").
                            The function will derive the base URL (e.g., "http://localhost:11434/").
    Returns:
        bool: True if the service is up and responds, False otherwise.
    """
    if not api_endpoint:
        logger.error("Ollama API endpoint not provided for availability check.")
        return False
    try:
        parsed_url = urlparse(api_endpoint)
        # For "/api/generate" like endpoints, base is just scheme+netloc. For "/" just use it.
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"

        logger.debug(f"Checking Ollama service availability at base URL: {base_url}")
        response = requests.get(base_url, timeout=5)
        # Ollama's base URL typically returns "Ollama is running" with a 200 OK.
        if response.status_code == 200 and "Ollama is running" in response.text:
            logger.info(f"Ollama service is available at {base_url}. Status: {response.status_code}.")
            return True
        else:
            logger.warning(f"Ollama service at {base_url} responded with status {response.status_code} or unexpected content. Content (first 100 chars): '{response.text[:100]}'")
            return False

    except requests.exceptions.Timeout:
        logger.error(f"Timeout trying to reach Ollama service at derived base URL from '{api_endpoint}'.")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error trying to reach Ollama service at derived base URL from '{api_endpoint}'. Is Ollama running?")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking Ollama service availability (derived base from '{api_endpoint}'): {e}", exc_info=True)
        return False
    except Exception as e_gen:
        logger.error(f"Unexpected error during Ollama availability check for endpoint '{api_endpoint}': {e_gen}", exc_info=True)
        return False

def get_system_snapshot() -> dict:
    """
    Gathers various pieces of system information to create a system snapshot.
    Returns:
        dict: A dictionary containing system metrics. Errors for individual metrics are logged.
    """
    snapshot = {}

    try:
        snapshot["debian_version"] = get_debian_version()
    except Exception as e:
        logger.error(f"Error getting Debian version for snapshot: {e}", exc_info=True)
        snapshot["debian_version"] = "Error"

    try:
        kernel_ver_out, _, rc = _execute_command("uname -r")
        snapshot["kernel_version"] = kernel_ver_out if rc == 0 and kernel_ver_out else "Error executing uname or empty output"
    except Exception as e:
        logger.error(f"Error getting kernel version for snapshot: {e}", exc_info=True)
        snapshot["kernel_version"] = "Error"

    try:
        snapshot["load_avg"] = os.getloadavg()
    except OSError as e: # More specific exception for getloadavg
        logger.error(f"Error getting load average (os.getloadavg not available or failed): {e}", exc_info=True)
        snapshot["load_avg"] = ("Error", "Error", "Error")
    except Exception as e:
        logger.error(f"Unexpected error getting load average: {e}", exc_info=True)
        snapshot["load_avg"] = ("Error", "Error", "Error")


    try:
        mem_info = {}
        meminfo_content = read_file_content("/proc/meminfo")
        if meminfo_content:
            for line in meminfo_content.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    val = parts[1]
                    if key in ["MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapTotal", "SwapFree"]:
                        mem_info[key] = f"{val} {parts[2] if len(parts) > 2 else 'kB'}"

            if all(k in mem_info and mem_info[k].split()[0].isdigit() for k in ["MemTotal", "MemFree", "Buffers", "Cached"]):
                mt = int(mem_info["MemTotal"].split()[0])
                mf = int(mem_info["MemFree"].split()[0])
                mb = int(mem_info["Buffers"].split()[0])
                mc = int(mem_info["Cached"].split()[0])
                mem_info["UsedApplication"] = f"{mt - mf - mb - mc} {mem_info['MemTotal'].split()[1]}"
            else:
                mem_info["UsedApplication"] = "Error calculating: one or more base values missing/non-numeric"
            snapshot["memory_usage"] = mem_info
        else:
            snapshot["memory_usage"] = {"error": "Could not read /proc/meminfo"}
    except Exception as e:
        logger.error(f"Error parsing /proc/meminfo for snapshot: {e}", exc_info=True)
        snapshot["memory_usage"] = {"error": str(e)}

    try:
        disk = shutil.disk_usage('/')
        snapshot["disk_usage_root"] = {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "total_readable": f"{disk.total / (1024**3):.2f} GB", # More human-readable
            "used_readable": f"{disk.used / (1024**3):.2f} GB ({disk.used*100/disk.total:.1f}%)",
            "free_readable": f"{disk.free / (1024**3):.2f} GB ({disk.free*100/disk.total:.1f}%)",
        }
    except Exception as e:
        logger.error(f"Error getting disk usage for /: {e}", exc_info=True)
        snapshot["disk_usage_root"] = {"error": str(e)}

    try:
        proc_count_out, _, rc = _execute_command("ps -e --no-headers | wc -l")
        snapshot["running_processes_count"] = int(proc_count_out) if rc == 0 and proc_count_out.isdigit() else "Error executing ps or non-numeric output"
    except Exception as e:
        logger.error(f"Error counting running processes: {e}", exc_info=True)
        snapshot["running_processes_count"] = "Error"

    try:
        uptime_out, _, rc = _execute_command("uptime -p")
        snapshot["system_uptime"] = uptime_out if rc == 0 and uptime_out else "Error executing uptime or empty output"
    except Exception as e:
        logger.error(f"Error getting system uptime: {e}", exc_info=True)
        snapshot["system_uptime"] = "Error"

    return snapshot


if __name__ == '__main__':
    if not logger.hasHandlers():
        logger_setup.setup_logger("SystemStateAnalyzer_main_fallback", level=logging.DEBUG)
        logger.info("Logger re-initialized for __main__ block (fallback) with DEBUG level.")
    else:
        logger.setLevel(logging.DEBUG)
        logger.info("Logger already initialized. Set level to DEBUG for __main__ block tests.")

    logger.info("--- SystemStateAnalyzer Test ---")

    debian_version = get_debian_version()
    logger.info(f"Debian Version: {debian_version if debian_version else 'Not found or error'}")

    installed_packages = get_installed_packages()
    if installed_packages:
        logger.info(f"Installed Packages (first 3): {installed_packages[:3]}")
        logger.info(f"Total Installed Packages: {len(installed_packages)}")
    else:
        logger.warning("No installed packages found or error retrieving them.")

    common_services = ["cron", "ssh", "systemd-journald", "nonexistentservice_xyz123"]
    for service in common_services:
        status = get_service_status(service)
        is_active_status = is_service_active(service)
        logger.info(f"Service '{service}': Raw Status='{status}', IsActive='{is_active_status}'")

    project_root_for_test = config.PROJECT_ROOT if hasattr(config, 'PROJECT_ROOT') else pathlib.Path(__file__).parent.resolve()
    dummy_file_path = project_root_for_test / "dummy_test_file_system_analyzer.txt"
    logger.info(f"Dummy file will be at: {dummy_file_path}")

    try:
        dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dummy_file_path, "w", encoding="utf-8") as f:
            f.write("Hello, AI OS Enhancer!\nThis is a test file for system_analyzer.\nLine 3.")
        logger.info(f"Created dummy file: {dummy_file_path}")

        content = read_file_content(str(dummy_file_path))
        if content is not None:
            log_content = content[:60].replace('\n', '\\n')
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
        if i < 3:
             logger.info(f"Area: {area['name']}, Type: {area['type']}, Paths (up to 2): {area['paths'][:2]}..., Heuristic: {area['heuristic']}")
        elif i == 3:
             logger.info("... and potentially more areas (logging limited for brevity).")
    if not key_areas:
        logger.info("No key config/script areas were identified.")

    # --- Test new functions ---
    logger.info("\n--- Testing new System Analyzer functions ---")

    # Test check_ollama_service_availability
    # This test will depend on whether an Ollama instance is actually running at config.OLLAMA_API_ENDPOINT
    logger.info(f"Checking Ollama service at '{config.OLLAMA_API_ENDPOINT}'...")
    ollama_available = check_ollama_service_availability(config.OLLAMA_API_ENDPOINT)
    logger.info(f"Ollama service available at '{config.OLLAMA_API_ENDPOINT}': {ollama_available}")
    logger.info(f"Checking Ollama service at 'http://localhost:11223' (expected fail)...") # Non-standard port
    ollama_fail_test = check_ollama_service_availability('http://localhost:11223')
    logger.info(f"Ollama service available at 'http://localhost:11223': {ollama_fail_test}")

    # Test get_system_snapshot
    logger.info("--- System Snapshot ---")
    system_snapshot = get_system_snapshot()
    # Pretty print the snapshot dictionary
    for key, value in system_snapshot.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("-----------------------")

    logger.info("--- End of SystemStateAnalyzer Test ---")
