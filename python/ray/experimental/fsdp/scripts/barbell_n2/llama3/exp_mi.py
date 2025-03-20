import logging
import os
import subprocess
import sys

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def run_bash_script(script_path):
    """
    Run a bash script and return its exit code, stdout, and stderr.

    Args:
        script_path (str): Path to the bash script to run

    Returns:
        tuple: (script_path, exit_code, stdout, stderr)
    """
    # First check if the file exists
    if not os.path.isfile(script_path):
        logger.error(f"Script file {script_path} does not exist")
        return script_path, -1, "", f"Script file {script_path} does not exist"

    # Make sure the script is executable
    if not os.access(script_path, os.X_OK):
        os.chmod(script_path, 0o755)

    try:
        # Run the script using bash explicitly with full path
        logger.info(f"Executing {os.path.abspath(script_path)}...")
        process = subprocess.run(
            ["bash", os.path.abspath(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            env=os.environ.copy(),  # Pass current environment
        )

        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr
        return script_path, exit_code, stdout, stderr
    except Exception as e:
        logger.error(f"Error executing {script_path}: {str(e)}")
        return script_path, -1, "", str(e)


def main():
    # Get current working directory for debugging
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")

    # Scripts to run sequentially
    scripts = [
        "scripts/barbell_n2/llama3/ray/cc_off/ov_off/exp_self.sh",
        "scripts/barbell_n2/llama3/ray/cc_on/ov_off/exp_self.sh",
        "scripts/barbell_n2/llama3/ray/cc_on/ov_on/exp_self.sh",
        "scripts/barbell_n2/llama3/torch/cc_off/fp_on/exp_self.sh",
        "scripts/barbell_n2/llama3/torch/cc_on/fp_on/pf_on/exp_self.sh",
    ]

    # Verify scripts exist
    for script in scripts:
        if not os.path.exists(script):
            logger.error(f"Error: Script '{script}' not found", file=sys.stderr)
            sys.exit(1)

    # Run scripts sequentially
    for script in scripts:
        run_bash_script(script)


if __name__ == "__main__":
    main()
