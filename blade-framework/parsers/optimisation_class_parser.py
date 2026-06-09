class OptimisationClassParser:
    """
    A utility class to parse and separate lines from a string containing 
    optimization class code.
    """
    def __init__(self, optclass: str):
        """
        Initialize the parser with the raw class code string.
        
        Args:
            optclass (str): The whole class string with lines separated by \n.
        """
        self.optclass = optclass

    def get_separated_lines(self) -> list[str]:
        """
        Returns the whole class with each line separated into a list.
        """
        return self.optclass.split('\n')

    def print_separated_lines(self):
        """
        Outputs the whole class with each line printed separately.
        """
        lines = self.get_separated_lines()
        for i, line in enumerate(lines, 1):
            print(f"{line}")

if __name__ == "__main__":
    # Example usage:
    example_code ="""
    "import os\n",
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "def get_workspace_root():\n",
    "    cwd = Path.cwd()\n",
    "    while cwd.name != 'llamea-cameralens-workspace' and cwd.parent != cwd:\n",
    "        cwd = cwd.parent\n",
    "    if cwd.name == 'llamea-cameralens-workspace':\n",
    "        return cwd\n",
    "    return Path.cwd()\n",
    "\n",
    "WORKSPACE_ROOT = get_workspace_root()\n",
    "SRC_PATH = WORKSPACE_ROOT / \"blade-framework\" / \"output_analysis\" / \"src\"\n",
    "\n",
    "if str(SRC_PATH) not in sys.path:\n",
    "    sys.path.append(str(SRC_PATH))\n",
    "\n",
    "import analysis\n",
    "\n",
    "EXPERIMENT_NAME = \"lens_v5_50000_False_03_06\"\n",
    "INPUT_DIR = WORKSPACE_ROOT / \"blade-framework\" / \"results\" / \"Lens_v5_50000_False_03_06\" / \"llamea_run_lens_v5_50000_F_03_06\"\n"
    "\n"
    "# Ensure output directories exist\n"
    "analysis.setup_directories(EXPERIMENT_NAME, str(INPUT_DIR))\n"
    "print(f\"Setup complete for experiment: {EXPERIMENT_NAME}\")"
    """
    
    parser = OptimisationClassParser(example_code)
    print("--- Separated Lines ---")
    parser.print_separated_lines()
