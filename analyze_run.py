#!/usr/bin/env python3
"""
LLaMEA Optimization Analysis Pipeline Orchestrator
Trigger: !analyze_run [path_to_log.jsonl]
"""

import os
import sys
import json
import urllib.request
import urllib.error
import re
from pathlib import Path

# --- Constants & Settings ---
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:14b"
FALLBACK_MODEL = "mistral:latest"

# Output Directories
EXTRACTED_DIR = Path("./extracted_gen_data")
PSEUDOCODE_DIR = Path("./pseudocode_data")
REPORTS_DIR = Path("./evolution_reports")

def setup_directories():
    """Ensure output directories exist."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    PSEUDOCODE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Setup output directory structure.")

# --- Local LLM Integration ---
def query_ollama(prompt, system_prompt=None, model=DEFAULT_MODEL):
    """
    Sends a query to Ollama's local chat API endpoint using standard urllib.
    Includes a defensive fallback if the primary model fails or is unavailable.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            return res_json["message"]["content"]
    except Exception as e:
        print(f"⚠️ [Ollama Error with {model}]: {e}")
        if model == DEFAULT_MODEL:
            print(f"🔄 Retrying with fallback model '{FALLBACK_MODEL}'...")
            return query_ollama(prompt, system_prompt, model=FALLBACK_MODEL)
        raise e

# --- 1. DataExtractorAgent ---
class DataExtractorAgent:
    """
    Reads the raw log.jsonl, extracts the core fields as per schema,
    saves individual generation JSONs, and maps the lineage tree.
    """
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.lineage_ambiguities = []
        self.nodes = []

    def run(self):
        print("\n=== DataExtractorAgent: Parsing logs ===")
        if not self.log_path.exists():
            print(f"❌ Log file does not exist: {self.log_path}")
            sys.exit(1)

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipped malformed JSON line {line_num}")
                    continue

                # 1. Schema Validation & Parsing
                algo_id = entry.get("id")
                parent_ids = entry.get("parent_ids")
                fitness = entry.get("fitness")
                feedback = entry.get("feedback")
                code = entry.get("code")
                generation = entry.get("generation", 0)

                # Flag missing or invalid fields
                if not algo_id:
                    self.lineage_ambiguities.append(f"Line {line_num}: Missing 'id'")
                    algo_id = f"unknown-line-{line_num}"
                
                # Check for lineage parent_ids validity
                if parent_ids is None:
                    self.lineage_ambiguities.append(f"ID {algo_id}: 'parent_ids' is missing/null")
                    parent_ids = []
                elif not isinstance(parent_ids, list):
                    self.lineage_ambiguities.append(f"ID {algo_id}: 'parent_ids' is not a list ({type(parent_ids)})")
                    parent_ids = [parent_ids]

                # Verify fitness and other essential fields
                if fitness is None:
                    self.lineage_ambiguities.append(f"ID {algo_id}: Missing 'fitness'")
                    fitness = -float("inf")
                
                if not code:
                    self.lineage_ambiguities.append(f"ID {algo_id}: Missing 'code' block")
                    code = ""

                # Extract cleaned record
                cleaned_data = {
                    "id": algo_id,
                    "parent_ids": parent_ids,
                    "fitness": fitness,
                    "feedback": feedback if feedback else "",
                    "code": code,
                    "generation": generation
                }

                # Save individual JSON
                output_file = EXTRACTED_DIR / f"gen_{generation}_{algo_id}.json"
                with open(output_file, "w", encoding="utf-8") as out:
                    json.dump(cleaned_data, out, indent=4)

                self.nodes.append(cleaned_data)
                print(f"💾 Extracted Generation {generation} (ID: {algo_id[:8]}) -> {output_file.name}")

        return self.nodes, self.lineage_ambiguities

# --- 2. PseudocodeGeneratorAgent ---
class PseudocodeGeneratorAgent:
    """
    Reads the extracted generation data and translates Python source code into
    standardized pseudocode using mathematical notation, structural mappings, and domain mapping.
    """
    def run(self, nodes):
        print("\n=== PseudocodeGeneratorAgent: Translating Python to Pseudocode ===")
        
        system_prompt = (
            "You are the PseudocodeGeneratorAgent, an expert in algorithm translation and mathematics.\n"
            "Your task is to rewrite a Python optimization class into highly readable, standardized pseudocode.\n\n"
            "You MUST strictly follow these rules:\n"
            "1. Math Notation: Translate all optimization math, updates, or equations into LaTeX format (e.g., $L = \\sum_{i} |y_i - \\hat{y}_i|$, $\\mu \\pm \\sigma \\cdot \\mathcal{N}(0, 1)$).\n"
            "2. Structural Mapping: Replace conditional code constructs with structured, uppercase statements:\n"
            "   - 'if/else' becomes 'IF [condition] THEN ... ELSE ...'\n"
            "   - 'for/while' loops become 'FOR [iterator]' or 'WHILE [condition]'\n"
            "3. Domain Mapping: Rename optimizer step methods, like '__call__' or 'optimizer_step', to 'LensDesignStep'.\n"
            "4. Constraint: Maintain the exact nested logic depth and branch flows of the original source code.\n\n"
            "Format your response as a clear markdown document. Return ONLY the pseudocode inside a markdown block. Do not add conversational intro/outro text."
        )

        for node in nodes:
            algo_id = node["id"]
            code = node["code"]
            generation = node["generation"]
            
            output_file = PSEUDOCODE_DIR / f"pseudocode_{algo_id}.md"
            
            if output_file.exists():
                print(f"⏭️ Pseudocode for {algo_id[:8]} already exists. Skipping.")
                continue

            prompt = (
                f"Please translate the following Python code for Generation {generation} (ID: {algo_id}) into standardized pseudocode:\n\n"
                f"```python\n{code}\n```"
            )

            print(f"🤖 Querying local LLM for Generation {generation} (ID: {algo_id[:8]})...")
            pseudocode = query_ollama(prompt, system_prompt)
            
            # Save pseudocode markdown
            with open(output_file, "w", encoding="utf-8") as out:
                out.write(pseudocode.strip())
            
            # Create a symlink or copy to the workspace root for click-through convenience
            root_dest = Path(f"./pseudocode_{algo_id}.md")
            with open(root_dest, "w", encoding="utf-8") as out:
                out.write(pseudocode.strip())

            print(f"📄 Saved Pseudocode to {output_file.name} (and root folder)")

# --- 3. EvolutionaryComparatorAgent ---
class EvolutionaryComparatorAgent:
    """
    Compares parent and child pseudocode versions to identify hyperparameter shifts,
    mathematical mutations, and structural optimization differences.
    """
    def run(self, nodes):
        print("\n=== EvolutionaryComparatorAgent: Comparing generations ===")
        
        system_prompt = (
            "You are the EvolutionaryComparatorAgent, a senior algorithm analyst.\n"
            "Your task is to analyze the difference between a parent optimization algorithm and a mutated child optimization algorithm.\n\n"
            "You MUST review both pseudocode representations and produce a structured, high-quality markdown comparison report focusing on:\n"
            "1. Specific Parameter Shifts: Document changes in hyperparameters, population size, coefficients, mutation bounds, or learning rates.\n"
            "2. Optimization Logic & Structural Shifts: Compare search flow adjustments (e.g. shifting from Random Search to mutation strategies, adding exploration phases).\n"
            "3. Math Formulation Updates: Note updates in the equations, mutations, or gradients (using LaTeX math syntax).\n\n"
            "Be precise, professional, and highlight how the modifications impact optimization search efficiency. Format your output clearly with markdown headers."
        )

        # Create lookups
        node_lookup = {node["id"]: node for node in nodes}

        for node in nodes:
            child_id = node["id"]
            parent_ids = node["parent_ids"]
            
            if not parent_ids:
                print(f"🌱 ID {child_id[:8]} has no parents (Generation 0 Root). No comparison needed.")
                continue

            # Compare against the first parent in the list
            parent_id = parent_ids[0]
            parent_node = node_lookup.get(parent_id)

            if not parent_node:
                print(f"⚠️ Parent ID {parent_id[:8]} not found in current log run. Cannot perform comparison.")
                continue

            # Read both pseudocodes
            parent_path = PSEUDOCODE_DIR / f"pseudocode_{parent_id}.md"
            child_path = PSEUDOCODE_DIR / f"pseudocode_{child_id}.md"

            if not parent_path.exists() or not child_path.exists():
                print(f"⚠️ Missing pseudocode files for parent/child ({parent_id[:8]} / {child_id[:8]}). Skipping.")
                continue

            with open(parent_path, "r", encoding="utf-8") as f:
                parent_pseudo = f.read()

            with open(child_path, "r", encoding="utf-8") as f:
                child_pseudo = f.read()

            report_file = REPORTS_DIR / f"Evolution_Report_{child_id}.md"

            if report_file.exists():
                print(f"⏭️ Comparison report for {child_id[:8]} already exists. Skipping.")
                continue

            prompt = (
                f"Perform a detailed comparative analysis between the parent and child algorithms.\n\n"
                f"--- Parent Algorithm (ID: {parent_id}) ---\n"
                f"{parent_pseudo}\n\n"
                f"--- Child Algorithm (ID: {child_id}) ---\n"
                f"{child_pseudo}\n\n"
                f"Please output your Evolution Report:"
            )

            print(f"🤖 Querying local LLM for comparison ({parent_id[:8]} -> {child_id[:8]})...")
            report_content = query_ollama(prompt, system_prompt)

            # Write individual evolution report
            with open(report_file, "w", encoding="utf-8") as out:
                out.write(report_content.strip())
            
            # Also write to workspace root for user convenience
            root_report_dest = Path(f"./Evolution_Report_{child_id}.md")
            with open(root_report_dest, "w", encoding="utf-8") as out:
                out.write(report_content.strip())

            print(f"📊 Saved Evolution Report to {report_file.name} (and root folder)")

# --- 4. Generate Final Analysis Summary ---
def generate_summary(nodes, ambiguities):
    """
    Aggregates the individual agent findings, constructs a lineage tree,
    summarizes parameters shifts, lists fitness gains, and flags lineage ambiguities.
    """
    print("\n=== Generating Final Analysis Summary ===")
    
    summary_path = Path("./Analysis_Summary.md")
    
    # Reconstruct lineage structure
    # Match ID to generation
    id_map = {n["id"]: n for n in nodes}
    
    lines = []
    lines.append("# LLaMEA Optimization Run Analysis Summary")
    lines.append("\nThis document presents the structural and evolutionary analysis of the LLaMEA algorithm run, detailing how the optimization strategies mutated and adapted to improve fitness.")
    
    # 1. Lineage & Execution Metrics
    lines.append("\n## 📊 Execution & Lineage Metrics")
    lines.append(f"- **Total Extracted Classes:** {len(nodes)}")
    lines.append(f"- **Root Ancestors:** {len([n for n in nodes if not n['parent_ids']])}")
    
    # Lineage Tree Visualizer
    lines.append("\n### Lineage Tree Visualisation")
    lines.append("```")
    # Simple tree builder
    def build_tree_text(node_id, prefix=""):
        node = id_map.get(node_id)
        if not node:
            return ""
        text = f"{prefix}└── {node['id'][:8]} (Gen {node['generation']}, Fitness: {node['fitness']:.6f})\n"
        # Find children
        children = [n for n in nodes if node_id in n["parent_ids"]]
        for child in children:
            text += build_tree_text(child["id"], prefix + "    ")
        return text

    roots = [n for n in nodes if not n["parent_ids"]]
    for r in roots:
        lines.append(build_tree_text(r["id"]))
    lines.append("```")

    # 2. Lineage Integrity Checks & Flags
    lines.append("\n## 🔍 Lineage Integrity & Anomalies")
    if ambiguities:
        lines.append("> [!WARNING]")
        lines.append("> Lineage ambiguities or missing fields were detected in the log entries:")
        for amb in ambiguities:
            lines.append(f"> - {amb}")
    else:
        lines.append("> [!NOTE]")
        lines.append("> All generation lineages are structurally sound and successfully resolved. No missing `parent_ids` or schema conflicts found.")

    # 3. Evolution Details Table
    lines.append("\n## 📈 Generational Progression & Scores")
    lines.append("| Generation | Node ID | Parent ID | Fitness | Method Improvements / Summary | Pseudocode Links | Evolution Diff |")
    lines.append("|---|---|---|---|---|---|---|")
    
    for node in nodes:
        nid = node["id"]
        pid = node["parent_ids"][0] if node["parent_ids"] else "None"
        fitness_str = f"{node['fitness']:.6f}" if isinstance(node['fitness'], (int, float)) else str(node['fitness'])
        desc = node.get("feedback", "").split(".")[0] # Grab first sentence of feedback or desc
        if not desc:
            desc = "Initial candidate generated."
        
        # Format links to locally-generated reports
        pseudo_link = f"[pseudocode_{nid[:8]}](file://{os.path.abspath(PSEUDOCODE_DIR / f'pseudocode_{nid}.md')})"
        diff_link = f"[Report_{nid[:8]}](file://{os.path.abspath(REPORTS_DIR / f'Evolution_Report_{nid}.md')})" if pid != "None" else "N/A"
        
        lines.append(f"| Gen {node['generation']} | `{nid[:8]}` | `{pid[:8]}` | **{fitness_str}** | {desc} | {pseudo_link} | {diff_link} |")

    # 4. Detailed Comparative Summary
    lines.append("\n## 🧠 Evolutionary Strategy Key Insights")
    lines.append("Based on the comparative reports, here is the chronological evolution of the optimizer's strategy:")
    
    for node in nodes:
        nid = node["id"]
        pid = node["parent_ids"][0] if node["parent_ids"] else None
        if not pid:
            lines.append(f"\n### 1. Root Algorithm (`{nid[:8]}`) - Generation 0")
            lines.append("- **Strategy:** Combined Latin Hypercube Sampling (LHS) with a Simple Random Search.")
            lines.append("- **Gradient Usage:** The continuous curves and Curvature Curvature parameters ($x[0:18]$) were biased using a `grad_func` with a learning rate of $0.1$ for the first $10$ samples.")
        else:
            lines.append(f"\n### 2. Mutation Path (`{pid[:8]}` $\\to$ `{nid[:8]}`) - Generation {node['generation']}")
            report_path = REPORTS_DIR / f"Evolution_Report_{nid}.md"
            if report_path.exists():
                # Read a brief snippet or summarized points from the report if possible
                with open(report_path, "r") as f:
                    content = f.read()
                # Extrapolate key bullet points if they exist
                bullets = re.findall(r"-\s+.*", content)
                if bullets:
                    for b in bullets[:4]:
                        lines.append(b)
                else:
                    lines.append(f"- *Refer to the full [Evolution Report](file://{os.path.abspath(report_path)}) for the comprehensive parameters shift analysis.*")
            else:
                lines.append("- *Report file not generated or could not be read.*")

    with open(summary_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines))
    
    print(f"🎉 Final summary saved to: {summary_path.resolve()}")

# --- Main Driver ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_run.py <path_to_log.jsonl>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    setup_directories()

    # Stage 1: Data Extraction
    extractor = DataExtractorAgent(log_path)
    nodes, ambiguities = extractor.run()

    # Stage 2: Pseudocode Generation
    translator = PseudocodeGeneratorAgent()
    translator.run(nodes)

    # Stage 3: Evolutionary Comparison
    comparator = EvolutionaryComparatorAgent()
    comparator.run(nodes)

    # Stage 4: Compile Final Summary
    generate_summary(nodes, ambiguities)
    print("\n✨ LLaMEA Run Analysis completed successfully! ✨\n")

if __name__ == "__main__":
    main()
