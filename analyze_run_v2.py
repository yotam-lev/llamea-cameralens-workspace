#!/usr/bin/env python3
"""
LLaMEA Scaled Optimization Analysis Pipeline Orchestrator (v2)
Trigger: !analyze_run_v2 [path_to_log.jsonl]
"""

import os
import sys
import json
import urllib.request
import urllib.error
import re
import difflib
from pathlib import Path
import concurrent.futures
import time

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
    print("✅ Setup v2 output directory structure.")

# --- Local LLM Integration ---
def query_ollama(prompt, system_prompt=None, model=DEFAULT_MODEL, retries=3, delay=2):
    """
    Sends a query to Ollama's local chat API endpoint using standard urllib.
    Includes an exponential backoff and automatic model fallback.
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
    
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=240) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                return res_json["message"]["content"]
        except Exception as e:
            print(f"⚠️ [Ollama Attempt {attempt} Error with {model}]: {e}")
            if attempt == retries:
                if model == DEFAULT_MODEL:
                    print(f"🔄 Retrying with fallback model '{FALLBACK_MODEL}'...")
                    return query_ollama(prompt, system_prompt, model=FALLBACK_MODEL, retries=retries, delay=delay)
                raise e
            time.sleep(delay * (2 ** (attempt - 1)))

# --- 1. DataExtractorAgent (Upgraded) ---
class DataExtractorAgent:
    """
    Parses logs, building a lineage mapping, generates fitness history,
    computes quartiles, and constructs a color-coded Mermaid.js DAG tree.
    """
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.lineage_ambiguities = []
        self.nodes = []

    def run(self):
        print("\n=== DataExtractorAgent (v2): Parsing logs & building DAG ===")
        if not self.log_path.exists():
            print(f"❌ Log file does not exist: {self.log_path}")
            sys.exit(1)

        # 1. Parsing JSONL
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

                algo_id = entry.get("id")
                parent_ids = entry.get("parent_ids")
                fitness = entry.get("fitness")
                feedback = entry.get("feedback")
                code = entry.get("code")
                generation = entry.get("generation", 0)

                if not algo_id:
                    self.lineage_ambiguities.append(f"Line {line_num}: Missing 'id'")
                    algo_id = f"unknown-line-{line_num}"
                
                if parent_ids is None:
                    self.lineage_ambiguities.append(f"ID {algo_id}: 'parent_ids' is missing/null")
                    parent_ids = []
                elif not isinstance(parent_ids, list):
                    self.lineage_ambiguities.append(f"ID {algo_id}: 'parent_ids' is not a list ({type(parent_ids)})")
                    parent_ids = [parent_ids]

                if fitness is None:
                    self.lineage_ambiguities.append(f"ID {algo_id}: Missing 'fitness'")
                    fitness = -float("inf")
                
                if not code:
                    self.lineage_ambiguities.append(f"ID {algo_id}: Missing 'code' block")
                    code = ""

                cleaned_data = {
                    "id": algo_id,
                    "parent_ids": parent_ids,
                    "fitness": fitness,
                    "feedback": feedback if feedback else "",
                    "code": code,
                    "generation": generation,
                    "name": entry.get("name", "Optimizer")
                }

                output_file = EXTRACTED_DIR / f"gen_{generation}_{algo_id}.json"
                with open(output_file, "w", encoding="utf-8") as out:
                    json.dump(cleaned_data, out, indent=4)

                self.nodes.append(cleaned_data)

        # 2. Generate fitness_history.csv
        csv_path = Path("fitness_history.csv")
        with open(csv_path, "w", encoding="utf-8") as csv:
            csv.write("generation,id,fitness\n")
            for node in self.nodes:
                csv.write(f"{node['generation']},{node['id']},{node['fitness']}\n")
        print(f"📈 Generated fitness history table in {csv_path.name}")

        # 3. Calculate Fitness Quartiles for color-coding
        valid_fitnesses = [
            node["fitness"] for node in self.nodes 
            if isinstance(node["fitness"], (int, float)) and node["fitness"] > -float("inf") and node["fitness"] < float("inf")
        ]
        
        q1 = q2 = q3 = 0.0
        if valid_fitnesses:
            valid_fitnesses.sort()
            n_len = len(valid_fitnesses)
            def get_percentile(p):
                idx = p * (n_len - 1)
                idx_low = int(idx)
                idx_high = min(idx_low + 1, n_len - 1)
                weight = idx - idx_low
                return (1 - weight) * valid_fitnesses[idx_low] + weight * valid_fitnesses[idx_high]
            q1 = get_percentile(0.25)
            q2 = get_percentile(0.50)
            q3 = get_percentile(0.75)
            print(f"📊 Fitness Quartiles: Q1={q1:.6f}, Q2={q2:.6f}, Q3={q3:.6f}")

        # 4. Generate Mermaid string
        mermaid_lines = ["graph TD"]
        
        # Styles
        mermaid_lines.append("    classDef q4 fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff;")
        mermaid_lines.append("    classDef q3 fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff;")
        mermaid_lines.append("    classDef q2 fill:#f1c40f,stroke:#d35400,stroke-width:2px,color:#fff;")
        mermaid_lines.append("    classDef q1 fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff;")
        mermaid_lines.append("    classDef failed fill:#7f8c8d,stroke:#34495e,stroke-width:2px,color:#fff;")

        # Nodes declaration & mapping
        for node in self.nodes:
            nid = node["id"]
            short_id = nid[:8]
            fit = node["fitness"]
            gen = node["generation"]
            fit_str = f"{fit:.4f}" if isinstance(fit, (int, float)) and fit != -float("inf") else "inf"
            
            # Node description
            node_desc = f"n_{nid}[\"{short_id}<br/>Gen {gen}<br/>f: {fit_str}\"]"
            mermaid_lines.append(f"    {node_desc}")
            
            # Parent relationships
            for pid in node["parent_ids"]:
                mermaid_lines.append(f"    n_{pid} --> n_{nid}")
                
            # Class assignment based on quartile
            if fit == -float("inf") or fit is None:
                q_class = "failed"
            else:
                if fit >= q3:
                    q_class = "q4"
                elif fit >= q2:
                    q_class = "q3"
                elif fit >= q1:
                    q_class = "q2"
                else:
                    q_class = "q1"
            mermaid_lines.append(f"    class n_{nid} {q_class};")

        mermaid_str = "\n".join(mermaid_lines)
        mmd_path = Path("lineage.mmd")
        with open(mmd_path, "w", encoding="utf-8") as out:
            out.write(mermaid_str)
        print(f"🗺️ Generated Mermaid lineage graph at {mmd_path.name}")

        return self.nodes, self.lineage_ambiguities, mermaid_str

# --- 2. PseudocodeGeneratorAgent (Concurrent) ---
class PseudocodeGeneratorAgent:
    """
    Rewrites optimization scripts into LaTeX standard math concurrently in a ThreadPoolExecutor.
    """
    def translate_node(self, node):
        algo_id = node["id"]
        code = node["code"]
        generation = node["generation"]
        
        output_file = PSEUDOCODE_DIR / f"pseudocode_{algo_id}.md"
        
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

        prompt = (
            f"Please translate the following Python code for Generation {generation} (ID: {algo_id}) into standardized pseudocode:\n\n"
            f"```python\n{code}\n```"
        )

        try:
            pseudocode = query_ollama(prompt, system_prompt)
            with open(output_file, "w", encoding="utf-8") as out:
                out.write(pseudocode.strip())
            
            # Save root copies for IDE click-through navigation
            root_dest = Path(f"./pseudocode_{algo_id}.md")
            with open(root_dest, "w", encoding="utf-8") as out:
                out.write(pseudocode.strip())
                
            return algo_id, True
        except Exception as e:
            print(f"❌ Failed pseudocode generation for ID {algo_id[:8]}: {e}")
            return algo_id, False

    def run(self, nodes):
        print("\n=== PseudocodeGeneratorAgent (v2): Processing concurrently ===")
        start_time = time.time()
        
        # Sizing thread pool to machine capabilities
        max_workers = min(8, os.cpu_count() or 4)
        print(f"🧵 Thread Pool configured with {max_workers} parallel workers.")
        
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for node in nodes:
                futures.append(executor.submit(self.translate_node, node))
            
            for future in concurrent.futures.as_completed(futures):
                node_id, success = future.result()
                if success:
                    print(f"📄 Concurrently processed pseudocode for ID {node_id[:8]}")

        print(f"⏳ Concurrent pseudocode generation completed in {time.time() - start_time:.2f} seconds.")

# --- 3. EvolutionaryComparatorAgent (Diff-focused) ---
class EvolutionaryComparatorAgent:
    """
    Calculates unified diffs between parent and child code using Python's difflib,
    passing only the diff and Delta f to the LLM to minimize context bloat.
    """
    def compare_node(self, node, nodes_lookup):
        child_id = node["id"]
        parent_ids = node["parent_ids"]
        
        if not parent_ids:
            return child_id, "None", None
            
        parent_id = parent_ids[0]
        parent_node = nodes_lookup.get(parent_id)
        if not parent_node:
            return child_id, "Missing", None

        # 1. Syntactic Diffing using difflib
        parent_lines = parent_node["code"].splitlines(keepends=True)
        child_lines = node["code"].splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            parent_lines, child_lines,
            fromfile=f"parent_{parent_id[:8]}.py",
            tofile=f"child_{child_id[:8]}.py",
            n=2
        )
        diff_str = "".join(diff)
        
        # 2. Fitness Progression Math
        f_parent = parent_node["fitness"]
        f_child = node["fitness"]
        delta_f = f_child - f_parent

        system_prompt = (
            "You are the EvolutionaryComparatorAgent, a senior algorithm analyst.\n"
            "Your task is to analyze the difference between a parent optimization algorithm and a child optimization algorithm.\n\n"
            "You will be given a unified diff of the code and the fitness improvement (delta f = f_child - f_parent).\n"
            "You MUST review this diff and output a concise report highlighting exactly WHAT changed, focusing on:\n"
            "1. Parameter Shifts: Hyperparameter changes, bounds, population settings (e.g. learning rate changes).\n"
            "2. Optimization Logic Shifts: Search adjustments (e.g. from LHS/random search to Gaussian mutation).\n"
            "3. Math updates: Detail the changes in perturbation math or gradient decays using LaTeX syntax.\n\n"
            "Ensure your response is dense, diff-focused, and formatted in clean markdown. Do not include excessive parent or child code blocks."
        )

        prompt = (
            f"Parent ID: {parent_id[:8]} (Fitness: {f_parent:.6f})\n"
            f"Child ID: {child_id[:8]} (Fitness: {f_child:.6f})\n"
            f"Calculated Delta Fitness (f_child - f_parent): {delta_f:.6f}\n\n"
            f"Unified Diff:\n"
            f"```diff\n{diff_str}\n```\n\n"
            f"Please output your semantic Evolutionary Comparison Report:"
        )

        try:
            report_content = query_ollama(prompt, system_prompt)
            output_file = REPORTS_DIR / f"Evolution_Report_{child_id}.md"
            with open(output_file, "w", encoding="utf-8") as out:
                out.write(report_content.strip())
            
            # Save root copies for IDE click-through navigation
            root_report_dest = Path(f"./Evolution_Report_{child_id}.md")
            with open(root_report_dest, "w", encoding="utf-8") as out:
                out.write(report_content.strip())

            return child_id, parent_id, report_content.strip()
        except Exception as e:
            print(f"❌ Failed comparison report for child ID {child_id[:8]}: {e}")
            return child_id, parent_id, None

    def run(self, nodes):
        print("\n=== EvolutionaryComparatorAgent (v2): Pre-diffing & analyzing ===")
        start_time = time.time()
        
        nodes_lookup = {node["id"]: node for node in nodes}
        reports = {}
        
        # Maximize concurrency for comparator too
        max_workers = min(8, os.cpu_count() or 4)
        futures = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for node in nodes:
                futures.append(executor.submit(self.compare_node, node, nodes_lookup))
                
            for future in concurrent.futures.as_completed(futures):
                child_id, parent_id, report = future.result()
                if report:
                    reports[child_id] = report
                    print(f"📊 Synthesized diff report for {parent_id[:8]} -> {child_id[:8]}")

        print(f"⏳ Concurrency-focused diff comparison completed in {time.time() - start_time:.2f} seconds.")
        return reports

# --- 4. MacroAnalyzerAgent (NEW) ---
class MacroAnalyzerAgent:
    """
    Groups generations into cohort epochs (e.g. chunks of 20), reads micro-reports,
    and queries Ollama to construct a high-level timeline of the global optimization trajectory.
    """
    def run(self, nodes, reports):
        print("\n=== MacroAnalyzerAgent: Synthesizing Epoch Cohorts ===")
        
        if not reports:
            print("⚠️ No comparator reports found. Bypassing Macro analysis.")
            return "No macro epoch comparisons available."

        # Group nodes by cohort (bins of 20 generations)
        cohorts = {}
        for node in nodes:
            gen = node["generation"]
            cohort_idx = gen // 20
            cohort_name = f"Generations {cohort_idx*20}-{(cohort_idx+1)*20-1}"
            
            if cohort_name not in cohorts:
                cohorts[cohort_name] = []
            cohorts[cohort_name].append(node)

        macro_summaries = []

        system_prompt = (
            "You are the MacroAnalyzerAgent, a principal algorithm auditor.\n"
            "Your task is to analyze chronological summaries of optimization mutations across a cohort of generations.\n\n"
            "You will be given the changes that occurred inside a cohort (epoch) of generations.\n"
            "You MUST review these shifts and compile a concise high-level synthesis of this epoch's trajectory. Identify:\n"
            "1. Structural search focus: Did it prioritize global exploration (random search, large steps) or local exploitation (gradient bias, micro Gaussian perturbation)?\n"
            "2. Critical mutations: Highlight any breakthrough hyperparameter or math changes in this cohort.\n\n"
            "Write in a dense, highly analytical style. Format as structured markdown."
        )

        for cohort_name, cohort_nodes in sorted(cohorts.items()):
            cohort_reports = []
            for node in cohort_nodes:
                nid = node["id"]
                if nid in reports:
                    cohort_reports.append(f"Node `{nid[:8]}` (Gen {node['generation']}):\n{reports[nid]}")

            if not cohort_reports:
                continue

            cohort_payload = "\n\n---\n\n".join(cohort_reports[:10]) # Cap at 10 reports to avoid token bloat
            
            prompt = (
                f"### Analysis Cohort: {cohort_name} ###\n\n"
                f"Here are the chronological evolutionary summaries for this epoch:\n\n"
                f"{cohort_payload}\n\n"
                f"Please compile the Macro Trajectory analysis for this epoch:"
            )

            print(f"🤖 Compiling Macro Trajectory timeline for {cohort_name}...")
            epoch_summary = query_ollama(prompt, system_prompt)
            macro_summaries.append(f"## 📅 Epoch: {cohort_name}\n\n{epoch_summary.strip()}")

        return "\n\n".join(macro_summaries)

# --- 5. Evolution Dashboard Compiler ---
def compile_dashboard(nodes, ambiguities, mermaid_graph, reports, macro_timeline):
    print("\n=== Compiling Evolution Dashboard ===")
    
    dashboard_path = Path("Evolution_Dashboard.md")
    
    lines = []
    lines.append("# LLaMEA Scaled Evolution Dashboard")
    lines.append("\nThis dashboard presents the parallelized, diff-focused analysis of the LLaMEA algorithm run across all generations.")
    
    # 1. Mermaid Lineage Diagram
    lines.append("\n## 🗺️ Lineage DAG Flowchart")
    lines.append("The following Mermaid.js DAG maps out all parent-child connections, color-coded by their fitness performance quartiles:")
    lines.append("\n> [!TIP]\n> **Color Coding Legend:**\n> - **Green (Q4):** Top 25% best performing fitness.\n> - **Blue (Q3):** Upper-middle performance (50%-75%)\n> - **Yellow (Q2):** Lower-middle performance (25%-50%)\n> - **Red (Q1):** Bottom 25% performing fitness.\n> - **Gray:** Failed runs (`-inf` fitness).")
    lines.append("\n```mermaid")
    lines.append(mermaid_graph)
    lines.append("```")

    # 2. Leaderboard: Top 5 Highest Fitness Mutations
    lines.append("\n## 🏆 Leaderboard: Top 5 Highest Fitness Mutations")
    lines.append("| Rank | Generation | Node ID | Parent ID | Fitness | Delta Fitness ($\\Delta f$) | Strategy Improvement Summary |")
    lines.append("|---|---|---|---|---|---|---|")
    
    # Filter valid sorted nodes
    valid_nodes = [
        n for n in nodes 
        if isinstance(n["fitness"], (int, float)) and n["fitness"] > -float("inf")
    ]
    valid_nodes.sort(key=lambda x: x["fitness"], reverse=True)
    
    # Map node lookups for delta f
    id_map = {n["id"]: n for n in nodes}

    for idx, node in enumerate(valid_nodes[:5], 1):
        nid = node["id"]
        pid = node["parent_ids"][0] if node["parent_ids"] else "None"
        parent_node = id_map.get(pid) if pid != "None" else None
        
        f_child = node["fitness"]
        f_parent = parent_node["fitness"] if parent_node else f_child
        delta_f = f_child - f_parent
        
        # Get brief report synopsis
        desc = node.get("feedback", "").split(".")[0]
        if not desc:
            desc = "Breakthrough candidate generated."

        lines.append(f"| #{idx} | Gen {node['generation']} | `{nid[:8]}` | `{pid[:8]}` | **{f_child:.6f}** | `+{delta_f:.6f}` | {desc} |")

    # 3. Macro Epoch Summaries
    lines.append("\n## 📅 Chronological Epoch Cohort Timeline")
    lines.append(macro_timeline)

    # 4. Lineage Integrity Checks
    lines.append("\n## 🔍 Lineage Integrity & Anomalies")
    if ambiguities:
        lines.append("> [!WARNING]")
        lines.append("> Lineage anomalies or invalid fields were flagged during processing:")
        for amb in ambiguities:
            lines.append(f"> - {amb}")
    else:
        lines.append("> [!NOTE]")
        lines.append("> All generation lineages are structurally sound and successfully resolved. No missing `parent_ids` or schema conflicts found.")

    # 5. Local links
    lines.append("\n## 📄 Reference Links")
    lines.append("- **Lineage DAG Schema:** [lineage.mmd](file://%s)" % os.path.abspath("lineage.mmd"))
    lines.append("- **Fitness History Table:** [fitness_history.csv](file://%s)" % os.path.abspath("fitness_history.csv"))
    lines.append("- **Individual Pseudocodes:** Located inside `./pseudocode_data/` directory.")
    lines.append("- **Individual Diff Reports:** Located inside `./evolution_reports/` directory.")

    with open(dashboard_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines))
        
    print(f"🎉 Compiled master Evolution Dashboard to: {dashboard_path.resolve()}")

# --- Main Driver ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_run_v2.py <path_to_log.jsonl>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    setup_directories()

    # Stage 1: Data Extraction & DAG Flowcharting
    extractor = DataExtractorAgent(log_path)
    nodes, ambiguities, mermaid_graph = extractor.run()

    # Stage 2: Concurrent Logic Translation
    translator = PseudocodeGeneratorAgent()
    translator.run(nodes)

    # Stage 3: Diff-Focused Comparative Analysis
    comparator = EvolutionaryComparatorAgent()
    reports = comparator.run(nodes)

    # Stage 4: MacroEpoch Summarization
    macro_agent = MacroAnalyzerAgent()
    macro_timeline = macro_agent.run(nodes, reports)

    # Stage 5: Compile Final Evolution Dashboard
    compile_dashboard(nodes, ambiguities, mermaid_graph, reports, macro_timeline)
    print("\n✨ Scaled LLaMEA Run Analysis v2 completed successfully! ✨\n")

if __name__ == "__main__":
    main()
