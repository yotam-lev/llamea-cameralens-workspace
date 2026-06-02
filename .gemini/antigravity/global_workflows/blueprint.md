# Workflow: /blueprint
# Description: Initialize a comprehensive multi-agent technical plan for any complex task.

## Step 1: Context Gathering (Orchestrator)
- Scan the root directory to understand the project type, language, and structural setup.
- Identify existing configuration files, environment definitions, and relevant dependencies.

## Step 2: Architecture Draft (Strategist)
Produce a clear markdown technical proposal using this exact structure:
- **Objective:** The core problem being solved.
- **Architectural Changes:** What new components, classes, data tables, or modules are needed.
- **File Matrix:** A markdown table listing every file to be created, modified, or deleted.
- **Edge Cases:** At least three potential breaking points or performance bottlenecks (e.g., memory overhead, API limits, race conditions).
- **Validation Plan:** Concrete steps to verify the solution works.

## Step 3: Handoff
- Present the blueprint to the user.
- Explicitly prompt: "Type `/execute` to begin implementing this plan, or provide feedback to adjust the architecture."