# Server Execution Security Protocol

**Warning:** Executing arbitrary LLM-generated code via network sockets carries significant risk. 

**Mandatory Security Layers:**
1. **Syntax Validation:** The server must execute `ast.parse(code_string)` before any `exec()`. If the LLM generates `os.system('rm -rf /')`, the parser will catch the structure, but execution must remain sandboxed.
2. **Global Namespace Isolation:** The `exec(code_string, sandbox_env)` must be restricted. Do not pass `__builtins__` fully. Only pass a whitelist of safe modules (`np`, `scipy`, `DoubleGaussObjective`).
3. **Timeout Enforcement:** Use a signal-based alarm (`signal.alarm`) to terminate any optimizer that exceeds the evaluation budget execution time.