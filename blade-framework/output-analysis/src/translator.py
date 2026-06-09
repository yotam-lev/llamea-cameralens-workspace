import requests
import json
from pathlib import Path
from typing import List, Optional
try:
    from .canonicalizer import Canonicalizer
except ImportError:
    # pyrefly: ignore [missing-import]
    from canonicalizer import Canonicalizer
class Node:
    def __init__(self, line: str, indent: int):
        self.line = line
        self.indent = indent
        self.children: List['Node'] = []
        self.translation: Optional[str] = None
class Translator:
    """
    Implements a deepest-first recursive translation parser based on indentation.
    Replaces internal chunks with boundary stickers and queries Ollama for translations.
    """
    def __init__(self, canonicalizer: Canonicalizer, model: str = "qwen2.5-coder:14b", api_url: str = "http://localhost:11434/api/generate", output_dir: Optional[Path] = None, opt_id: Optional[str] = None):
        self.canonicalizer = canonicalizer
        self.model = model
        self.api_url = api_url
        self.output_dir = output_dir
        self.opt_id = opt_id
        self.iteration = 0
    def build_tree(self, code: str) -> Node:
        """
        Parses python code by indentation into a tree of Nodes.
        """
        lines = code.split('\n')
        root = Node("", -1)
        stack = [root]
        
        for line in lines:
            if not line.strip():
                continue
            
            indent = len(line) - len(line.lstrip())
            node = Node(line, indent)
            
            # Pop stack until we find a parent with less indentation
            while stack and stack[-1].indent >= indent:
                stack.pop()
                
            if not stack:
                stack.append(root)
                
            stack[-1].children.append(node)
            stack.append(node)
            
        return root
    def get_node_code_with_stickers(self, node: Node) -> str:
        """
        Reconstruct the code for this node, replacing translated children
        with // :::PSEUDOCODE::: boundary stickers.
        """
        lines = []
        if node.line:
            lines.append(node.line)
            
        for child in node.children:
            if child.translation:
                # Add the boundary sticker for the translated child
                indent_str = " " * child.indent
                t_lines = child.translation.strip().split('\n')
                for i, tline in enumerate(t_lines):
                    if i == 0:
                        lines.append(f"{indent_str}// :::PSEUDOCODE::: {tline}")
                    else:
                        lines.append(f"{indent_str}// {tline}")
            else:
                # If the child has no translation, recursively get its code
                child_code = self.get_node_code_with_stickers(child)
                if child_code:
                    lines.append(child_code)
                    
        return '\n'.join(lines)
    def get_full_tree_state(self, root: Node) -> str:
        """
        Get the current state of the whole translation tree.
        If the root is translated, return its clean translation.
        Otherwise, return the code reconstructed with boundary stickers.
        """
        if root.translation:
            return root.translation
        return self.get_node_code_with_stickers(root)
    def get_raw_code(self, node: Node) -> str:
        """Get the original, unmodified python code for a node for canonicalizer."""
        lines = []
        if node.line:
            lines.append(node.line)
        for child in node.children:
            child_raw = self.get_raw_code(child)
            if child_raw:
                lines.append(child_raw)
        return '\n'.join(lines)
    def query_ollama(self, code_chunk: str) -> str:
        """
        Passes the abstracted chunk and the known translation dictionary to the LLM.
        """
        # We pass a summary of the known dictionary context or just prompt context
        translations = self.canonicalizer.get_all_translations()
        # For very large dictionaries, we might want to trim this, but as per instructions
        # we pass the known translation dictionary.
        dict_context = json.dumps(translations, indent=2)
        
        prompt = (
            "You are an expert Python code summarizer.\n"
            "Translate the following abstracted Python code chunk into concise pseudocode.\n"
            "The code contains '// :::PSEUDOCODE:::' boundary stickers which represent already translated inner blocks. "
            "Incorporate the meaning of these stickers into your final pseudocode.\n\n"
            "Here is the known translation dictionary of previously canonicalized blocks for context:\n"
            f"{dict_context}\n\n"
            "Abstracted Code Chunk:\n"
            "```python\n"
            f"{code_chunk}\n"
            "```\n\n"
            "Concise Pseudocode:"
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
            except Exception as e:
                print(f"Error querying Ollama (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return "Translation failed."
    def translate_node(self, node: Node, root: Node) -> None:
        """
        Performs deepest-first recursive translation of the AST.
        """
        # Deepest-first: traverse children first
        for child in node.children:
            if child.children:
                self.translate_node(child, root)
                
        # Translate this node if it represents a block with children,
        # or if it's the root node with code.
        is_block = bool(node.children)
        is_root = (node.indent == -1)
        
        if is_block or is_root:
            chunk_with_stickers = self.get_node_code_with_stickers(node)
            if not chunk_with_stickers.strip():
                return
                
            raw_code = self.get_raw_code(node)
            cached_translation = self.canonicalizer.get_translation(raw_code)
            
            if cached_translation:
                node.translation = cached_translation
            else:
                # Canonicalize the chunk with stickers so the LLM receives abstracted code
                abstracted_chunk = self.canonicalizer.canonicalize(chunk_with_stickers, discover=False)
                translation = self.query_ollama(abstracted_chunk)
                node.translation = translation
                self.canonicalizer.add_translation(raw_code, translation)
            
            # Save iteration
            if self.output_dir and self.opt_id:
                self.iteration += 1
                class_dir = self.output_dir / f"class_{self.opt_id}"
                class_dir.mkdir(parents=True, exist_ok=True)
                iter_file = class_dir / f"iteration_{self.iteration}.md"
                current_state = self.get_full_tree_state(root)
                with open(iter_file, 'w', encoding='utf-8') as f:
                    f.write(current_state)
    def translate_code(self, code: str) -> str:
        """
        Entrypoint for translating a full code snippet.
        """
        self.iteration = 0
        root = self.build_tree(code)
        self.translate_node(root, root)
        return root.translation if root.translation else "No code to translate."
