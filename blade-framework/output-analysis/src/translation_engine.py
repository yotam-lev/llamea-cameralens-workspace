import re
import requests
import difflib
from typing import Dict, Optional, Tuple

class TranslationEngine:
    """
    A text-processing engine to regularize pseudocode extracted from optimization classes.
    It normalizes strings using basic regex/spacing rules, applies known domain-specific
    synonyms, and leverages a local LLM to verify semantic equivalence for strings with high similarity.
    """

    OLLAMA_URL = "http://localhost:11434/api/chat"
    DEFAULT_MODEL = "qwen2.5-coder:14b"
    FALLBACK_MODEL = "mistral:latest"

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initializes the TranslationEngine with a similarity threshold.
        
        Args:
            similarity_threshold (float): The difflib ratio threshold (0.0 to 1.0) above which
                                          the LLM is queried to confirm semantic equivalence.
        """
        self.similarity_threshold = similarity_threshold
        
        # Core data structures
        # Maps observed normalized line -> Line_ID
        self.normalized_to_id: Dict[str, int] = {}
        
        # Maps Line_ID -> Canonical_Pseudocode_String (the first observed version of a semantic line)
        self.id_to_canonical: Dict[int, str] = {}
        
        self._next_id: int = 1

        # Domain-specific synonym dictionary (can be extended)
        self.synonyms: Dict[str, str] = {
            "CAST": "CONVERT_TO_INT",
            "INTEGER": ""  # Handle second arg of CAST if needed, or rely on LLM
        }

    def normalize_basic(self, raw_line: str) -> str:
        """
        Performs basic textual normalization using regex and known synonyms.
        Removes extra spacing, normalizes brackets and assignments, and unifies terminology.
        
        Args:
            raw_line (str): The raw pseudocode line.
            
        Returns:
            str: The basic normalized pseudocode line.
        """
        # Strip leading/trailing whitespaces
        line = raw_line.strip()
        
        # Normalize multiple spaces to a single space
        line = re.sub(r'\s+', ' ', line)
        
        # Apply domain-specific synonym replacements (e.g. CAST -> CONVERT_TO_INT)
        for old_term, new_term in self.synonyms.items():
            if new_term:
                line = re.sub(rf'\b{old_term}\b', new_term, line)
        
        # Remove spaces around common operators
        # E.g. 'eval_x [ 18 : 24 ] = ...' -> 'eval_x[18:24]=...'
        line = re.sub(r'\s*([=\+\-\*/\[\]\(\),:])\s*', r'\1', line)
        
        return line

    def _call_llm(self, prompt: str, model: str = DEFAULT_MODEL) -> Optional[str]:
        """
        Queries the local Ollama LLM to verify semantic equivalence.
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            response = requests.post(self.OLLAMA_URL, json=payload, timeout=15)
            response.raise_for_status()
            return response.json().get('message', {}).get('content', '').strip()
        except requests.exceptions.RequestException as e:
            print(f"LLM call failed with {model}: {e}")
            if model == self.DEFAULT_MODEL:
                print("Falling back to fallback model...")
                return self._call_llm(prompt, model=self.FALLBACK_MODEL)
            return None

    def _verify_semantic_equivalence(self, new_line: str, candidate_line: str) -> Optional[Tuple[str, str]]:
        """
        Verifies if two syntactically different lines are semantically identical.
        Returns a tuple of (new_term, canonical_term) if identical and a specific synonym
        was identified, or ("", "") if identical but no simple synonym exists.
        Returns None if they are NOT identical.
        """
        prompt = (
            "You are an expert code analyzer. Are the following two lines of pseudocode "
            "semantically identical (i.e., performing the exact same logic)?\n"
            f"Line 1 (New): {new_line}\n"
            f"Line 2 (Canonical): {candidate_line}\n"
            "If they are NOT identical, answer strictly with 'NO'.\n"
            "If they ARE identical, identify the specific differing term that makes them different "
            "(e.g., 'CAST' and 'CONVERT_TO_INT') and answer strictly in the format 'YES|NewTerm|CanonicalTerm'. "
            "If the difference is structural and no simple term substitution applies, answer 'YES||'."
        )
        response = self._call_llm(prompt)
        if response:
            parts = response.split('|')
            if parts[0].upper().startswith('YES'):
                if len(parts) >= 3 and parts[1].strip() and parts[2].strip():
                    return (parts[1].strip(), parts[2].strip())
                return ("", "")
        return None

    def process_line(self, raw_line: str) -> int:
        """
        Processes a raw string, assigns or retrieves a Line_ID, and maintains state.
        
        Args:
            raw_line (str): The raw line of pseudocode.
            
        Returns:
            int: The unique Line_ID for the semantic meaning of this line.
        """
        normalized_line = self.normalize_basic(raw_line)
        
        # 1. Exact string match in existing normalized lines (fast path)
        if normalized_line in self.normalized_to_id:
            return self.normalized_to_id[normalized_line]
            
        # 2. Similarity search against existing canonical lines
        best_match = None
        highest_ratio = 0.0
        
        for canonical_id, canonical_line in self.id_to_canonical.items():
            ratio = difflib.SequenceMatcher(None, normalized_line, canonical_line).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = canonical_line
                
        # 3. If similarity is above threshold, use LLM to verify semantic match
        if best_match and highest_ratio >= self.similarity_threshold:
            synonym_pair = self._verify_semantic_equivalence(normalized_line, best_match)
            if synonym_pair is not None:
                new_term, canonical_term = synonym_pair
                if new_term and canonical_term:
                    # Dynamically add the translation to the synonyms dictionary
                    self.synonyms[new_term] = canonical_term
                
                # Match confirmed! Link this new string variation to the existing Line_ID
                line_id = self.normalized_to_id[best_match]
                self.normalized_to_id[normalized_line] = line_id
                return line_id
                
        # 4. No match found, assign new Line_ID
        new_id = self._next_id
        self._next_id += 1
        
        self.normalized_to_id[normalized_line] = new_id
        # We record the first appearance of a line as its "Canonical" representation
        self.id_to_canonical[new_id] = normalized_line
        
        return new_id