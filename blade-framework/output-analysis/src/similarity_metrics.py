import os
import difflib
from typing import Dict, List, Tuple

try:
    from src.translation_engine import TranslationEngine
except ImportError:
    from translation_engine import TranslationEngine

class PseudocodeSimilarityAnalyzer:
    def __init__(self, class_files: Dict[str, str], threshold: int = 85):
        self.class_files = class_files
        self.threshold = threshold / 100.0 if threshold > 1 else threshold
        self.engine = TranslationEngine(similarity_threshold=self.threshold)
        
        # Mapping from opt_id to a list of Line IDs
        self.class_sequences: Dict[str, List[int]] = {}
        
        self._parse_files()

    def _parse_files(self):
        for opt_id, filepath in self.class_files.items():
            if not os.path.exists(filepath):
                self.class_sequences[opt_id] = []
                continue
                
            sequence = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith('// :::'):
                        continue
                    # Remove the '// ' prefix if present
                    if stripped.startswith('// '):
                        stripped = stripped[3:]
                    elif stripped.startswith('//'):
                        stripped = stripped[2:]
                        
                    line_id = self.engine.process_line(stripped)
                    sequence.append(line_id)
                    
            self.class_sequences[opt_id] = sequence

    def get_line_id_mapping(self) -> Dict[str, int]:
        """Returns the dictionary mapping normalized lines to their ID."""
        return self.engine.normalized_to_id

    def get_sequence_similarity_matrix(self) -> Tuple[List[str], List[List[float]]]:
        """
        Computes pairwise similarity matrices for the sequences of Line IDs.
        Returns a tuple of (class_ids, similarity_matrix).
        """
        class_ids = sorted(list(self.class_sequences.keys()))
        n = len(class_ids)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                elif i < j:
                    seq1 = self.class_sequences[class_ids[i]]
                    seq2 = self.class_sequences[class_ids[j]]
                    
                    if not seq1 and not seq2:
                        ratio = 1.0
                    elif not seq1 or not seq2:
                        ratio = 0.0
                    else:
                        matcher = difflib.SequenceMatcher(None, seq1, seq2)
                        ratio = matcher.ratio()
                        
                    matrix[i][j] = ratio
                    matrix[j][i] = ratio
                    
        return class_ids, matrix

def analyze_pseudocode_similarities(class_files: Dict[str, str], threshold: int = 85) -> PseudocodeSimilarityAnalyzer:
    return PseudocodeSimilarityAnalyzer(class_files, threshold)
