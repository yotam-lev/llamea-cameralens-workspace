import os
import pytest
import tempfile
from pathlib import Path
import sys

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.similarity_metrics import analyze_pseudocode_similarities, PseudocodeSimilarityAnalyzer

def test_identical_classes():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        file1 = temp_dir_path / "opt_1_Iteration_Final.md"
        file2 = temp_dir_path / "opt_2_Iteration_Final.md"
        
        content = """
// :::PSEUDOCODE:::
// x = 5
// y = 10
// z = x + y
// :::END_PSEUDOCODE:::
"""
        with open(file1, 'w') as f: f.write(content)
        with open(file2, 'w') as f: f.write(content)
        
        class_files = {
            "opt_1": str(file1),
            "opt_2": str(file2)
        }
        
        analyzer = analyze_pseudocode_similarities(class_files)
        class_ids, matrix = analyzer.get_sequence_similarity_matrix()
        
        assert len(class_ids) == 2
        assert matrix[0][0] == 1.0
        assert matrix[1][1] == 1.0
        assert matrix[0][1] == 1.0
        assert matrix[1][0] == 1.0

def test_divergent_classes():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        file1 = temp_dir_path / "opt_1_Iteration_Final.md"
        file2 = temp_dir_path / "opt_2_Iteration_Final.md"
        
        content1 = """// x = 5"""
        content2 = """// something_completely_different = 10"""
        
        with open(file1, 'w') as f: f.write(content1)
        with open(file2, 'w') as f: f.write(content2)
        
        class_files = {
            "opt_1": str(file1),
            "opt_2": str(file2)
        }
        
        analyzer = analyze_pseudocode_similarities(class_files, threshold=100) # strict
        class_ids, matrix = analyzer.get_sequence_similarity_matrix()
        
        assert matrix[0][1] == 0.0

def test_missing_files_handled_gracefully():
    class_files = {
        "opt_1": "does_not_exist.md",
        "opt_2": "also_does_not_exist.md"
    }
    
    analyzer = analyze_pseudocode_similarities(class_files)
    class_ids, matrix = analyzer.get_sequence_similarity_matrix()
    
    assert len(class_ids) == 2
    assert matrix[0][1] == 1.0  # Two empty sequences are 100% similar
