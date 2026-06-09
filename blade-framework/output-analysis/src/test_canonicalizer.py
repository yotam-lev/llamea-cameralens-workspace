import unittest
import tempfile
import shutil
import json
import threading
from pathlib import Path
import sys

# Add the current directory to path to import canonicalizer
sys.path.append(str(Path(__file__).parent))

from canonicalizer import Canonicalizer

class TestCanonicalizer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_operation_standardization(self):
        canonicalizer = Canonicalizer(self.temp_dir, "test_class")
        code = "y = np.clip(x, 0, 1)\nz = jnp.clip(y, 2, 3)\nw = CLIP(z)\nval = x.astype(int)\nreturn w + val\n"
        result = canonicalizer.canonicalize(code)
        print(f"this is the result: {result}")
        
        # Verify OP_BOUND and OP_TYPECAST replacements
        self.assertIn("[OP_BOUND](", result)
        self.assertIn("[OP_TYPECAST]", result)
        
    def test_input_pinning_and_discovery(self):
        canonicalizer = Canonicalizer(self.temp_dir, "class_abc")
        # eval_x and low_bound are inputs (read before defined)
        # y and result are internal variables
        code = "y = [OP_BOUND](eval_x, low_bound, 5)\nresult = y + 10\n"
        tokenized = canonicalizer.canonicalize(code)
        
        # Verify JSON file has been written
        json_path = self.temp_dir / "analysis" / "ind_variables.json"
        self.assertTrue(json_path.exists())
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Pinned inputs should be eval_x, low_bound. Discovered should be y, result.
        original_vars = [item["original_var_name"] for item in data]
        self.assertIn("eval_x", original_vars)
        self.assertIn("low_bound", original_vars)
        self.assertIn("y", original_vars)
        self.assertIn("result", original_vars)
        
        # Verify that class_source is correct
        for item in data:
            self.assertEqual(item["class_source"], "class_abc")
            self.assertTrue(item["line_id"].startswith("class_abc_line#"))

    def test_logical_reduction_copy_propagation(self):
        # y = eval_x should propagate eval_x downstream and delete the line
        canonicalizer = Canonicalizer(self.temp_dir, "class_cp")
        code = "def optimize(eval_x):\n    y = eval_x\n    z = np.clip(y, 0, 5)\n    return z\n"
        result = canonicalizer.canonicalize(code)
        
        # After copy propagation:
        # y = eval_x is pruned, y is replaced by eval_x
        # The line containing y = eval_x should be gone
        self.assertNotIn("[VAR_1] = [VAR_0]", result)
        
    def test_logical_reduction_dead_pruning(self):
        canonicalizer = Canonicalizer(self.temp_dir, "class_dp")
        # z is assigned but never used, so it should be pruned entirely
        code = "def optimize(eval_x):\n    y = np.clip(eval_x, 0, 5)\n    z = y + 1\n    return y\n"
        result = canonicalizer.canonicalize(code)
        
        # z = y + 1 should be gone because z is never read or returned
        # [VAR_2] is the token for z (eval_x -> VAR_0, y -> VAR_1, z -> VAR_2)
        self.assertNotIn("[VAR_2]", result)
        self.assertIn("[VAR_1]", result)
        
    def test_thread_safety(self):
        from canonicalizer import save_variable_mappings
        output_file = self.temp_dir / "analysis" / "ind_variables.json"
        
        threads = []
        num_threads = 10
        num_items_per_thread = 5
        
        def worker(thread_idx):
            mappings = []
            for i in range(num_items_per_thread):
                mappings.append({
                    "id": f"t{thread_idx}_item_{i}",
                    "token": f"[VAR_{thread_idx}_{i}]",
                    "original_var_name": f"var_{thread_idx}_{i}",
                    "class_source": "thread_class",
                    "line_id": f"thread_class_line#{i}"
                })
            save_variable_mappings(output_file, mappings)
            
        for t_idx in range(num_threads):
            t = threading.Thread(target=worker, args=(t_idx,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Read file and assert correct count of items
        with open(output_file, "r") as f:
            data = json.load(f)
            
        self.assertEqual(len(data), num_threads * num_items_per_thread)

if __name__ == "__main__":
    unittest.main()
