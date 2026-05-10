import time
import os
import sys

# Add framework to path
_FRAMEWORK_ROOT = os.path.dirname(os.path.abspath(__file__))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from iohblade.llm import LLM, Multi_LLM, RateLimited_LLM

class MockGemini(LLM):
    def __init__(self, key):
        super().__init__(key, "mock-gemini")
        self.key = key
    def _query(self, session, **kwargs):
        print(f"  [MockGemini] Querying with key: {self.key}")
        return "OK"

def test_rotation_and_rate_limit():
    print("--- Starting Rotation and Rate Limit Test ---")
    
    llm1 = MockGemini("KEY_1")
    llm2 = MockGemini("KEY_2")
    
    # Strictly alternating
    multi = Multi_LLM([llm1, llm2])
    
    # Rate limit: 6 calls per minute (1 call every 10 seconds on average)
    # To speed up test, let's test 10 calls in 10 seconds limit (60 calls/min)
    # But wait, I want to verify it ACTUALLY waits.
    # Let's use 12 calls per minute (1 call every 5 seconds).
    limited = RateLimited_LLM(multi, calls_per_minute=6)
    
    start_time = time.time()
    for i in range(8):
        elapsed = time.time() - start_time
        print(f"\nCall {i+1} at T+{elapsed:.2f}s")
        limited.query([{"role": "user", "content": "hello"}])
    
    total_time = time.time() - start_time
    print(f"\nTotal time for 8 calls: {total_time:.2f}s")
    
    # With 6 calls per minute, 8 calls should take at least ~10-20 seconds 
    # (Call 1-6 immediate, Call 7 waits for Call 1 to be 60s old... wait)
    # If limit is 6/min:
    # C1: 0s
    # C2: 0s
    # C3: 0s
    # C4: 0s
    # C5: 0s
    # C6: 0s
    # C7: must wait until C1 is 60s old. So T+60s.
    # C8: must wait until C2 is 60s old. So T+60s.
    
    # For a QUICK test, let's use a very high rate limit but verify alternation.
    print("\n--- Testing Strict Alternation ---")
    multi_test = Multi_LLM([MockGemini("A"), MockGemini("B")])
    for i in range(4):
        # We access _pick_llm directly or just query
        res = multi_test._pick_llm()
        print(f"Pick {i+1}: {res.key}")
        expected = "A" if i % 2 == 0 else "B"
        assert res.key == expected

    print("\n✅ Rotation test passed!")

if __name__ == "__main__":
    test_rotation_and_rate_limit()
