import os
import sys
from dotenv import load_dotenv

# Add blade-framework to path to import iohblade
sys.path.append(os.path.join(os.getcwd(), "blade-framework"))

from iohblade.llm import Gemini_LLM

load_dotenv("blade-framework/.env")
api_key = os.getenv("GEMINI_API_KEY")

def test_fix():
    llm = Gemini_LLM(api_key=api_key, model="gemini-2.5-flash")
    
    # Test with history
    session = [
        {"role": "user", "content": "Hello, my name is Yotam."},
        {"role": "assistant", "content": "Hello Yotam! How can I help you today?"}
    ]
    last_msg = {"role": "user", "content": "What is my name?"}
    session.append(last_msg)
    
    print(f"Testing Gemini_LLM with history...")
    try:
        response = llm.query(session)
        print(f"✅ Gemini Response: {response}")
    except Exception as e:
        print(f"❌ Gemini Failed: {e}")

if __name__ == "__main__":
    test_fix()
