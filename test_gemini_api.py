import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv

def debug_env():
    print(f"--- Debugging Environment ---")
    print(f"Current Working Directory: {os.getcwd()}")
    
    # Check for .env in current and parent dirs
    search_paths = [Path("."), Path("blade-framework"), Path("..")]
    found_env = None
    
    for p in search_paths:
        env_path = p / ".env"
        if env_path.exists():
            print(f"✅ Found .env at: {env_path.absolute()}")
            found_env = env_path
            break
    
    if found_env:
        # Try loading it
        load_dotenv(found_env)
        
        # Manually parse it as a fallback if os.getenv fails
        with open(found_env, 'r') as f:
            content = f.read()
            match = re.search(r'GEMINI_API_KEY\s*=\s*["\']?([^"\']+)["\']?', content)
            if match:
                key = match.group(1).strip()
                os.environ["GEMINI_API_KEY"] = key
                print(f"✅ Manually extracted key: {key[:5]}...{key[-5:]}")
    else:
        print("❌ No .env file found in search paths.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Final Result: GEMINI_API_KEY is still EMPTY.")
        return None
    
    print(f"✅ GEMINI_API_KEY successfully loaded.")
    return api_key

def test_gemini(api_key):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("❌ Error: google-genai library not installed.")
        return

    # 1. LIST MODELS
    print("\n--- Listing Available Models ---")
    try:
        client = genai.Client(api_key=api_key)
        for model in client.models.list():
            print(f"Name: {model.name}")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")

    # 2. TEST CONFIGS
    test_configs = [
        {"model": "gemini-2.5-flash", "version": "v1"},
        {"model": "gemini-2.0-flash", "version": None}, 
        {"model": "gemini-3-flash-preview", "version": None}
    ]

    for config in test_configs:
        model_id = config["model"]
        version = config["version"]
        
        print(f"\n--- Testing Model: {model_id} (Version: {version}) ---")
        try:
            # Replicate the fix I added to llm.py
            http_options = {'api_version': version} if version else None
            client = genai.Client(api_key=api_key, http_options=http_options)
            
            # Simple non-chat test
            print(f"Testing simple generation...")
            response = client.models.generate_content(
                model=model_id,
                contents="Say 'API is working' and nothing else."
            )
            print(f"✅ Simple Response: {response.text.strip()}")

            # Chat test with Role Mapping fix (assistant -> model)
            print(f"Testing chat history with role mapping...")
            chat = client.chats.create(
                model=model_id,
                history=[
                    {"role": "user", "parts": [{"text": "Hello"}]},
                    {"role": "model", "parts": [{"text": "Hi there!"}]} # Gemini role
                ]
            )
            response = chat.send_message("What was the first thing I said?")
            print(f"✅ Chat Response: {response.text.strip()}")

        except Exception as e:
            print(f"❌ Failed for {model_id}:")
            print(f"Error Message: {str(e)}")
            if "404" in str(e):
                print("💡 Analysis: This specific configuration (Model+Version) is NOT found.")

if __name__ == "__main__":
    key = debug_env()
    if key:
        test_gemini(key)
