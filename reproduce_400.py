import os
from google import genai
from dotenv import load_dotenv

load_dotenv("blade-framework/.env")
api_key = os.getenv("GEMINI_API_KEY")

def reproduce():
    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.5-flash"
    
    print(f"Testing with parts: [string] (as in llm.py)")
    try:
        # This mimics llm.py's history construction
        history = [
            {"role": "user", "parts": ["Hello"]},
            {"role": "model", "parts": ["Hi there!"]}
        ]
        chat = client.chats.create(model=model_id, history=history)
        response = chat.send_message("How are you?")
        print(f"✅ Success with [string]: {response.text}")
    except Exception as e:
        print(f"❌ Failed with [string]: {e}")

    print(f"\nTesting with parts: [{{'text': string}}] (as in test_gemini_api.py)")
    try:
        history = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]}
        ]
        chat = client.chats.create(model=model_id, history=history)
        response = chat.send_message("How are you?")
        print(f"✅ Success with [{{'text': string}}]: {response.text}")
    except Exception as e:
        print(f"❌ Failed with [{{'text': string}}]: {e}")

if __name__ == "__main__":
    reproduce()
