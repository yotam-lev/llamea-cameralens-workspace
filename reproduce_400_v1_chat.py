import os
from google import genai
from dotenv import load_dotenv

load_dotenv("blade-framework/.env")
api_key = os.getenv("GEMINI_API_KEY")

def reproduce():
    # Force api_version='v1'
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
    model_id = "gemini-2.5-flash"
    
    print(f"Testing chat with api_version='v1' and history")
    try:
        history = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]}
        ]
        chat = client.chats.create(model=model_id, history=history)
        response = chat.send_message("How are you?")
        print(f"✅ Success with v1 chat: {response.text}")
    except Exception as e:
        print(f"❌ Failed with v1 chat: {e}")

if __name__ == "__main__":
    reproduce()
