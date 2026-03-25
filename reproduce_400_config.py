import os
from google import genai
from dotenv import load_dotenv

load_dotenv("blade-framework/.env")
api_key = os.getenv("GEMINI_API_KEY")

def reproduce():
    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.5-flash"
    
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    print(f"Testing with empty history and generation_config")
    try:
        chat = client.chats.create(model=model_id, history=[], config=generation_config)
        response = chat.send_message("Respond with only: OK")
        print(f"✅ Success: {response.text}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    reproduce()
