import os
from google import genai
from dotenv import load_dotenv

load_dotenv("blade-framework/.env")
api_key = os.getenv("GEMINI_API_KEY")

def reproduce():
    # Force api_version='v1'
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
    model_id = "gemini-2.5-flash"
    
    print(f"Testing with api_version='v1'")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents="Respond with only: OK"
        )
        print(f"✅ Success with v1: {response.text}")
    except Exception as e:
        print(f"❌ Failed with v1: {e}")

    # Testing with api_version='v1beta' (default for google-genai usually)
    client_beta = genai.Client(api_key=api_key) # Default should be fine
    print(f"\nTesting with default (v1beta)")
    try:
        response = client_beta.models.generate_content(
            model=model_id,
            contents="Respond with only: OK"
        )
        print(f"✅ Success with default: {response.text}")
    except Exception as e:
        print(f"❌ Failed with default: {e}")

if __name__ == "__main__":
    reproduce()
