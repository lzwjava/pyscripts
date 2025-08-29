import requests
import os

# Get the API key from environment variable or replace with your key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise Exception("OPENROUTER_API_KEY environment variable is not set or is empty")


MODEL_MAPPING = {
    "claude-opus": "anthropic/claude-opus-4",
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "gemini-flash": "google/gemini-2.5-flash",
    "gemini-pro": "google/gemini-2.5-pro",
    "kimi-k2": "moonshotai/kimi-k2",
    "deepseek-v3": "deepseek/deepseek-chat-v3-0324",
    "deepseek-v3.1":"deepseek/deepseek-chat-v3.1",
    "mistral-medium": "mistralai/mistral-medium-3.1",
    "qwen-coder":"qwen/qwen3-coder",
    "gpt-oss": "openai/gpt-oss-120b",
    "gpt-5": "openai/gpt-5"
}


def call_openrouter_api_with_messages(messages, model="mistral-medium", debug=False):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Check if the model exists in the mapping
    if model not in MODEL_MAPPING:
        raise Exception(f"Model '{model}' not found in MODEL_MAPPING")

    data = {"model": MODEL_MAPPING[model], "messages": messages}
    
    if debug:
        print(f"Request URL: {url}")
        print(f"Request Data: {data}")
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if debug:
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")


def call_openrouter_api(prompt, model="mistral-medium", debug=False):
    messages = [{"role": "user", "content": prompt}]
    return call_openrouter_api_with_messages(messages, model, debug)


if __name__ == "__main__":
    import sys
    import subprocess

    # Read from stdin if piped, else try macOS clipboard via pbpaste, else use default prompt
    if not sys.stdin.isatty():
        content = sys.stdin.read()
    else:
        try:
            content = subprocess.check_output(["pbpaste"], text=True)
        except Exception:
            content = "Hello, 9.11 or 9.9, which one is bigger? "

    if not content or not content.strip():
        content = "Hello, 9.11 or 9.9, which one is bigger? "

    model = os.getenv("OPENROUTER_MODEL", "mistral-medium")

    try:
        result = call_openrouter_api(content, model, debug=True)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
