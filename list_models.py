#!/usr/bin/env python3
"""List available Gemini models"""

from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Available Gemini Models:")
print("=" * 60)

try:
    for model in client.models.list():
        print(f"\nName: {model.name}")
        if hasattr(model, 'display_name'):
            print(f"Display Name: {model.display_name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"Methods: {model.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {e}")
