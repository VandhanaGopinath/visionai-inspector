"""
List the Gemini models your GEMINI_API_KEY can actually call for text generation.

Usage:  python scripts/list_gemini_models.py
Then put a name from the list into .env as  GEMINI_MODEL=<name>
Never prints the key.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rag_generation import get_gemini_client, normalize_model_name  # also loads .env

client = get_gemini_client()
if client is None:
    sys.exit("GEMINI_API_KEY set: False  -- put it in a .env file (see .env.example) or set it in THIS terminal.")
print("GEMINI_API_KEY set: True")
for m in client.models.list():
    if "generateContent" in (getattr(m, "supported_actions", None) or []):
        print(normalize_model_name(m.name))
