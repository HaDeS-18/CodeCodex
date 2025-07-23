"""Configuration settings for the project"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# API Configuration
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Model Configuration - Lightweight versions
MODELS = {
    "CodeLLaMA": "codellama/CodeLlama-7b-Python-hf",  # Keep 7B for now, we'll use quantized version
    "StarCoder": "bigcode/starcoderbase-1b",          # Much smaller 1B version
    "Replit Coder": "replit/replit-code-v1-3b"        # Keep 3B, it's reasonable
}

# Dataset Configuration - Using free alternatives
DATASETS = {
    "CoderEval": "CoderEval/CoderEval",  # Alternative to HumanEval - real-world functions
    "CodeContests": "deepmind/code_contests",  # Competitive programming problems
    "Python Problems": "codeparrot/github-code-clean"  # Clean Python code examples
}

# Generation settings
GENERATION_CONFIG = {
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9
}

# Streamlit config
STREAMLIT_CONFIG = {
    "page_title": "Code Generation & Bug Fixing",
    "page_icon": "⚡",
    "layout": "wide"
}