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

# Model Configuration
MODELS = {
    "CodeLLaMA": "codellama/CodeLlama-7b-Python-hf",
    "StarCoder": "bigcode/starcoder", 
    "Replit Coder": "replit/replit-code-v1-3b"
}

# Dataset Configuration
DATASETS = {
    "HumanEval": "openai_humaneval",
    "MBPP": "mbpp", 
    "CodeXGLUE": "microsoft/CodeXGLUE-TT-text-to-text"
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