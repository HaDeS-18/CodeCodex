"""Dataset loading and processing"""

import pandas as pd
from datasets import load_dataset
import streamlit as st

@st.cache_data
def load_humaneval():
    """Load HumanEval dataset"""
    try:
        dataset = load_dataset("openai_humaneval", split="test")
        return dataset.to_pandas()
    except Exception as e:
        st.error(f"Error loading HumanEval: {e}")
        return pd.DataFrame()

@st.cache_data  
def load_mbpp():
    """Load MBPP dataset"""
    try:
        dataset = load_dataset("mbpp", split="test")
        return dataset.to_pandas()
    except Exception as e:
        st.error(f"Error loading MBPP: {e}")
        return pd.DataFrame()

@st.cache_data
def load_codexglue():
    """Load CodeXGLUE dataset"""
    try:
        dataset = load_dataset("microsoft/CodeXGLUE-TT-text-to-text", split="test")
        return dataset.to_pandas()
    except Exception as e:
        st.error(f"Error loading CodeXGLUE: {e}")
        return pd.DataFrame()

def get_dataset_info(df: pd.DataFrame, name: str) -> dict:
    """Get basic dataset information"""
    if df.empty:
        return {"name": name, "size": 0, "columns": []}
    
    return {
        "name": name,
        "size": len(df),
        "columns": list(df.columns),
        "sample": df.head(3).to_dict('records') if len(df) > 0 else []
    }