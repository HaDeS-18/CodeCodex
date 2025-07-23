"""Helper utility functions"""

import streamlit as st
from typing import Dict, Any

def display_model_info(model_name: str, is_loaded: bool):
    """Display model status in sidebar"""
    status = "✅ Loaded" if is_loaded else "❌ Not Loaded"
    st.sidebar.write(f"**{model_name}**: {status}")

def format_code_output(code: str) -> str:
    """Format and clean generated code"""
    # Remove extra whitespace and clean up
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines at the beginning
        if not cleaned_lines and not line.strip():
            continue
        cleaned_lines.append(line.rstrip())
    
    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)

def create_model_comparison_table(results: Dict[str, str]) -> None:
    """Create a comparison table for model outputs"""
    if not results:
        return
    
    st.subheader("Model Comparison")
    
    for model_name, output in results.items():
        with st.expander(f"{model_name} Output"):
            st.code(output, language='python')

def estimate_code_quality(code: str) -> Dict[str, Any]:
    """Simple code quality estimation"""
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    quality_metrics = {
        "lines_of_code": len(non_empty_lines),
        "has_functions": "def " in code,
        "has_classes": "class " in code,
        "has_comments": "#" in code or '"""' in code,
        "estimated_complexity": "Low" if len(non_empty_lines) < 10 else "Medium" if len(non_empty_lines) < 30 else "High"
    }
    
    return quality_metrics