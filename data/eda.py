"""EDA and analysis functions"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

def analyze_coder_eval(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze CoderEval dataset"""
    if df.empty:
        return {}
    
    # Basic stats
    analysis = {
        "total_problems": len(df),
        "avg_prompt_length": df['prompt'].str.len().mean() if 'prompt' in df.columns else 0
    }
    
    # Problem difficulty distribution (mock data for demo)
    difficulty_data = {
        'Difficulty': ['Easy', 'Medium', 'Hard'],
        'Count': [45, 89, 30]
    }
    
    fig = px.bar(
        difficulty_data, 
        x='Difficulty', 
        y='Count',
        title='CoderEval Problem Difficulty Distribution',
        color='Difficulty'
    )
    
    analysis['difficulty_chart'] = fig
    return analysis

def analyze_code_contests(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze CodeContests dataset"""
    if df.empty:
        return {}
    
    analysis = {
        "total_problems": len(df),
        "avg_text_length": df['text'].str.len().mean() if 'text' in df.columns else 0
    }
    
    # Code complexity distribution (mock data)
    complexity_data = {
        'Lines of Code': ['1-10', '11-20', '21-30', '31+'],
        'Count': [245, 387, 234, 134]
    }
    
    fig = px.pie(
        complexity_data,
        values='Count',
        names='Lines of Code', 
        title='CodeContests Problem Complexity Distribution'
    )
    
    analysis['complexity_chart'] = fig
    return analysis

def analyze_python_problems(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze Python Problems dataset"""
    if df.empty:
        return {}
    
    analysis = {
        "total_samples": len(df)
    }
    
    # Task distribution (mock data)
    task_data = {
        'Task': ['Code Generation', 'Code Summarization', 'Code Translation', 'Bug Detection'],
        'Samples': [2880, 2134, 1780, 1456]
    }
    
    fig = px.bar(
        task_data,
        x='Task',
        y='Samples',
        title='Python Problems Difficulty Distribution'
    )
    fig.update_layout(xaxis_tickangle=45)
    
    analysis['task_chart'] = fig
    return analysis

def create_overview_charts():
    """Create overview comparison charts"""
    
    # Dataset size comparison
    dataset_sizes = {
        'Dataset': ['CoderEval', 'CodeContests', 'Python Problems'],
        'Size': [164, 1000, 8000],
        'Type': ['Evaluation', 'Training', 'Multi-task']
    }
    
    fig = px.bar(
        dataset_sizes,
        x='Dataset',
        y='Size',
        color='Type',
        title='Dataset Size Comparison'
    )
    
    return fig