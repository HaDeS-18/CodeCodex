"""
Interactive EDA App for CodeCodex Datasets
Advanced exploratory data analysis of CoderEval, CodeContests, and Python Problems datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from textstat import flesch_reading_ease, flesch_kincaid_grade
import re
from collections import Counter
import ast
from data.datasets import load_coder_eval, load_code_contests, load_python_problems

# Page configuration
st.set_page_config(
    page_title="CodeCodex EDA Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_datasets():
    """Load all three datasets and combine them for analysis"""
    coder_eval = load_coder_eval()
    code_contests = load_code_contests()
    python_problems = load_python_problems()
    
    # Add dataset source column
    coder_eval['dataset'] = 'CoderEval'
    code_contests['dataset'] = 'CodeContests'
    python_problems['dataset'] = 'Python Problems'
    
    return coder_eval, code_contests, python_problems

@st.cache_data
def extract_text_features(df, text_column):
    """Extract various text features for analysis"""
    features = []
    
    for text in df[text_column]:
        if pd.isna(text):
            text = ""
        
        # Basic text statistics
        char_count = len(text)
        word_count = len(text.split())
        line_count = text.count('\n') + 1
        
        # Code-specific features
        function_count = text.count('def ')
        class_count = text.count('class ')
        import_count = text.count('import ') + text.count('from ')
        comment_count = text.count('#') + text.count('"""') + text.count("'''")
        
        # Complexity indicators
        if_count = text.count('if ') + text.count('elif ')
        loop_count = text.count('for ') + text.count('while ')
        try_count = text.count('try:') + text.count('except')
        
        # Readability scores
        try:
            readability = flesch_reading_ease(text)
            grade_level = flesch_kincaid_grade(text)
        except:
            readability = 0
            grade_level = 0
        
        features.append({
            'char_count': char_count,
            'word_count': word_count,
            'line_count': line_count,
            'function_count': function_count,
            'class_count': class_count,
            'import_count': import_count,
            'comment_count': comment_count,
            'if_count': if_count,
            'loop_count': loop_count,
            'try_count': try_count,
            'readability': readability,
            'grade_level': grade_level
        })
    
    return pd.DataFrame(features)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 CodeCodex EDA Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("**Comprehensive analysis of CoderEval, CodeContests, and Python Problems datasets**")
    
    # Load datasets
    with st.spinner("Loading datasets..."):
        coder_eval, code_contests, python_problems = load_all_datasets()
    
    # Sidebar for navigation and filters
    st.sidebar.header("🔍 Analysis Controls")
    
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["📈 Overview & Statistics", "🔍 Dataset Comparison", "📝 Text Analysis", 
         "🎯 Interactive Exploration"]
    )
    
    # Dataset selection
    selected_datasets = st.sidebar.multiselect(
        "Select Datasets:",
        ["CoderEval", "CodeContests", "Python Problems"],
        default=["CoderEval", "CodeContests", "Python Problems"]
    )
    
    # Filter datasets based on selection
    datasets = {}
    if "CoderEval" in selected_datasets:
        datasets["CoderEval"] = coder_eval
    if "CodeContests" in selected_datasets:
        datasets["CodeContests"] = code_contests
    if "Python Problems" in selected_datasets:
        datasets["Python Problems"] = python_problems
    
    if not datasets:
        st.warning("Please select at least one dataset from the sidebar.")
        return
    
    # Main analysis sections
    if analysis_type == "📈 Overview & Statistics":
        show_overview_statistics(datasets)
    elif analysis_type == "🔍 Dataset Comparison":
        show_dataset_comparison(datasets)
    elif analysis_type == "📝 Text Analysis":
        show_text_analysis(datasets)
    elif analysis_type == "🎯 Interactive Exploration":
        show_interactive_exploration(datasets)

def show_overview_statistics(datasets):
    """Display overview statistics for selected datasets"""
    st.header("📈 Dataset Overview & Statistics")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_problems = sum(len(df) for df in datasets.values())
    avg_problems = total_problems / len(datasets) if datasets else 0
    
    with col1:
        st.metric("Total Problems", total_problems)
    with col2:
        st.metric("Datasets Selected", len(datasets))
    with col3:
        st.metric("Avg Problems/Dataset", f"{avg_problems:.1f}")
    with col4:
        st.metric("Analysis Features", "12+")
    
    # Dataset size comparison
    st.subheader("📊 Dataset Sizes")
    
    size_data = []
    for name, df in datasets.items():
        size_data.append({"Dataset": name, "Problems": len(df)})
    
    size_df = pd.DataFrame(size_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(size_df, x="Dataset", y="Problems", 
                     title="Number of Problems per Dataset",
                     color="Dataset")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(size_df, values="Problems", names="Dataset",
                     title="Dataset Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed statistics for each dataset
    st.subheader("📋 Detailed Dataset Statistics")
    
    for name, df in datasets.items():
        with st.expander(f"📊 {name} Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Info:**")
                st.write(f"- Total Problems: {len(df)}")
                st.write(f"- Columns: {', '.join(df.columns)}")
                
                # Text length analysis
                if 'prompt' in df.columns:
                    text_col = 'prompt'
                elif 'text' in df.columns:
                    text_col = 'text'
                elif 'nl' in df.columns:
                    text_col = 'nl'
                else:
                    text_col = None
                
                if text_col:
                    avg_length = df[text_col].str.len().mean()
                    st.write(f"- Avg {text_col.title()} Length: {avg_length:.1f} chars")
            
            with col2:
                st.write("**Sample Data:**")
                st.dataframe(df.head(2), use_container_width=True)
            
            # Add explanatory text for each dataset
            st.markdown("---")
            st.markdown("### 📖 **Understanding This Dataset**")
            
            if name == "CoderEval":
                st.markdown("""
                **What is CoderEval?**  
                CoderEval is our alternative to the HumanEval dataset. It contains programming problems similar to those used to evaluate AI coding models.
                
                **Key Features:**
                - **task_id**: Unique identifier for each problem (e.g., "CoderEval/0")
                - **prompt**: The problem description with function signature and examples
                - **canonical_solution**: The correct solution to the problem
                - **test**: Test cases to verify the solution works correctly
                - **entry_point**: The main function name to be implemented
                
                **Why This Matters:**  
                This dataset helps us understand what types of coding problems are commonly used to test AI models. The problems range from simple list operations to more complex algorithmic challenges.
                """)
            
            elif name == "CodeContests":
                st.markdown("""
                **What is CodeContests?**  
                CodeContests contains competitive programming-style problems, similar to those found in coding competitions.
                
                **Key Features:**
                - **task_id**: Unique identifier for each contest problem
                - **text**: Problem description in natural language
                - **code**: The solution code in Python
                - **test_list**: List of test cases to validate solutions
                - **test_setup_code**: Any setup code needed for testing
                
                **Why This Matters:**  
                These problems test fundamental programming skills like basic arithmetic, string manipulation, and simple algorithms. They're great for beginners learning to code.
                """)
            
            elif name == "Python Problems":
                st.markdown("""
                **What is Python Problems?**  
                Python Problems is a curated collection of programming exercises covering common algorithms and data structures.
                
                **Key Features:**
                - **id**: Unique identifier for each exercise
                - **nl**: Natural language description of what to implement
                - **code**: The Python implementation
                - **difficulty**: Categorized as Easy, Medium, or Hard
                
                **Why This Matters:**  
                This dataset covers essential programming concepts like recursion (factorial, fibonacci), string processing (palindromes), and data manipulation (sorting, filtering). It's perfect for learning fundamental programming patterns.
                """)

def show_dataset_comparison(datasets):
    """Compare datasets across multiple dimensions"""
    st.header("🔍 Dataset Comparison Analysis")
    
    if len(datasets) < 2:
        st.warning("Please select at least 2 datasets for comparison.")
        return
    
    # Feature extraction for comparison
    comparison_data = []
    
    for name, df in datasets.items():
        # Determine text columns
        text_cols = []
        if 'prompt' in df.columns:
            text_cols.append('prompt')
        if 'canonical_solution' in df.columns:
            text_cols.append('canonical_solution')
        if 'code' in df.columns:
            text_cols.append('code')
        if 'text' in df.columns:
            text_cols.append('text')
        if 'nl' in df.columns:
            text_cols.append('nl')
        
        for text_col in text_cols:
            if text_col in df.columns:
                features = extract_text_features(df, text_col)
                features['dataset'] = name
                features['text_type'] = text_col
                comparison_data.append(features)
    
    if not comparison_data:
        st.warning("No suitable text columns found for comparison.")
        return
    
    # Combine all features
    all_features = pd.concat(comparison_data, ignore_index=True)
    
    # Comparison visualizations
    st.subheader("📊 Feature Comparisons")
    
    # Select features to compare
    numeric_features = ['char_count', 'word_count', 'line_count', 'function_count', 
                       'if_count', 'loop_count', 'readability']
    
    selected_feature = st.selectbox("Select feature to compare:", numeric_features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot comparison
        fig = px.box(all_features, x='dataset', y=selected_feature,
                     title=f"{selected_feature.replace('_', ' ').title()} Distribution by Dataset")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Violin plot for distribution shape
        fig = px.violin(all_features, x='dataset', y=selected_feature,
                        title=f"{selected_feature.replace('_', ' ').title()} Distribution Shape")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    
    correlation_features = ['char_count', 'word_count', 'line_count', 'function_count', 
                           'if_count', 'loop_count', 'try_count']
    
    corr_matrix = all_features[correlation_features].corr()
    
    fig = px.imshow(corr_matrix, 
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    
    summary_stats = all_features.groupby('dataset')[numeric_features].agg(['mean', 'std', 'min', 'max'])
    st.dataframe(summary_stats, use_container_width=True)
    
    # Add explanatory text for dataset comparison
    st.markdown("---")
    st.markdown("### 📖 **Understanding Dataset Comparison**")
    st.markdown("""
    **What Are We Comparing?**  
    We're analyzing different characteristics of text and code across our three datasets to understand their complexity and structure.
    
    **Key Metrics Explained:**
    - **char_count**: Total characters in the text - longer texts might be more complex
    - **word_count**: Number of words - indicates verbosity and detail level
    - **line_count**: Lines of code - more lines often mean more complex solutions
    - **function_count**: Number of functions defined - shows code organization
    - **if_count**: Conditional statements - indicates logical complexity
    - **loop_count**: For/while loops - shows iterative complexity
    - **readability**: Flesch reading ease score - higher = easier to read
    
    **How to Interpret the Charts:**
    - **Box plots** show the distribution range - wider boxes mean more variation
    - **Violin plots** show the shape of data distribution - peaks show common values
    - **Correlation heatmap** shows relationships - red means strong positive correlation, blue means negative
    
    **What This Tells Us:**  
    By comparing these metrics, we can see which datasets contain more complex problems, which are more beginner-friendly, and how different types of coding challenges vary in their characteristics.
    """)

def show_text_analysis(datasets):
    """Perform detailed text analysis"""
    st.header("📝 Advanced Text Analysis")
    
    # Dataset selection for text analysis
    dataset_name = st.selectbox("Select dataset for text analysis:", list(datasets.keys()))
    df = datasets[dataset_name]
    
    # Text column selection
    text_columns = []
    if 'prompt' in df.columns:
        text_columns.append('prompt')
    if 'canonical_solution' in df.columns:
        text_columns.append('canonical_solution')
    if 'code' in df.columns:
        text_columns.append('code')
    if 'text' in df.columns:
        text_columns.append('text')
    if 'nl' in df.columns:
        text_columns.append('nl')
    
    if not text_columns:
        st.warning("No text columns found for analysis.")
        return
    
    selected_text_col = st.selectbox("Select text column:", text_columns)
    
    # Extract features
    features_df = extract_text_features(df, selected_text_col)
    
    # Text analysis visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📏 Text Length Distribution")
        fig = px.histogram(features_df, x='char_count', nbins=20,
                          title="Character Count Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔤 Word Count Analysis")
        fig = px.box(features_df, y='word_count',
                     title="Word Count Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Line Count Distribution")
        fig = px.histogram(features_df, x='line_count', nbins=15,
                          title="Line Count Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🎯 Complexity Indicators")
        complexity_data = features_df[['function_count', 'if_count', 'loop_count', 'try_count']].sum()
        fig = px.bar(x=complexity_data.index, y=complexity_data.values,
                     title="Code Complexity Features")
        st.plotly_chart(fig, use_container_width=True)
    
    # Readability analysis
    st.subheader("📖 Readability Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_readability = features_df['readability'].mean()
        st.metric("Avg Readability Score", f"{avg_readability:.1f}")
    
    with col2:
        avg_grade = features_df['grade_level'].mean()
        st.metric("Avg Grade Level", f"{avg_grade:.1f}")
    
    with col3:
        complexity_score = (features_df['line_count'] + features_df['function_count'] * 2).mean()
        st.metric("Avg Complexity Score", f"{complexity_score:.1f}")
    
    # Most/Least complex problems
    st.subheader("🏆 Complexity Rankings")
    
    col1, col2 = st.columns(2)
    
    complexity_scores = (features_df['line_count'] + 
                        features_df['function_count'] * 2 + 
                        features_df['if_count'] + 
                        features_df['loop_count'] * 1.5)
    
    with col1:
        st.write("**Most Complex Problems:**")
        most_complex_idx = complexity_scores.nlargest(3).index
        for i, idx in enumerate(most_complex_idx):
            st.write(f"{i+1}. Problem {idx} (Score: {complexity_scores[idx]:.1f})")
    
    with col2:
        st.write("**Least Complex Problems:**")
        least_complex_idx = complexity_scores.nsmallest(3).index
        for i, idx in enumerate(least_complex_idx):
            st.write(f"{i+1}. Problem {idx} (Score: {complexity_scores[idx]:.1f})")
    
    # Add explanatory text for text analysis
    st.markdown("---")
    st.markdown("### 📖 **Understanding Text Analysis**")
    st.markdown("""
    **What Is Text Analysis?**  
    Text analysis examines the structure and characteristics of the text and code in our datasets to understand their complexity and readability.
    
    **Key Measurements:**
    - **Character/Word Count**: Basic size metrics - longer doesn't always mean harder, but can indicate detail level
    - **Line Count**: For code, more lines often mean more complex logic or detailed implementation
    - **Function Count**: Shows code organization - well-structured code uses functions appropriately
    - **Complexity Indicators**: if/else statements, loops, and try/except blocks show logical complexity
    - **Readability Score**: Flesch Reading Ease - scores 90-100 are very easy, 60-70 are standard, below 30 are very difficult
    - **Grade Level**: Flesch-Kincaid Grade Level - indicates the education level needed to understand the text
    
    **How to Read the Charts:**
    - **Histograms** show how values are distributed - peaks show common values
    - **Box plots** show the range and median - the line in the middle is the average
    - **Bar charts** show totals or counts for different categories
    
    **Complexity Rankings Explained:**
    We calculate a complexity score by combining line count, function count, conditional statements, and loops. Higher scores indicate more complex problems that might be harder for beginners.
    
    **Why This Matters:**  
    Understanding text complexity helps us choose appropriate problems for different skill levels and identify which datasets are best suited for beginners vs. advanced programmers.
    """)


def show_interactive_exploration(datasets):
    """Interactive data exploration interface"""
    st.header("🎯 Interactive Data Exploration")
    
    # Dataset selection
    dataset_name = st.selectbox("Select dataset to explore:", list(datasets.keys()))
    df = datasets[dataset_name]
    
    # Show raw data with filtering
    st.subheader("🔍 Data Explorer")
    
    # Add filters
    col1, col2 = st.columns(2)
    
    with col1:
        if 'task_id' in df.columns:
            selected_problems = st.multiselect(
                "Select specific problems:",
                df['task_id'].tolist(),
                default=df['task_id'].tolist()[:3]
            )
            filtered_df = df[df['task_id'].isin(selected_problems)]
        else:
            filtered_df = df
    
    with col2:
        show_columns = st.multiselect(
            "Select columns to display:",
            df.columns.tolist(),
            default=df.columns.tolist()[:3]
        )
    
    # Display filtered data
    if show_columns:
        st.dataframe(filtered_df[show_columns], use_container_width=True)
    
    # Interactive problem viewer
    st.subheader("📖 Problem Viewer")
    
    if 'task_id' in df.columns:
        selected_problem = st.selectbox("Select a problem to view in detail:", df['task_id'].tolist())
        problem_data = df[df['task_id'] == selected_problem].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Problem Details:**")
            for col, value in problem_data.items():
                if col != 'task_id':
                    st.write(f"**{col.replace('_', ' ').title()}:**")
                    if isinstance(value, str) and len(value) > 100:
                        st.code(value, language='python' if 'code' in col.lower() or 'solution' in col.lower() else None)
                    else:
                        st.write(value)
        
        with col2:
            # Extract and show features for this problem
            if 'canonical_solution' in df.columns:
                text_col = 'canonical_solution'
            elif 'code' in df.columns:
                text_col = 'code'
            else:
                text_col = None
            
            if text_col:
                st.write("**Problem Analysis:**")
                features = extract_text_features(pd.DataFrame([problem_data]), text_col)
                
                for feature, value in features.iloc[0].items():
                    st.metric(feature.replace('_', ' ').title(), f"{value:.1f}" if isinstance(value, float) else str(value))
    
    # Add explanatory text for interactive exploration
    st.markdown("---")
    st.markdown("### 📖 **Understanding Interactive Exploration**")
    st.markdown("""
    **What Is Interactive Exploration?**  
    Interactive exploration lets you dive deep into individual problems and datasets, giving you hands-on control over what you want to analyze.
    
    **How to Use the Data Explorer:**
    - **Filter by Problems**: Select specific problems you want to examine in detail
    - **Choose Columns**: Pick which data fields to display in the table
    - **Customize View**: Focus on the information most relevant to your analysis
    
    **Problem Viewer Features:**
    - **Detailed View**: See all aspects of a single problem including description, solution, and tests
    - **Code Display**: Properly formatted code with syntax highlighting
    - **Feature Analysis**: Automatic calculation of complexity metrics for the selected problem
    
    **What the Metrics Tell You:**
    - **Character/Word Count**: Size and verbosity of the problem description or code
    - **Line Count**: Code length - more lines often indicate more complex solutions
    - **Function Count**: How many functions are defined - shows code organization
    - **Conditional/Loop Counts**: Logical complexity indicators
    - **Readability Score**: How easy the text is to understand (higher = easier)
    
    **Why This Is Useful:**
    - **Problem Selection**: Find problems that match specific criteria (length, complexity, etc.)
    - **Learning**: Understand what makes certain problems more challenging
    - **Comparison**: See how different problems vary in their characteristics
    - **Quality Assessment**: Identify well-structured vs. poorly-structured problems
    
    **Tips for Effective Exploration:**
    - Start with a few problems to get familiar with the interface
    - Compare problems with different complexity scores to understand the differences
    - Use the feature analysis to identify patterns in problem difficulty
    - Look for problems that might be good examples for teaching or learning
    """)



if __name__ == "__main__":
    main()
