"""
Main Streamlit application for Code Generation & Bug Fixing System
"""

import streamlit as st
import pandas as pd
from config import STREAMLIT_CONFIG, HUGGINGFACE_TOKEN, MODELS
from models.codellama import CodeLLaMA
from models.starcoder import StarCoder  
from models.replit_coder import ReplitCoder
from data.datasets import load_humaneval, load_mbpp, load_codexglue, get_dataset_info
from data.eda import analyze_humaneval, analyze_mbpp, analyze_codexglue, create_overview_charts
from utils.helpers import display_model_info, format_code_output, create_model_comparison_table
from utils.code_executor import execute_python_code, validate_code_safety

# Page configuration
st.set_page_config(**STREAMLIT_CONFIG)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {
        'CodeLLaMA': None,
        'StarCoder': None,
        'Replit Coder': None
    }
    st.session_state.models_loaded = {
        'CodeLLaMA': False,
        'StarCoder': False, 
        'Replit Coder': False
    }

# Sidebar
with st.sidebar:
    st.header("System Configuration")
    
    # API Token check
    if not HUGGINGFACE_TOKEN:
        st.error("Please set HUGGINGFACE_TOKEN in .env file")
        st.stop()
    else:
        st.success("API Token configured")
    
    st.subheader("Model Status")
    for model_name in MODELS.keys():
        display_model_info(model_name, st.session_state.models_loaded[model_name])
    
    # Model loading
    st.subheader("Load Models")
    selected_model = st.selectbox("Choose model to load:", list(MODELS.keys()))
    
    if st.button("Load Model"):
        with st.spinner(f"Loading {selected_model}..."):
            try:
                if selected_model == 'CodeLLaMA':
                    st.session_state.models[selected_model] = CodeLLaMA()
                elif selected_model == 'StarCoder':
                    st.session_state.models[selected_model] = StarCoder()
                elif selected_model == 'Replit Coder':
                    st.session_state.models[selected_model] = ReplitCoder()
                
                success = st.session_state.models[selected_model].load_model()
                st.session_state.models_loaded[selected_model] = success
                
                if success:
                    st.success(f"{selected_model} loaded successfully!")
                else:
                    st.error(f"Failed to load {selected_model}")
                    
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")

# Main header
st.title("Code Generation & Bug Fixing System")
st.markdown("Generate code and debug issues using CodeLLaMA, StarCoder, and Replit Coder")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Code Generation & Debugging", "Dataset Analysis", "Model Comparison"])

with tab1:
    st.header("Code Generation & Debugging")
    
    # Check if any model is loaded
    loaded_models = [name for name, loaded in st.session_state.models_loaded.items() if loaded]
    
    if not loaded_models:
        st.warning("Please load at least one model from the sidebar first!")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input")
            
            # Task type selection
            task_type = st.selectbox("Task Type:", ["Generate Code", "Debug Code"])
            
            if task_type == "Generate Code":
                prompt = st.text_area(
                    "Describe what you want to code:",
                    placeholder="e.g., Create a function to calculate fibonacci numbers",
                    height=100
                )
                error_msg = ""
            else:
                prompt = st.text_area(
                    "Paste your buggy code:",
                    placeholder="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                    height=150
                )
                error_msg = st.text_input("Error message (optional):", placeholder="RecursionError: maximum recursion depth exceeded")
            
            # Model selection for generation
            selected_model_gen = st.selectbox("Select model:", loaded_models)
            
            # Generation parameters
            with st.expander("Advanced Settings"):
                max_length = st.slider("Max length:", 50, 1000, 300)
                temperature = st.slider("Temperature:", 0.1, 1.0, 0.7)
            
            generate_btn = st.button("Generate/Debug", type="primary")
        
        with col2:
            st.subheader("Output")
            
            if generate_btn and prompt:
                if selected_model_gen in st.session_state.models and st.session_state.models[selected_model_gen]:
                    model = st.session_state.models[selected_model_gen]
                    
                    with st.spinner("Generating..."):
                        if task_type == "Generate Code":
                            result = model.generate_code(prompt, max_length=max_length, temperature=temperature)
                        else:
                            result = model.debug_code(prompt, error_msg)
                        
                        # Clean and display result
                        cleaned_result = format_code_output(result)
                        st.code(cleaned_result, language='python')
                        
                        # Code safety check
                        safety_check = validate_code_safety(cleaned_result)
                        if safety_check['is_safe']:
                            st.success("Code appears safe")
                            
                            # Code execution option
                            if st.button("Execute Code"):
                                exec_result = execute_python_code(cleaned_result)
                                
                                if exec_result['success']:
                                    st.success("Code executed successfully!")
                                    if exec_result['output']:
                                        st.text("Output:")
                                        st.code(exec_result['output'])
                                else:
                                    st.error("Execution failed:")
                                    st.code(exec_result['error'])
                        else:
                            st.warning("Potential safety issues detected:")
                            for issue in safety_check['issues']:
                                st.write(f"• {issue}")

with tab2:
    st.header("Dataset Analysis")
    
    dataset_tab1, dataset_tab2, dataset_tab3, dataset_tab4 = st.tabs(["Overview", "HumanEval", "MBPP", "CodeXGLUE"])
    
    with dataset_tab1:
        st.subheader("Dataset Overview")
        
        # Load basic info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("HumanEval", "164", "Hand-written problems")
        with col2:
            st.metric("MBPP", "1000+", "Crowd-sourced problems") 
        with col3:
            st.metric("CodeXGLUE", "Multiple", "Multi-task datasets")
        
        # Overview chart
        overview_fig = create_overview_charts()
        st.plotly_chart(overview_fig, use_container_width=True)
    
    with dataset_tab1:
        st.subheader("HumanEval Analysis")
        
        with st.spinner("Loading HumanEval dataset..."):
            humaneval_df = load_humaneval()
            
        if not humaneval_df.empty:
            # Dataset info
            info = get_dataset_info(humaneval_df, "HumanEval")
            st.write(f"**Dataset Size:** {info['size']} problems")
            st.write(f"**Columns:** {', '.join(info['columns'])}")
            
            # Analysis
            analysis = analyze_humaneval(humaneval_df)
            if 'difficulty_chart' in analysis:
                st.plotly_chart(analysis['difficulty_chart'], use_container_width=True)
            
            # Sample data
            if info['sample']:
                st.subheader("Sample Problems")
                st.dataframe(pd.DataFrame(info['sample']))
        else:
            st.error("Failed to load HumanEval dataset")
    
    with dataset_tab2:
        st.subheader("MBPP Analysis")
        
        with st.spinner("Loading MBPP dataset..."):
            mbpp_df = load_mbpp()
            
        if not mbpp_df.empty:
            info = get_dataset_info(mbpp_df, "MBPP")
            st.write(f"**Dataset Size:** {info['size']} problems")
            st.write(f"**Columns:** {', '.join(info['columns'])}")
            
            analysis = analyze_mbpp(mbpp_df)
            if 'complexity_chart' in analysis:
                st.plotly_chart(analysis['complexity_chart'], use_container_width=True)
            
            if info['sample']:
                st.subheader("Sample Problems")
                st.dataframe(pd.DataFrame(info['sample']))
        else:
            st.error("Failed to load MBPP dataset")
    
    with dataset_tab3:
        st.subheader("CodeXGLUE Analysis")
        
        with st.spinner("Loading CodeXGLUE dataset..."):
            codexglue_df = load_codexglue()
            
        if not codexglue_df.empty:
            info = get_dataset_info(codexglue_df, "CodeXGLUE")
            st.write(f"**Dataset Size:** {info['size']} samples")
            st.write(f"**Columns:** {', '.join(info['columns'])}")
            
            analysis = analyze_codexglue(codexglue_df)
            if 'task_chart' in analysis:
                st.plotly_chart(analysis['task_chart'], use_container_width=True)
            
            if info['sample']:
                st.subheader("Sample Data")
                st.dataframe(pd.DataFrame(info['sample']))
        else:
            st.error("Failed to load CodeXGLUE dataset")

with tab3:
    st.header("Model Comparison")
    
    loaded_models = [name for name, loaded in st.session_state.models_loaded.items() if loaded]
    
    if len(loaded_models) < 2:
        st.warning("Please load at least 2 models to compare")
    else:
        st.subheader("Compare Model Outputs")
        
        comparison_prompt = st.text_area(
            "Enter a coding task to compare across models:",
            placeholder="Create a function to sort a list of dictionaries by a specific key",
            height=100
        )
        
        if st.button("Compare Models"):
            if comparison_prompt:
                results = {}
                
                for model_name in loaded_models:
                    if st.session_state.models[model_name]:
                        with st.spinner(f"Generating with {model_name}..."):
                            try:
                                result = st.session_state.models[model_name].generate_code(
                                    comparison_prompt, 
                                    max_length=300,
                                    temperature=0.5
                                )
                                results[model_name] = format_code_output(result)
                            except Exception as e:
                                results[model_name] = f"Error: {str(e)}"
                
                # Display comparison
                create_model_comparison_table(results)
                
                # Simple comparison metrics
                st.subheader("Quick Comparison")
                comparison_data = []
                
                for model_name, code in results.items():
                    if not code.startswith("Error:"):
                        lines = len([line for line in code.split('\n') if line.strip()])
                        has_functions = "def " in code
                        comparison_data.append({
                            "Model": model_name,
                            "Lines of Code": lines,
                            "Has Functions": "Yes" if has_functions else "No",
                            "Code Length": len(code)
                        })
                
                if comparison_data:
                    st.dataframe(pd.DataFrame(comparison_data))

# Footer
st.markdown("---")
st.markdown("**Code Generation & Bug Fixing System** | Built with Streamlit | Powered by Hugging Face Transformers")