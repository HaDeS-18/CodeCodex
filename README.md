# Code Generation & Bug Fixing System

A multi-LLM system for code generation and debugging using CodeLLaMA, StarCoder, and Replit Coder models with comprehensive dataset analysis.

## Features

- **Code Generation**: Generate code from natural language descriptions
- **Bug Debugging**: Fix buggy code with intelligent suggestions
- **Multi-Model Support**: Compare outputs from 3 different LLMs
- **Dataset Analysis**: Interactive EDA on HumanEval, MBPP, and CodeXGLUE datasets
- **Safe Execution**: Validate and execute generated code securely
- **Interactive UI**: Clean Streamlit web interface

## Project Structure

```
code_generation_project/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── .env                             # Environment variables (API keys)
├── .gitignore                       # Git ignore rules
├── config.py                        # Configuration settings
├── app.py                           # Main Streamlit application
│
├── models/                          # LLM model implementations
│   ├── __init__.py
│   ├── base_model.py               # Base class for models
│   ├── codellama.py                # CodeLLaMA implementation
│   ├── starcoder.py                # StarCoder implementation
│   └── replit_coder.py             # Replit Coder implementation
│
├── data/                           # Dataset handling
│   ├── __init__.py
│   ├── datasets.py                 # Dataset loaders
│   └── eda.py                      # EDA and analysis functions
│
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── code_executor.py            # Safe code execution
│   └── helpers.py                  # General helper functions
│
├── codecodex/                        # Conda environment (local)
└── cache/                          # Model and data cache (auto-created)
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd code_generation_project
```

### 2. Create Conda Environment
```bash
# Create conda environment with local path
conda create -p codecodex python=3.10 -y

# Activate the environment
conda activate ./codecodex

# Install pip in the environment
conda install pip -y
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
1. Copy the `.env` file and add your Hugging Face token:
```bash
cp .env.example .env
```

2. Edit `.env` and add your token:
```bash
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

Get your Hugging Face token from: https://huggingface.co/settings/tokens

### 5. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Loading Models
1. In the sidebar, select a model (CodeLLaMA, StarCoder, or Replit Coder)
2. Click "Load Model" and wait for it to download and initialize
3. Green checkmark indicates the model is ready

### Code Generation
1. Go to "Code Generation & Debugging" tab
2. Select "Generate Code" task type
3. Describe what you want in natural language
4. Choose your loaded model and click "Generate/Debug"
5. Optionally execute the generated code safely

### Bug Debugging
1. Select "Debug Code" task type  
2. Paste your buggy code
3. Add error message (optional)
4. Click "Generate/Debug" to get fixed code

### Dataset Analysis
1. Visit "Dataset Analysis" tab
2. Explore interactive visualizations of:
   - HumanEval: 164 hand-written programming problems
   - MBPP: 1000+ crowd-sourced Python problems  
   - CodeXGLUE: Multi-task code understanding benchmark

### Model Comparison
1. Go to "Model Comparison" tab
2. Load 2+ models first
3. Enter a coding task
4. Compare side-by-side outputs from different models

## Models

### CodeLLaMA
- **Focus**: Python code generation and completion
- **Strengths**: High-quality algorithmic code, function implementations
- **Model**: `codellama/CodeLlama-7b-Python-hf`

### StarCoder  
- **Focus**: Multi-language code generation
- **Strengths**: Supports 80+ programming languages, code translation
- **Model**: `bigcode/starcoder`

### Replit Coder
- **Focus**: Interactive coding assistance  
- **Strengths**: Educational explanations, beginner-friendly
- **Model**: `replit/replit-code-v1-3b`

## Datasets

### HumanEval
- **Size**: 164 problems
- **Type**: Hand-written programming challenges
- **Use**: Model evaluation and benchmarking

### MBPP (Mostly Basic Python Problems)
- **Size**: 1000+ problems
- **Type**: Crowd-sourced Python problems
- **Use**: Training and diverse problem types

### CodeXGLUE
- **Size**: Multiple sub-datasets
- **Type**: Multi-task code understanding benchmark
- **Use**: Comprehensive evaluation across tasks

## System Requirements

### Minimum
- **RAM**: 8GB  
- **Storage**: 20GB free space
- **Python**: 3.8+
- **Internet**: For model downloads

### Recommended
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for faster inference)
- **Storage**: 50GB+ SSD
- **CPU**: 8+ cores

## Troubleshooting

### Common Issues

**Model loading fails:**
- Check your Hugging Face token is valid
- Ensure sufficient RAM/storage
- Try smaller models first (Replit Coder)

**Out of memory:**
- Close other applications
- Try CPU inference instead of GPU
- Use smaller models

**Slow generation:**
- Use GPU if available
- Reduce max_length parameter
- Consider model quantization

**Dataset loading fails:**
- Check internet connection
- Clear cache folder and retry
- Some datasets may require HF authentication

### Performance Tips

1. **GPU Usage**: Models automatically use GPU if available
2. **Memory Management**: Models are loaded on-demand, unload unused ones
3. **Caching**: Datasets and model outputs are cached for faster reloads
4. **Batch Processing**: Use model comparison for efficient multi-model inference

## Development

### Adding New Models
1. Create new model class in `models/` following `base_model.py`
2. Add model configuration in `config.py`
3. Update UI in `app.py`

### Adding New Datasets
1. Add loader function in `data/datasets.py`
2. Create analysis function in `data/eda.py`
3. Add new tab in Streamlit interface

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face** for providing the model hosting and transformers library
- **Meta AI** for CodeLLaMA
- **BigCode** for StarCoder
- **Replit** for Replit Coder
- **OpenAI** for HumanEval dataset
- **Microsoft** for CodeXGLUE benchmark

## Citation

If you use this project in your research, please cite:

```bibtex
@software{code_generation_system,
  title={Code Generation & Bug Fixing System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/code_generation_project}
}
```