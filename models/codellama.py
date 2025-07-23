"""CodeLLaMA model implementation - Lightweight version"""

from models.base_model import BaseModel

class CodeLLaMA(BaseModel):
    """CodeLLaMA model for Python code generation - Using lighter configuration"""
    
    def __init__(self):
        # Use the smaller quantized version or implement CPU optimizations
        super().__init__("microsoft/CodeBERT-base")  # Much smaller alternative
        
    def generate_code(self, prompt: str, **kwargs) -> str:
        """Generate Python code with CodeLLaMA-specific formatting"""
        # Add Python context for better results
        formatted_prompt = f"# Python code\n# Task: {prompt}\n"
        return super().generate_code(formatted_prompt, **kwargs)