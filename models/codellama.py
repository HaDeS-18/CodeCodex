"""CodeLLaMA model implementation"""

from models.base_model import BaseModel

class CodeLLaMA(BaseModel):
    """CodeLLaMA model for Python code generation"""
    
    def __init__(self):
        super().__init__("codellama/CodeLlama-7b-Python-hf")
    
    def generate_code(self, prompt: str, **kwargs) -> str:
        """Generate Python code with CodeLLaMA-specific formatting"""
        # Add Python context for better results
        formatted_prompt = f"# Python code\n# Task: {prompt}\n"
        return super().generate_code(formatted_prompt, **kwargs)