"""Replit Coder model implementation"""

from models.base_model import BaseModel

class ReplitCoder(BaseModel):
    """Replit Coder model for interactive coding"""
    
    def __init__(self):
        super().__init__("replit/replit-code-v1-3b")
    
    def generate_code(self, prompt: str, **kwargs) -> str:
        """Generate code with Replit-style formatting"""
        return super().generate_code(prompt, **kwargs)