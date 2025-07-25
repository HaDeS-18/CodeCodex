"""Replit Coder model implementation - Optimized version"""

from models.base_model import BaseModel

class ReplitCoder(BaseModel):
    """Replit Coder model for interactive coding - Using smaller version"""
    
    def __init__(self):
        super().__init__("replit/replit-code-v1_5-3b")  # Latest optimized version
    
    def generate_code(self, prompt: str, **kwargs) -> str:
        """Generate code with Replit-style formatting"""
        return super().generate_code(prompt, **kwargs)