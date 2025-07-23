"""StarCoder model implementation - Lightweight version"""

from models.base_model import BaseModel

class StarCoder(BaseModel):
    """StarCoder model for multi-language code generation - Using 1B version"""
    
    def __init__(self):
        super().__init__("bigcode/starcoderbase-1b")  # Much smaller 1B version
    
    def generate_code(self, prompt: str, language: str = "python", **kwargs) -> str:
        """Generate code with language specification"""
        formatted_prompt = f"# {language.title()} code for: {prompt}\n"
        return super().generate_code(formatted_prompt, **kwargs)