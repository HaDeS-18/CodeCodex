"""Base model class for all LLMs"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import HUGGINGFACE_TOKEN, GENERATION_CONFIG

class BaseModel(ABC):
    """Base class for all code generation models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self) -> bool:
        """Load the model and tokenizer with optimizations for lighter hardware"""
        try:
            print(f"Loading {self.model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=HUGGINGFACE_TOKEN,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Optimized loading for lighter hardware
            model_kwargs = {
                "token": HUGGINGFACE_TOKEN,
                "trust_remote_code": True,
            }
            
            # Use lighter configuration based on available resources
            if torch.cuda.is_available():
                # GPU available - use float16 for memory efficiency
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                })
            else:
                # CPU only - use even more aggressive optimizations
                model_kwargs.update({
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True
                })
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to CPU if no GPU to avoid memory issues
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            
            self.is_loaded = True
            print(f"✅ {self.model_name} loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {str(e)}")
            return False
    
    def generate_code(self, 
                     prompt: str, 
                     max_length: Optional[int] = None,
                     temperature: Optional[float] = None) -> str:
        """Generate code from prompt"""
        
        if not self.is_loaded:
            return "Error: Model not loaded"
        
        max_length = max_length or GENERATION_CONFIG["max_length"]
        temperature = temperature or GENERATION_CONFIG["temperature"]
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=GENERATION_CONFIG["top_p"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return result.strip()
    
    def debug_code(self, code: str, error_msg: str = "") -> str:
        """Debug and fix code"""
        prompt = f"""Fix this code:

Code:
{code}

Error: {error_msg}

Fixed code:"""
        
        return self.generate_code(prompt, temperature=0.3)
    
    def unload_model(self):
        """Unload model from memory"""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False