"""Safe code execution utilities"""

import subprocess
import tempfile
import os
from typing import Dict, Any

def execute_python_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Safely execute Python code in a temporary file
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
        
    Returns:
        Dictionary with execution results
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Execute code
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Clean up
        os.unlink(temp_file)
        
        return {
            'success': result.returncode == 0,
            'output': result.stdout.strip(),
            'error': result.stderr.strip(),
            'return_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'error': f'Code execution timed out after {timeout} seconds',
            'return_code': -1
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e),
            'return_code': -1
        }

def validate_code_safety(code: str) -> Dict[str, Any]:
    """
    Basic validation to check if code is safe to execute
    
    Args:
        code: Code to validate
        
    Returns:
        Dictionary with validation results
    """
    dangerous_patterns = [
        'import os', 'import sys', 'import subprocess',
        'open(', 'file(', 'eval(', 'exec(',
        '__import__', 'globals()', 'locals()',
        'input(', 'raw_input('
    ]
    
    issues = []
    for pattern in dangerous_patterns:
        if pattern in code.lower():
            issues.append(f"Potentially unsafe: {pattern}")
    
    return {
        'is_safe': len(issues) == 0,
        'issues': issues,
        'recommendation': 'Code appears safe to execute' if len(issues) == 0 else 'Review code before execution'
    }