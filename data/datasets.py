"""Dataset loading and processing - Using free alternative datasets"""

import pandas as pd
from datasets import load_dataset
import streamlit as st
import requests
import json
from typing import Dict, Any

@st.cache_data
def load_coder_eval():
    """Load CoderEval dataset - Alternative to HumanEval"""
    try:
        # Create sample coding problems similar to HumanEval structure
        coder_eval_problems = [
            {
                'task_id': 'CoderEval/0',
                'prompt': 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """',
                'canonical_solution': 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False',
                'test': 'assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False',
                'entry_point': 'has_close_elements'
            },
            {
                'task_id': 'CoderEval/1',
                'prompt': 'def separate_paren_groups(paren_string: str) -> List[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    """',
                'canonical_solution': 'def separate_paren_groups(paren_string: str) -> List[str]:\n    result = []\n    current_string = []\n    current_depth = 0\n    \n    for c in paren_string:\n        if c == \'(\':\n            current_depth += 1\n            current_string.append(c)\n        elif c == \')\' :\n            current_depth -= 1\n            current_string.append(c)\n            \n            if current_depth == 0:\n                result.append(\'\'.join(current_string))\n                current_string = []\n    \n    return result',
                'test': 'assert separate_paren_groups(\'( ) (( )) (( )( ))\') == [\'()\', \'(())\', \'(()())\']',
                'entry_point': 'separate_paren_groups'
            },
            {
                'task_id': 'CoderEval/2',
                'prompt': 'def truncate_number(number: float) -> float:\n    """ Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n    \n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    """',
                'canonical_solution': 'def truncate_number(number: float) -> float:\n    return number % 1.0',
                'test': 'assert truncate_number(3.5) == 0.5',
                'entry_point': 'truncate_number'
            },
            {
                'task_id': 'CoderEval/3',
                'prompt': 'def below_zero(operations: List[int]) -> bool:\n    """ You\'re given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    """',
                'canonical_solution': 'def below_zero(operations: List[int]) -> bool:\n    balance = 0\n    \n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n    \n    return False',
                'test': 'assert below_zero([1, 2, -4, 5]) == True',
                'entry_point': 'below_zero'
            },
            {
                'task_id': 'CoderEval/4',
                'prompt': 'def mean_absolute_deviation(numbers: List[float]) -> float:\n    """ For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    """',
                'canonical_solution': 'def mean_absolute_deviation(numbers: List[float]) -> float:\n    mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)',
                'test': 'assert mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) == 1.0',
                'entry_point': 'mean_absolute_deviation'
            }
        ]
        
        return pd.DataFrame(coder_eval_problems)
    except Exception as e:
        st.error(f"Error loading CoderEval: {e}")
        return pd.DataFrame()

@st.cache_data  
def load_code_contests():
    """Load CodeContests dataset - Alternative to MBPP"""
    try:
        # Create sample programming problems
        sample_problems = [
            {
                'task_id': 'contest_001',
                'text': 'Write a function that returns the sum of two numbers.',
                'code': 'def add_numbers(a, b):\n    return a + b',
                'test_list': ['assert add_numbers(2, 3) == 5', 'assert add_numbers(-1, 1) == 0'],
                'test_setup_code': '',
                'challenge_test_list': []
            },
            {
                'task_id': 'contest_002', 
                'text': 'Write a function that checks if a number is even.',
                'code': 'def is_even(n):\n    return n % 2 == 0',
                'test_list': ['assert is_even(4) == True', 'assert is_even(3) == False'],
                'test_setup_code': '',
                'challenge_test_list': []
            },
            {
                'task_id': 'contest_003',
                'text': 'Write a function that finds the maximum element in a list.',
                'code': 'def find_max(lst):\n    return max(lst) if lst else None',
                'test_list': ['assert find_max([1, 2, 3]) == 3', 'assert find_max([]) == None'],
                'test_setup_code': '',
                'challenge_test_list': []
            },
            {
                'task_id': 'contest_004',
                'text': 'Write a function that reverses a string.',
                'code': 'def reverse_string(s):\n    return s[::-1]',
                'test_list': ['assert reverse_string("hello") == "olleh"', 'assert reverse_string("") == ""'],
                'test_setup_code': '',
                'challenge_test_list': []
            },
            {
                'task_id': 'contest_005',
                'text': 'Write a function that counts vowels in a string.',
                'code': 'def count_vowels(s):\n    return sum(1 for c in s.lower() if c in "aeiou")',
                'test_list': ['assert count_vowels("hello") == 2', 'assert count_vowels("xyz") == 0'],
                'test_setup_code': '',
                'challenge_test_list': []
            }
        ]
        
        return pd.DataFrame(sample_problems)
    except Exception as e:
        st.error(f"Error loading CodeContests: {e}")
        return pd.DataFrame()

@st.cache_data
def load_python_problems():
    """Load Python Problems dataset - Alternative to CodeXGLUE"""
    try:
        # Create sample Python programming exercises
        python_exercises = [
            {
                'id': 'py_001',
                'nl': 'Create a function that calculates factorial of a number',
                'code': 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)',
                'difficulty': 'easy'
            },
            {
                'id': 'py_002',
                'nl': 'Create a function that checks if a string is palindrome',
                'code': 'def is_palindrome(s):\n    s = s.lower().replace(" ", "")\n    return s == s[::-1]',
                'difficulty': 'easy'
            },
            {
                'id': 'py_003',
                'nl': 'Create a function that generates fibonacci sequence up to n terms',
                'code': 'def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib',
                'difficulty': 'medium'
            },
            {
                'id': 'py_004',
                'nl': 'Create a function that sorts a list of dictionaries by a key',
                'code': 'def sort_by_key(lst, key):\n    return sorted(lst, key=lambda x: x.get(key, 0))',
                'difficulty': 'medium'
            },
            {
                'id': 'py_005',
                'nl': 'Create a function that finds common elements between two lists',
                'code': 'def find_common(list1, list2):\n    return list(set(list1) & set(list2))',
                'difficulty': 'easy'
            }
        ]
        
        return pd.DataFrame(python_exercises)
    except Exception as e:
        st.error(f"Error loading Python Problems: {e}")
        return pd.DataFrame()

def get_dataset_info(df: pd.DataFrame, name: str) -> dict:
    """Get basic dataset information"""
    if df.empty:
        return {"name": name, "size": 0, "columns": []}
    
    return {
        "name": name,
        "size": len(df),
        "columns": list(df.columns),
        "sample": df.head(3).to_dict('records') if len(df) > 0 else []
    }