#!/usr/bin/env python
"""
Script to convert a Python template file to a Jupyter notebook.
This script takes a Python file with special comments indicating cell boundaries
and converts it to a Jupyter notebook (.ipynb) file.
"""

import json
import re
import sys
import os

def convert_to_notebook(py_file, ipynb_file):
    """
    Convert a Python file to a Jupyter notebook.
    
    Args:
        py_file (str): Path to the Python file
        ipynb_file (str): Path to save the Jupyter notebook
    """
    # Read the Python file
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content by cell markers
    # In this case, we use "# In[X]:" as cell markers
    cell_pattern = r'# In\[\d+\]:'
    cells = re.split(cell_pattern, content)
    
    # The first cell might contain imports and setup code
    if not cells[0].strip().startswith('#'):
        first_cell = cells[0]
        cells = cells[1:]
    else:
        first_cell = ""
    
    # Create notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Add the first cell (if it exists)
    if first_cell.strip():
        # Extract markdown content from the beginning (lines starting with #)
        markdown_lines = []
        code_lines = []
        
        for line in first_cell.split('\n'):
            if line.strip().startswith('#'):
                markdown_lines.append(line.strip()[2:])  # Remove '# ' prefix
            else:
                code_lines.append(line)
        
        # Add markdown cell if there are markdown lines
        if markdown_lines:
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": '\n'.join(markdown_lines)
            })
        
        # Add code cell if there are code lines
        if code_lines:
            notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": '\n'.join(code_lines)
            })
    
    # Process the rest of the cells
    for cell in cells:
        # Skip empty cells
        if not cell.strip():
            continue
        
        # Extract markdown content (lines starting with #)
        markdown_lines = []
        code_lines = []
        
        for line in cell.split('\n'):
            if line.strip().startswith('#'):
                markdown_lines.append(line.strip()[2:])  # Remove '# ' prefix
            else:
                code_lines.append(line)
        
        # Add markdown cell if there are markdown lines
        if markdown_lines:
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": '\n'.join(markdown_lines)
            })
        
        # Add code cell if there are code lines
        if code_lines:
            notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": '\n'.join(code_lines)
            })
    
    # Write the notebook to file
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Converted {py_file} to {ipynb_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_notebook.py <python_file> [output_notebook]")
        sys.exit(1)
    
    py_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        ipynb_file = sys.argv[2]
    else:
        # Use the same name but with .ipynb extension
        ipynb_file = os.path.splitext(py_file)[0] + '.ipynb'
    
    convert_to_notebook(py_file, ipynb_file)