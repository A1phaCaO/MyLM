import json
import sys

# Read and validate the notebook file
try:
    with open('pre_train_notebook.ipynb', 'r', encoding='utf-8') as f:
        content = f.read()
        print(f'File read successfully. Length: {len(content)} characters')
        
    # Try to parse as JSON
    notebook = json.loads(content)
    print('JSON parsing: SUCCESS')
    
    # Validate notebook structure
    required_keys = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
    for key in required_keys:
        if key not in notebook:
            print(f'ERROR: Missing required key: {key}')
            sys.exit(1)
    
    print(f'Notebook format: {notebook["nbformat"]}.{notebook["nbformat_minor"]}')
    print(f'Number of cells: {len(notebook["cells"])}')
    
    # Validate each cell
    cell_types = {'markdown', 'code'}
    for i, cell in enumerate(notebook['cells']):
        if 'cell_type' not in cell:
            print(f'ERROR: Cell {i} missing cell_type')
            sys.exit(1)
            
        if cell['cell_type'] not in cell_types:
            print(f'ERROR: Cell {i} has invalid type: {cell["cell_type"]}')
            sys.exit(1)
            
        if 'source' not in cell:
            print(f'ERROR: Cell {i} missing source')
            sys.exit(1)
            
        if not isinstance(cell['source'], (list, str)):
            print(f'ERROR: Cell {i} source has invalid type: {type(cell["source"])}')
            sys.exit(1)
    
    print('All validations passed!')
    print('Notebook is syntactically correct and ready to use.')
    
    # Test if the code would run by parsing each code cell
    import ast
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    print(f'Found {len(code_cells)} code cells')
    
    for i, cell in enumerate(code_cells):
        if isinstance(cell['source'], list):
            code = ''.join(cell['source'])
        else:
            code = cell['source']
            
        if code.strip():  # Only check if there's actual content
            try:
                ast.parse(code)
                print(f'Code cell {i}: VALID syntax')
            except SyntaxError as e:
                print(f'Code cell {i}: INVALID syntax - {e}')
                sys.exit(1)
    
    print('All code cells have valid Python syntax!')
    
except FileNotFoundError:
    print('ERROR: File pre_train_notebook.ipynb not found')
except json.JSONDecodeError as e:
    print(f'ERROR: Invalid JSON in notebook: {e}')
except Exception as e:
    print(f'ERROR: {e}')