import ast
import re

def clean_and_normalize_source(code_str):
    try:
        parsed = ast.parse(code_str)
        clean_code = ast.unparse(parsed)
        
        def replacer(m):
            if m.group(1): 
                return m.group(1)
            else:
                return '='
                
        regex = r'("[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')|(?<![=<>!])\s*=\s*(?![=])'
        
        res = []
        for line in clean_code.splitlines():
            if not line.strip(): continue
            new_line = re.sub(regex, replacer, line)
            res.append(new_line)
        return "\n".join(res)
    except Exception as e:
        print("Error:", e)
        return code_str

sample = """
def foo():
    # this is a comment
    a  =  5
    b == 6
    if a == b:
        c = "hello = world" # string with =
    return a
"""
print(repr(clean_and_normalize_source(sample)))
