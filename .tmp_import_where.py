import os, ast
ROOT = os.path.abspath('.')
SRC = os.path.join(ROOT, 'src', 'threadx')
CHECK_DIRS = [os.path.join(ROOT,'src','threadx'), os.path.join(ROOT,'tests')]
missing = {}
entries = []
for base in CHECK_DIRS:
    for dirpath,_,filenames in os.walk(base):
        for f in filenames:
            if not f.endswith('.py'): continue
            p = os.path.join(dirpath,f)
            try:
                src = open(p,'r',encoding='utf-8').read()
                tree = ast.parse(src, filename=p)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ''
                    if mod.startswith('threadx'):
                        entries.append((p, node.lineno, f'from {mod} import ...'))
                elif isinstance(node, ast.Import):
                    for n in node.names:
                        name = n.name
                        if name.startswith('threadx'):
                            entries.append((p, node.lineno, f'import {name}'))

def resolves(mod:str)->bool:
    rel = mod.split('.')
    base = os.path.join(SRC, *rel[1:]) if rel[0]=='threadx' else None
    if base is None: return True
    if os.path.isfile(base + '.py'): return True
    if os.path.isdir(base) and os.path.isfile(os.path.join(base,'__init__.py')):
        return True
    return False

unresolved = []
for p,lineno,text in entries:
    mod = text.split()[1]
    if not resolves(mod):
        unresolved.append((p, lineno, text))

for p,lineno,text in sorted(unresolved):
    print(f'{os.path.relpath(p,ROOT)}:{lineno}: {text}')
