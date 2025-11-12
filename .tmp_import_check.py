import os, ast, sys
ROOT = os.path.abspath('.')
SRC = os.path.join(ROOT, 'src', 'threadx')
CHECK_DIRS = [os.path.join(ROOT,'src','threadx'), os.path.join(ROOT,'tests')]
missing = {}
imports = set()
for base in CHECK_DIRS:
    for dirpath,_,filenames in os.walk(base):
        for f in filenames:
            if not f.endswith('.py'): continue
            p = os.path.join(dirpath,f)
            try:
                with open(p,'r',encoding='utf-8') as fh:
                    tree = ast.parse(fh.read(), filename=p)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ''
                    if mod.startswith('threadx'):
                        imports.add(mod)
                elif isinstance(node, ast.Import):
                    for n in node.names:
                        name = n.name
                        if name.startswith('threadx'):
                            imports.add(name)

def resolves(mod:str)->bool:
    rel = mod.split('.')
    base = os.path.join(SRC, *rel[1:]) if rel[0]=='threadx' else None
    if base is None: return True
    if os.path.isfile(base + '.py'): return True
    if os.path.isdir(base) and os.path.isfile(os.path.join(base,'__init__.py')):
        return True
    return False

missing_mods = sorted([m for m in imports if not resolves(m)])
for m in missing_mods:
    print(m)
