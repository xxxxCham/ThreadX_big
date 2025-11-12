import os, ast
ROOT = os.path.abspath('.')
SRC_DIR = os.path.join(ROOT,'src','threadx')
all_mods = set()
for dirpath, dirnames, filenames in os.walk(SRC_DIR):
    # package module
    if '__init__.py' in filenames:
        rel = os.path.relpath(dirpath, SRC_DIR)
        if rel == '.':
            pkg = 'threadx'
        else:
            pkg = 'threadx.' + rel.replace(os.sep,'.')
        all_mods.add(pkg)
    for f in filenames:
        if f.endswith('.py') and f != '__init__.py':
            rel = os.path.relpath(os.path.join(dirpath,f), SRC_DIR)
            mod = 'threadx.' + rel.replace(os.sep,'.')[:-3]
            all_mods.add(mod)

imported = set()
CHECK_DIRS = [SRC_DIR, os.path.join(ROOT,'tests')]
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
                        imported.add(mod)
                elif isinstance(node, ast.Import):
                    for n in node.names:
                        name = n.name
                        if name.startswith('threadx'):
                            imported.add(name)

unused = sorted([m for m in all_mods if m not in imported and not m.endswith('.__init__')])
for m in unused:
    print(m)
