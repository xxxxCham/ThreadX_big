import importlib, inspect, sys

print("sys.path (first 12):")
for p in sys.path[:12]:
    print(p)
print("\ntrying to import threadx.data.validate...")
try:
    m = importlib.import_module("threadx.data.validate")
    print("imported module file:", getattr(m, "__file__", None))
    print("has validate_dataset:", hasattr(m, "validate_dataset"))
    if hasattr(m, "validate_dataset"):
        src = inspect.getsource(m.validate_dataset)
        print("\nvalidate_dataset source (first 10 lines):")
        print("\n".join(src.splitlines()[:10]))
except Exception as e:
    import traceback

    print("IMPORT ERROR:", type(e).__name__, e)
    traceback.print_exc()
