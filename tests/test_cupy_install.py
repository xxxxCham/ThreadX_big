"""Test CuPy installation and GPU detection."""

import cupy as cp

print(f"✅ CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"Device count: {cp.cuda.runtime.getDeviceCount()}")

for i in range(cp.cuda.runtime.getDeviceCount()):
    device = cp.cuda.Device(i)
    name = (
        device.attributes.get("Name", b"Unknown").decode()
        if isinstance(device.attributes.get("Name"), bytes)
        else device.attributes.get("Name", "Unknown")
    )
    print(f"GPU {i}: {name} (Compute Capability {device.compute_capability})")

# Test simple computation
x = cp.array([1, 2, 3])
y = cp.array([4, 5, 6])
z = x + y
print(f"\n✅ CuPy computation test: {x.tolist()} + {y.tolist()} = {z.tolist()}")
print("\n🎉 CuPy is fully operational!")
