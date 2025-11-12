"""Test rapide de Numba JIT"""

from numba import njit
import numpy as np


@njit(fastmath=True, cache=True)
def test_function(arr):
    """Fonction test compilée JIT"""
    result = 0.0
    for i in range(len(arr)):
        result += arr[i] * 2.0
    return result


# Test
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = test_function(arr)
print(f"✅ Numba JIT fonctionne ! Résultat: {result}")
print(f"Attendu: {sum(arr) * 2}")
