import itertools
import pytest

from threadx.optimization.scenarios import generate_param_grid

def test_generate_param_grid_accepts_value_and_values():
    grid_spec = {
        "timeframe": {"value": "1h"},               # scalaire
        "lookback_hours": {"values": [24, 48, 72]}, # liste
        "symbols": {"values": ["BTCUSDC", "ETHUSDC"]},
    }
    combos = generate_param_grid(grid_spec)

    # 1 * 3 * 2 = 6 combinaisons attendues
    assert isinstance(combos, list)
    assert len(combos) == 6

    # Un petit échantillon de couvertures
    expected = [
        {"timeframe": "1h", "lookback_hours": 24, "symbols": "BTCUSDC"},
        {"timeframe": "1h", "lookback_hours": 72, "symbols": "ETHUSDC"},
    ]
    for e in expected:
        assert e in combos

def test_generate_param_grid_rejects_unknown_shape():
    # Clé non supportée: 'grid'
    bad = {"lookback_hours": {"grid": [24, 48]}}
    with pytest.raises(ValueError):
        generate_param_grid(bad)

def test_generate_param_grid_rejects_non_mapping():
    # Valeur brute au lieu d'un mapping {'value': ...} / {'values': [...]}
    bad = {"lookback_hours": [24, 48]}
    with pytest.raises(ValueError):
        generate_param_grid(bad)
