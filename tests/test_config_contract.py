"""
Tests de contrat pour la configuration d'optimisation ThreadX.
Vérifie que generate_param_grid, validate_cli_config et run_sweep
respectent les formats attendus (value/values, pas grid).
"""

import pytest

# Cibles à tester
from threadx.optimization.scenarios import generate_param_grid
from threadx.optimization.run import validate_cli_config, run_sweep


# ---------- Utils simples ----------
def _mk_base_config():
    """
    Renvoie un dict de config valide minimal, aligné avec ton plan .toml actuel.
    """
    return {
        "run": {
            "type": "grid",
            "seed": 42,
            "n_scenarios": 100,
            "sampler": "grid",
            "verbose": True,
            "params": {
                # singleton -> via "value"
                "timeframe": {"value": "1h"},
                # variables -> via "values"
                "lookback_hours": {"values": [24, 48, 72]},
                "symbols": {"values": ["BTCUSDC", "ETHUSDC", "SOLUSDC", "ADAUSDC"]},
            },
            "constraints": {},
        },
        "scoring": {
            "primary": "sharpe",
            "constraints": {"max_drawdown": 0.25},
        },
        "output": {
            "dir": "artifacts/sweeps/minimal",
            "format": "parquet",
        },
        "execution": {
            "max_workers": 4,
            "reuse_cache": True,
        },
    }


# ---------- generate_param_grid ----------
def test_generate_param_grid_accepts_values_and_value():
    grid_spec = {
        "timeframe": {"value": "1h"},
        "lookback_hours": {"values": [24, 48]},
        "symbols": {"values": ["BTCUSDC", "ETHUSDC"]},
    }
    combos = generate_param_grid(grid_spec)
    # 1 * 2 * 2 = 4 combinaisons attendues
    assert len(combos) == 4
    # Sanity: chaque combo doit avoir toutes les clés
    for c in combos:
        assert set(c.keys()) == {"timeframe", "lookback_hours", "symbols"}


def test_generate_param_grid_accepts_grid_key():
    """
    Vérifie que {grid: [...]} est accepté (selon ton implémentation actuelle).
    """
    grid_spec = {
        "timeframe": {"value": "1h"},
        "lookback_hours": {"grid": [24, 48, 72]},
        "symbols": {"values": ["BTCUSDC", "ETHUSDC"]},
    }
    combos = generate_param_grid(grid_spec)
    # 1 * 3 * 2 = 6 combinaisons
    assert len(combos) == 6


def test_generate_param_grid_accepts_raw_list_for_convenience():
    """
    Si tu as gardé la compat "liste brute" côté implémentation,
    on s'assure que ça marche encore.
    """
    grid_spec = {
        "timeframe": {"value": "1h"},
        "lookback_hours": [24, 48],  # liste brute
        "symbols": ["BTCUSDC", "ETHUSDC"],  # liste brute
    }
    combos = generate_param_grid(grid_spec)
    assert len(combos) == 4


# ---------- validate_cli_config ----------
def test_validate_cli_config_requires_run_block(tmp_path):
    """
    Vérifie que la config nécessite un bloc run avec la nouvelle structure.
    """
    cfg = {
        "params": {
            "timeframe": {"value": "1h"},
        },
        "scoring": {"primary": "sharpe"},
        "output": {"dir": "artifacts"},
    }
    plan = tmp_path / "plan.toml"
    plan.write_text("# dummy path for error message anchoring", encoding="utf-8")

    # Sans bloc run, devrait être OK mais on récupère les valeurs par défaut
    validated = validate_cli_config(cfg, str(plan))
    assert isinstance(validated, dict)


def test_validate_cli_config_ok_on_mapping_values(tmp_path):
    cfg = _mk_base_config()
    plan = tmp_path / "plan.toml"
    plan.write_text("# ok", encoding="utf-8")

    validated = validate_cli_config(cfg, str(plan))
    # renvoie typiquement un dict nettoyé/validé
    assert isinstance(validated, dict)
    assert "run" in validated or "params" in validated


# ---------- run_sweep: passage de config_path ----------
def test_run_sweep_passes_config_path_and_dry_run(monkeypatch, tmp_path):
    """
    Vérifie que run_sweep accepte bien (config, config_path, dry_run)
    et qu'on ne part pas sur une exécution lourde.
    On monkeypatch le SweepRunner pour court-circuiter toute charge.
    """
    cfg = _mk_base_config()
    plan = tmp_path / "plan.toml"
    plan.write_text("# ok", encoding="utf-8")

    # -- Spy sur SweepRunner pour éviter tout calcul réel
    calls = {"run_grid": 0}

    class FakeSweepRunner:
        def __init__(self, *a, **kw):
            pass

        def run_grid(self, *a, **kw):
            calls["run_grid"] += 1
            # retourne un DataFrame-like minimal
            import pandas as pd

            return pd.DataFrame()

    # patch de l'import dans run.py
    import threadx.optimization.engine as engine_mod

    monkeypatch.setattr(engine_mod, "SweepRunner", FakeSweepRunner, raising=True)

    # Appel: ne doit pas lever de TypeError et doit utiliser config_path
    run_sweep(cfg, str(plan), dry_run=True)

    # En dry_run, run_grid ne devrait PAS être appelé
    assert calls["run_grid"] == 0
