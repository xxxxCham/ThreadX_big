import argparse
import types
import pytest

# Nous testons la signature et l'appel de run_sweep() avec config_path
from threadx.optimization import run as run_mod


def test_run_sweep_signature_includes_config_path():
    fn = run_mod.run_sweep
    # Vérifie que le 2e paramètre positionnel est bien config_path
    params = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
    # Tolerant au premier argument nommé 'config' ou 'cfg'
    assert params[0] in {"config", "cfg"}
    assert params[1] == "config_path"


def test_main_passes_config_path(monkeypatch, tmp_path):
    # Construit un TOML minimal valide
    cfg_dir = tmp_path / "configs" / "sweeps"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "plan.toml"
    cfg.write_text(
        """
        [run]
        iterations = 1
        seed = 0
        verbose = false

        [params]
        timeframe = { value = "1h" }
        lookback_hours = { values = [24] }
        symbols = { values = ["BTCUSDC"] }

        [scoring]
        primary = "sharpe"

        [output]
        dir = "artifacts/sweeps/minimal"
        format = "parquet"
        """,
        encoding="utf-8",
    )

    called = {"args": None}

    def fake_run_sweep(config, config_path, dry_run=False):  # signature attendue
        called["args"] = {"config": config, "config_path": config_path, "dry_run": dry_run}
        # pas d'exécution réelle
        return None

    # Monkeypatch run_sweep pour capturer l'appel depuis main()
    monkeypatch.setattr(run_mod, "run_sweep", fake_run_sweep)

    # Monkeypatch argparse pour injecter notre chemin de config et --dry-run
    def fake_parse_args():
        ns = types.SimpleNamespace()
        ns.config = str(cfg)
        ns.dry_run = True
        ns.verbose = False
        return ns

    monkeypatch.setattr(run_mod.argparse.ArgumentParser, "parse_args", staticmethod(lambda self=None: fake_parse_args()))

    # Exécute main() qui doit appeler fake_run_sweep avec le bon config_path
    run_mod.main()

    assert called["args"] is not None
    assert called["args"]["config_path"] == str(cfg)
