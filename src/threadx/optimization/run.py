"""Command-line interface for running optimization sweeps."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict

from threadx.config import ConfigurationError, load_config_dict
from threadx.indicators.bank import IndicatorBank
from threadx.optimization.engine import SweepRunner
from threadx.optimization.reporting import write_reports
from threadx.optimization.scenarios import ScenarioSpec
from threadx.utils.determinism import set_global_seed
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Deprecated compatibility wrapper around :func:`load_config_dict`."""
    warnings.warn(
        "load_config(config_path) est déprécié. Utilisez "
        "threadx.config.load_config_dict(config_path).",
        DeprecationWarning,
        stacklevel=2,
    )
    config = load_config_dict(config_path)
    logger.info("CFG_LOAD_OK path=%s keys=%d", config_path, len(config or {}))
    return config


def _ensure_mapping(value: Any, path: str, message: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(path, message)
    return value


def validate_cli_config(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigurationError(config_path, "Configuration root must be a mapping")

    _ensure_mapping(
        config.get("dataset"), config_path, "Invalid `dataset`: expected mapping"
    )
    scenario = _ensure_mapping(
        config.get("scenario"), config_path, "Invalid `scenario`: expected mapping"
    )
    params = _ensure_mapping(
        config.get("params"), config_path, "Invalid `params`: expected mapping"
    )
    constraints = _ensure_mapping(
        config.get("constraints"),
        config_path,
        "Invalid `constraints`: expected mapping",
    )

    rules = constraints.get("rules", [])
    if not isinstance(rules, list):
        raise ConfigurationError(
            config_path, "Invalid `constraints.rules`: expected list"
        )
    if any(not isinstance(rule, dict) for rule in rules):
        raise ConfigurationError(
            config_path, "Invalid `constraints.rules`: each entry must be a mapping"
        )

    if scenario.get("type") not in (None, "grid", "monte_carlo"):
        raise ConfigurationError(
            config_path, "scenario.type must be 'grid' or 'monte_carlo'"
        )

    if scenario.get("type") == "monte_carlo":
        n_scenarios = scenario.get("n_scenarios", 0)
        if not isinstance(n_scenarios, int) or n_scenarios <= 0:
            raise ConfigurationError(
                config_path,
                "scenario.n_scenarios must be a positive integer for Monte Carlo",
            )

    for name, block in params.items():
        if not isinstance(block, dict):
            raise ConfigurationError(
                config_path,
                f"Parameter block `{name}` must be a mapping of configuration values",
            )

    return config


def build_scenario_spec(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    scenario = config.get("scenario", {})
    params = config.get("params", {})
    constraints = config.get("constraints", {})

    # Si la config utilise run.* au lieu de scenario.*
    run_config = config.get("run", {})
    if run_config:
        scenario_type = run_config.get("type", "grid")
        scenario_seed = run_config.get("seed", 42)
        scenario_n = run_config.get("n_scenarios", 100)
        scenario_sampler = run_config.get(
            "sampler", "grid" if scenario_type == "grid" else "sobol"
        )
        scenario_params = run_config.get("params", params)
        scenario_constraints = run_config.get(
            "constraints", constraints.get("rules", [])
        )
    else:
        scenario_type = scenario.get("type", "grid")
        scenario_seed = scenario.get("seed", 42)
        scenario_n = scenario.get("n_scenarios", 100)
        scenario_sampler = scenario.get(
            "sampler",
            "sobol" if scenario_type == "monte_carlo" else "grid",
        )
        scenario_params = params
        scenario_constraints = constraints.get("rules", [])

    spec_dict = {
        "type": scenario_type,
        "params": scenario_params,
        "seed": scenario_seed,
        "n_scenarios": scenario_n,
        "sampler": scenario_sampler,
        "constraints": scenario_constraints,
    }

    logger.info(
        "Configuration validée: %s avec %d paramètres",
        spec_dict["type"],
        len(spec_dict.get("params", {})),
    )
    return spec_dict


def run_sweep(config: Dict[str, Any], config_path: str, dry_run: bool = False) -> None:
    validated = validate_cli_config(config, config_path)
    scenario_spec = build_scenario_spec(validated, config_path)

    execution = validated.get("execution", {})
    output = validated.get("output", {})

    set_global_seed(scenario_spec["seed"])

    if dry_run:
        logger.info("=== MODE DRY RUN ===")
        logger.info("Type de sweep: %s", scenario_spec["type"])
        logger.info("Paramètres: %s", list(scenario_spec["params"].keys()))
        logger.info("Seed: %s", scenario_spec["seed"])
        logger.info("GPU activé: %s", execution.get("use_gpu", False))
        logger.info("Cache réutilisé: %s", execution.get("reuse_cache", True))

        if scenario_spec["type"] == "monte_carlo":
            logger.info("Scénarios Monte Carlo: %s", scenario_spec["n_scenarios"])
            logger.info("Sampler: %s", scenario_spec["sampler"])

        logger.info("Configuration valide - prêt pour exécution")
        return

    indicator_bank = IndicatorBank()
    sweep_runner = SweepRunner(
        indicator_bank=indicator_bank,
        max_workers=execution.get("max_workers", 4),
    )

    start_time = time.time()
    try:
        if scenario_spec["type"] == "grid":
            logger.info("Exécution du sweep de grille...")
            results_df = sweep_runner.run_grid(
                scenario_spec,
                reuse_cache=execution.get("reuse_cache", True),
            )
        elif scenario_spec["type"] == "monte_carlo":
            logger.info("Exécution du sweep Monte Carlo...")
            results_df = sweep_runner.run_monte_carlo(
                scenario_spec,
                reuse_cache=execution.get("reuse_cache", True),
            )
        else:  # pragma: no cover - guarded by validation
            raise ConfigurationError(
                config_path,
                f"Type de sweep non supporté: {scenario_spec['type']}",
            )

        if results_df.empty:
            logger.warning("Aucun résultat généré")
        else:
            reports_dir = output.get("reports_dir", "artifacts/reports")
            logger.info(
                "Génération des rapports: %d résultats → %s",
                len(results_df),
                reports_dir,
            )
            created_files = write_reports(
                results_df,
                reports_dir,
                seeds=[scenario_spec["seed"]],
                devices=["CPU", "GPU"],
                gpu_ratios={"5090": 0.75, "2060": 0.25},
                min_samples=execution.get("min_samples", 1000),
            )
            for file_type, file_path in created_files.items():
                logger.info("  %s: %s", file_type, file_path)
    except Exception as exc:
        logger.error("Erreur lors de l'exécution: %s", exc)
        raise
    finally:
        execution_time = time.time() - start_time
        logger.info("Sweep terminé en %.1fs", execution_time)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThreadX Optimization Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples:\n"
            "  python -m threadx.optimization.run --config configs/sweeps/bb_atr_grid.toml\n"
            "  python -m threadx.optimization.run --config configs/sweeps/bb_atr_montecarlo.toml --dry-run"
        ),
    )

    parser.add_argument(
        "--config", "-c", required=True, help="Chemin vers le fichier TOML"
    )
    parser.add_argument("--dry-run", action="store_true", help="Valide sans exécuter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbose")

    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.getLogger("threadx").setLevel(logging.DEBUG)

    config_path = Path(args.config).resolve()

    if not config_path.exists():
        logger.error(f"❌ Fichier de configuration introuvable: {config_path}")
        sys.exit(1)

    try:
        # Chargement et exécution
        config = load_config_dict(str(config_path))
        logger.info(f"Configuration chargée: {args.config}")
        run_sweep(config, str(config_path), dry_run=args.dry_run)

        if not args.dry_run:
            logger.info("✅ Sweep terminé avec succès")

    except ConfigurationError as exc:
        logger.error(exc.user_message)
        logger.debug("Configuration error", exc_info=True)
        sys.exit(2)
    except Exception as exc:
        logger.error("❌ Erreur: %s", exc)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
