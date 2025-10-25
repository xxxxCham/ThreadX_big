"""ThreadX Unified Diversity Pipeline - Option B
=============================================

Pipeline unifié pour traiter les données de diversité crypto avec délégation
complète des calculs d'indicateurs à IndicatorBank.

Architecture Option B :
1. TokenDiversityDataSource : OHLCV uniquement, aucun indicateur
2. IndicatorBank : Tous les calculs d'indicateurs (RSI, MACD, BB, etc.)
3. Pipeline unifié : Orchestration et intégration complète
4. Persistance : Cache/registry pour optimisation

Usage CLI : python -m threadx.data.unified_diversity_pipeline --mode diversity
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, cast

import pandas as pd

from threadx.data.tokens import (
    TokenDiversityDataSource,
    TokenDiversityConfig,
    create_default_config,
)
from threadx.config import load_config_dict
from threadx.indicators.bank import IndicatorBank
from threadx.data.io import write_frame

log = logging.getLogger(__name__)

# Configuration par défaut pour le pipeline diversity
DEFAULT_DIVERSITY_CONFIG = {
    "groups": {
        "L1": ["BTCUSDT", "ETHUSDT"],
        "L2": ["ARBUSDT", "OPUSDT", "MATICUSDT"],
        "DeFi": ["UNIUSDT", "AAVEUSDT", "COMPUSDT"],
        "AI": ["FETUSD", "RENDERUSDT", "AGIXUSDT"],
        "Gaming": ["AXSUSDT", "SANDUSDT", "MANAUSDT"],
        "Meme": ["DOGEUSDT", "SHIBUSD", "PEPEUSDT"],
    },
    "default_timeframes": ["1h", "4h", "1d"],
    "default_indicators": ["rsi", "macd", "bb", "sma_20", "ema_50"],
    "lookback_periods": {"1h": 200, "4h": 100, "1d": 50},
}


class UnifiedDiversityPipeline:
    """
    Pipeline unifié pour traitement des données diversity avec Option B.

    Fonctionnalités :
    - Récupération OHLCV via TokenDiversityDataSource (Option B)
    - Calcul indicateurs via IndicatorBank (délégation complète)
    - Gestion cache et persistance
    - Support batch processing pour optimisation
    - Interface CLI standardisée
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        enable_persistence: bool = True,
    ):
        """
        Initialise le pipeline unifié.

        Args:
            config: Configuration personnalisée (utilise DEFAULT_DIVERSITY_CONFIG sinon)
            cache_dir: Répertoire de cache (auto si None)
            enable_persistence: Activer la persistance
        """
        self.config = config or DEFAULT_DIVERSITY_CONFIG
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/diversity")
        self.enable_persistence = enable_persistence

        # Initialisation TokenDiversityDataSource avec Option B
        groups_dict = cast(Dict[str, List[str]], self.config["groups"])
        provider_config = TokenDiversityConfig(
            groups=groups_dict,
            symbols=self._extract_all_symbols(groups_dict),
            supported_tf=tuple(self.config["default_timeframes"]),
            cache_dir=str(self.cache_dir),
        )
        self.provider = TokenDiversityDataSource(provider_config)

        # Initialisation IndicatorBank pour délégation complète
        self.indicator_bank = IndicatorBank()

        # Registry désactivé (fonctions utilitaires disponibles dans
        # threadx.data.registry)
        self.registry = None

        log.info(
            "UnifiedDiversityPipeline initialisé : %d groupes, %d symboles, "
            "cache=%s, persistence=%s",
            len(self.config["groups"]),
            len(provider_config.symbols),
            self.cache_dir,
            enable_persistence,
        )

    def _extract_all_symbols(self, groups: Dict[str, List[str]]) -> List[str]:
        """Extrait tous les symboles de tous les groupes."""
        all_symbols = set()
        for symbols in groups.values():
            all_symbols.update(symbols)
        return sorted(list(all_symbols))

    def process_symbol(
        self,
        symbol: str,
        timeframe: str = "1h",
        indicators: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Traite un symbole avec Option B : OHLCV + indicateurs via IndicatorBank.

        Args:
            symbol: Symbole à traiter (ex: "BTCUSDT")
            timeframe: Timeframe (ex: "1h", "4h", "1d")
            indicators: Liste d'indicateurs à calculer (défaut depuis config)
            start: Date début optionnelle
            end: Date fin optionnelle

        Returns:
            DataFrame avec OHLCV + indicateurs calculés

        Raises:
            DataNotFoundError: Si symbole/timeframe non supporté
        """
        if indicators is None:
            indicators = cast(List[str], self.config["default_indicators"])

        log.info(
            "Traitement symbole %s@%s avec %d indicateurs",
            symbol,
            timeframe,
            len(indicators),
        )

        # Phase 1 : Récupération OHLCV via provider (Option B - aucun indicateur)
        try:
            ohlcv_df = self.provider.get_frame(symbol, timeframe, start, end)
            log.debug(
                "OHLCV récupéré : %d lignes pour %s@%s",
                len(ohlcv_df),
                symbol,
                timeframe,
            )
        except Exception as e:
            log.error("Échec récupération OHLCV %s@%s: %s", symbol, timeframe, e)
            raise

        # Validation conformité OHLCV (Option B)
        if not self.provider.validate_frame(ohlcv_df):
            raise ValueError(f"DataFrame OHLCV non conforme pour {symbol}@{timeframe}")

        # Phase 2 : Calcul indicateurs via IndicatorBank (délégation complète)
        enriched_df = ohlcv_df.copy()

        for indicator in indicators:
            try:
                log.debug(
                    "Calcul indicateur %s pour %s@%s", indicator, symbol, timeframe
                )

                # Délégation à IndicatorBank selon Option B
                # Paramètres par défaut pour les indicateurs courants
                default_params = {
                    "rsi": {"period": 14},
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "bb": {"period": 20, "std": 2.0},
                    "bollinger": {"period": 20, "std": 2.0},
                    "sma_20": {"period": 20},
                    "ema_50": {"period": 50},
                    "atr": {"period": 14},
                }

                params = cast(
                    Dict[str, Any], default_params.get(indicator, {"period": 14})
                )
                indicator_result = self.indicator_bank.ensure(
                    indicator_type=indicator.replace("_", ""),  # Normalisation
                    params=params,
                    data=ohlcv_df,
                    symbol=symbol,
                    timeframe=timeframe,
                )

                # Intégration résultat dans DataFrame principal
                if isinstance(indicator_result, pd.Series):
                    enriched_df[indicator] = indicator_result
                elif isinstance(indicator_result, pd.DataFrame):
                    # Indicateurs multi-colonnes (ex: MACD, Bollinger Bands)
                    for col in indicator_result.columns:
                        enriched_df[f"{indicator}_{col}"] = indicator_result[col]
                else:
                    log.warning(
                        "Format indicateur non supporté: %s -> %s",
                        indicator,
                        type(indicator_result),
                    )

            except Exception as e:
                log.error(
                    "Échec calcul indicateur %s pour %s@%s: %s",
                    indicator,
                    symbol,
                    timeframe,
                    e,
                )
                # Continuer avec les autres indicateurs
                continue

        # Phase 3 : Persistance optionnelle
        if self.enable_persistence:
            self._save_enriched_data(enriched_df, symbol, timeframe, indicators)

        log.info(
            "Traitement terminé %s@%s : %d colonnes finales",
            symbol,
            timeframe,
            len(enriched_df.columns),
        )

        return enriched_df

    def process_group(
        self,
        group_name: str,
        timeframe: str = "1h",
        indicators: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Traite tous les symboles d'un groupe de diversité.

        Args:
            group_name: Nom du groupe (ex: "L2", "DeFi", "AI")
            timeframe: Timeframe pour tous les symboles
            indicators: Indicateurs à calculer
            limit: Limite nombre de symboles (pour tests)

        Returns:
            Dict {symbol: DataFrame enrichi} pour chaque symbole du groupe
        """
        groups_dict = cast(Dict[str, List[str]], self.config["groups"])
        if group_name not in groups_dict:
            available = list(groups_dict.keys())
            raise ValueError(f"Groupe '{group_name}' inconnu. Disponibles: {available}")

        symbols = self.provider.list_symbols(group=group_name, limit=limit or 100)
        results = {}

        log.info(
            "Traitement groupe '%s' : %d symboles@%s",
            group_name,
            len(symbols),
            timeframe,
        )

        for symbol in symbols:
            try:
                results[symbol] = self.process_symbol(
                    symbol=symbol, timeframe=timeframe, indicators=indicators
                )
            except Exception as e:
                log.error(
                    "Échec traitement %s dans groupe %s: %s", symbol, group_name, e
                )
                continue

        log.info(
            "Groupe '%s' traité : %d/%d symboles réussis",
            group_name,
            len(results),
            len(symbols),
        )

        return results

    def process_all_groups(
        self,
        timeframe: str = "1h",
        indicators: Optional[List[str]] = None,
        limit_per_group: Optional[int] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Traite tous les groupes de diversité.

        Args:
            timeframe: Timeframe global
            indicators: Indicateurs globaux
            limit_per_group: Limite symboles par groupe (pour tests)

        Returns:
            Dict {group_name: {symbol: DataFrame}} structure complète
        """
        all_results = {}

        log.info(
            "Traitement complet : %d groupes@%s", len(self.config["groups"]), timeframe
        )

        for group_name in self.config["groups"].keys():
            try:
                all_results[group_name] = self.process_group(
                    group_name=group_name,
                    timeframe=timeframe,
                    indicators=indicators,
                    limit=limit_per_group,
                )
            except Exception as e:
                log.error("Échec traitement groupe %s: %s", group_name, e)
                continue

        total_symbols = sum(len(group_data) for group_data in all_results.values())
        log.info(
            "Traitement complet terminé : %d symboles traités dans %d groupes",
            total_symbols,
            len(all_results),
        )

        return all_results

    def _save_enriched_data(
        self, df: pd.DataFrame, symbol: str, timeframe: str, indicators: List[str]
    ) -> None:
        """Sauvegarde les données enrichies avec métadonnées."""
        try:
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

                # Chemin de sauvegarde
                filename = f"{symbol}_{timeframe}_enriched.parquet"
                filepath = self.cache_dir / filename

                # Sauvegarde données
                write_frame(df, str(filepath))

                # Enregistrement métadonnées dans registry
                if self.registry:
                    metadata = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "indicators": indicators,
                        "rows": len(df),
                        "columns": list(df.columns),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "filepath": str(filepath),
                    }
                    self.registry.register_dataset(
                        key=f"diversity_{symbol}_{timeframe}", metadata=metadata
                    )

                log.debug("Données enrichies sauvées : %s", filepath)

        except Exception as e:
            log.warning("Échec sauvegarde enrichie %s@%s: %s", symbol, timeframe, e)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Retourne statistiques du pipeline."""
        total_symbols = len(self._extract_all_symbols(self.config["groups"]))

        stats = {
            "total_groups": len(self.config["groups"]),
            "total_symbols": total_symbols,
            "supported_timeframes": list(self.config["default_timeframes"]),
            "default_indicators": self.config["default_indicators"],
            "cache_dir": str(self.cache_dir),
            "persistence_enabled": self.enable_persistence,
            "provider_type": "TokenDiversityDataSource (Option B)",
            "indicator_engine": "IndicatorBank (Délégation complète)",
        }

        # Statistiques par groupe
        for group_name, symbols in self.config["groups"].items():
            stats[f"group_{group_name}_count"] = len(symbols)

        return stats


# ============================================================================
# PIPELINE FUNCTIONS (intégré depuis diversity_pipeline.py)
# ============================================================================


def run_unified_diversity(
    config_path: Optional[str] = None,
    groups: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    timeframe: str = "1h",
    lookback_days: int = 30,
    indicators: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline unifié d'analyse de diversité avec délégation Indicator Bank.

    Étapes :
    1. Configuration TokenDiversityDataSource (Option B)
    2. Récupération données OHLCV brutes
    3. Délégation calculs d'indicateurs à IndicatorBank
    4. Agrégation et analyse de diversité
    5. Sauvegarde artifacts + Registry

    Args:
        config_path: Chemin vers config TOML personnalisée
        groups: Groupes de tokens à analyser (ex: ["L1", "DeFi"])
        symbols: Symboles spécifiques si pas de groupes
        timeframe: Timeframe (ex: "1h", "4h", "1d")
        lookback_days: Période d'analyse en jours
        indicators: Indicateurs à calculer via IndicatorBank
        output_dir: Répertoire de sauvegarde
        save_artifacts: Si True, sauvegarde + enregistrement Registry

    Returns:
        Dict avec clés : ohlcv_data, indicators_data, diversity_metrics
    """
    from datetime import timedelta

    start_time = datetime.now()
    log.info(
        "run_unified_diversity: START - groups=%s symbols=%s tf=%s " "lookback=%dd",
        groups,
        symbols,
        timeframe,
        lookback_days,
    )

    # 1) Configuration du provider TokenDiversity
    if config_path:
        try:
            custom_config = load_config_dict(config_path)
            td_config = TokenDiversityConfig(**custom_config.get("token_diversity", {}))
        except Exception as e:
            log.error("Erreur chargement config %s: %s", config_path, e)
            raise ValueError(f"Invalid config file: {config_path}") from e
    else:
        td_config = create_default_config()

    # 2) Initialisation du provider
    try:
        provider = TokenDiversityDataSource(td_config)
        log.info(
            "TokenDiversityDataSource initialisé: %d symboles",
            len(provider.list_symbols()),
        )
    except Exception as e:
        log.error("Erreur init TokenDiversityDataSource: %s", e)
        raise RuntimeError("Failed to initialize TokenDiversityDataSource") from e

    # 3) Résolution des symboles cibles
    target_symbols = _resolve_target_symbols(provider, groups, symbols)
    if not target_symbols:
        raise ValueError("No symbols resolved from groups/symbols parameters")

    log.info(
        "Symboles cibles résolus: %d - %s",
        len(target_symbols),
        target_symbols[:5],
    )

    # 4) Calcul période de récupération
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)

    # 5) Récupération données OHLCV
    ohlcv_data = {}
    failed_symbols = []

    for symbol in target_symbols:
        try:
            log.debug("Récupération OHLCV: %s@%s", symbol, timeframe)
            df = provider.fetch_ohlcv(symbol, timeframe, start_date, end_date)

            if df is not None and not df.empty:
                ohlcv_data[symbol] = df
                log.debug("OHLCV OK: %s → %d rows", symbol, len(df))
            else:
                failed_symbols.append(symbol)
                log.warning("OHLCV vide: %s", symbol)

        except Exception as e:
            failed_symbols.append(symbol)
            log.error("Erreur OHLCV %s: %s", symbol, e)

    if not ohlcv_data:
        raise RuntimeError("No OHLCV data retrieved for any symbol")

    log.info(
        "OHLCV récupéré: %d/%d symboles (échecs: %s)",
        len(ohlcv_data),
        len(target_symbols),
        failed_symbols[:3] if failed_symbols else "aucun",
    )

    # 6) Délégation calculs d'indicateurs à IndicatorBank
    indicators_data = {}
    if indicators:
        try:
            log.info("Délégation IndicatorBank: %d indicateurs", len(indicators))
            bank = IndicatorBank()

            for symbol, ohlcv_df in ohlcv_data.items():
                try:
                    indicators_result = bank.compute_batch(
                        data=ohlcv_df, indicators=indicators, symbol=symbol
                    )
                    indicators_data[symbol] = indicators_result
                    log.debug(
                        "Indicateurs OK: %s → %d indicateurs",
                        symbol,
                        len(indicators_result),
                    )

                except Exception as e:
                    log.error("Erreur indicateurs %s: %s", symbol, e)

        except Exception as e:
            log.error("Erreur init IndicatorBank: %s", e)

    # 7) Calcul métriques de diversité
    diversity_metrics = _compute_diversity_metrics(ohlcv_data, indicators_data, groups)

    # 8) Sauvegarde artifacts
    artifacts_info = {}
    if save_artifacts:
        artifacts_info = _save_pipeline_artifacts(
            ohlcv_data,
            indicators_data,
            diversity_metrics,
            output_dir or td_config.cache_dir,
            timeframe,
            lookback_days,
        )

    # 9) Résultat pipeline
    duration = (datetime.now() - start_time).total_seconds()
    log.info(
        "run_unified_diversity: SUCCESS - %d symboles, " "%d indicateurs, %.1fs",
        len(ohlcv_data),
        len(indicators_data),
        duration,
    )

    return {
        "ohlcv_data": ohlcv_data,
        "indicators_data": indicators_data,
        "diversity_metrics": diversity_metrics,
        "metadata": {
            "timeframe": timeframe,
            "lookback_days": lookback_days,
            "symbols_count": len(ohlcv_data),
            "failed_symbols": failed_symbols,
            "duration_seconds": duration,
            "artifacts": artifacts_info,
        },
    }


def _resolve_target_symbols(
    provider: TokenDiversityDataSource,
    groups: Optional[List[str]],
    symbols: Optional[List[str]],
) -> List[str]:
    """Résout les symboles cibles à partir des groupes ou symboles."""
    if symbols:
        available_symbols = set(provider.list_symbols())
        resolved = [s for s in symbols if s in available_symbols]
        missing = set(symbols) - set(resolved)

        if missing:
            log.warning("Symboles introuvables: %s", sorted(missing))

        return resolved

    elif groups:
        resolved = []
        for group in groups:
            group_symbols = provider.list_symbols(group=group)
            resolved.extend(group_symbols)
            log.debug("Groupe %s: %d symboles", group, len(group_symbols))

        return list(dict.fromkeys(resolved))  # Préserve l'ordre

    else:
        return provider.list_symbols()[:10]


def _compute_diversity_metrics(
    ohlcv_data: Dict[str, pd.DataFrame],
    indicators_data: Dict[str, pd.DataFrame],
    groups: Optional[List[str]],
) -> pd.DataFrame:
    """Calcul des métriques de diversité inter-tokens."""
    log.info("Calcul métriques de diversité: %d tokens", len(ohlcv_data))

    metrics = []

    for symbol, df in ohlcv_data.items():
        if df.empty:
            continue

        close_prices = df["close"]
        returns = close_prices.pct_change().dropna()

        metric = {
            "symbol": symbol,
            "data_points": len(df),
            "price_start": (close_prices.iloc[0] if len(close_prices) > 0 else None),
            "price_end": (close_prices.iloc[-1] if len(close_prices) > 0 else None),
            "total_return": (
                (close_prices.iloc[-1] / close_prices.iloc[0] - 1)
                if len(close_prices) > 1
                else 0
            ),
            "volatility": returns.std() if len(returns) > 0 else 0,
            "volume_mean": df["volume"].mean(),
            "volume_std": df["volume"].std(),
        }

        # Métriques d'indicateurs (si disponibles)
        if symbol in indicators_data:
            indicators_df = indicators_data[symbol]
            if "rsi_14" in indicators_df.columns:
                metric["rsi_mean"] = indicators_df["rsi_14"].mean()
            if "sma_20" in indicators_df.columns:
                metric["sma_trend"] = (
                    (
                        indicators_df["sma_20"].iloc[-1]
                        / indicators_df["sma_20"].iloc[0]
                        - 1
                    )
                    if len(indicators_df) > 1
                    else 0
                )

        metrics.append(metric)

    diversity_df = pd.DataFrame(metrics)

    if not diversity_df.empty and len(ohlcv_data) > 1:
        # Calcul corrélations inter-tokens
        returns_matrix = {}
        for symbol, df in ohlcv_data.items():
            if not df.empty:
                returns = df["close"].pct_change().dropna()
                if len(returns) > 0:
                    returns_matrix[symbol] = returns

        if len(returns_matrix) > 1:
            returns_df = pd.DataFrame(returns_matrix).fillna(0)
            correlation_matrix = returns_df.corr()

            diversity_scores: List[float] = []
            for symbol in diversity_df["symbol"]:
                if symbol in correlation_matrix.index:
                    corr_with_others = correlation_matrix.loc[symbol].drop(symbol)
                    avg_correlation = (
                        corr_with_others.abs().mean()
                        if len(corr_with_others) > 0
                        else 0
                    )
                    diversity_score = 1 - avg_correlation
                    diversity_scores.append(diversity_score)
                else:
                    diversity_scores.append(0.5)

            diversity_df["diversity_score"] = diversity_scores

    log.info("Métriques diversité calculées: %d tokens", len(diversity_df))
    return diversity_df


def _save_pipeline_artifacts(
    ohlcv_data: Dict[str, pd.DataFrame],
    indicators_data: Dict[str, pd.DataFrame],
    diversity_metrics: pd.DataFrame,
    output_dir: str,
    timeframe: str,
    lookback_days: int,
) -> Dict[str, str]:
    """Sauvegarde les artifacts du pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts = {}

    try:
        # 1) Sauvegarde données OHLCV
        ohlcv_file = (
            output_path / f"diversity_ohlcv_{timeframe}_{lookback_days}d_"
            f"{timestamp}.parquet"
        )
        combined_ohlcv = pd.concat(
            {symbol: df for symbol, df in ohlcv_data.items()},
            names=["symbol", "datetime"],
        ).reset_index()
        write_frame(combined_ohlcv, str(ohlcv_file))
        artifacts["ohlcv"] = str(ohlcv_file)

        # 2) Sauvegarde indicateurs
        if indicators_data:
            indicators_file = (
                output_path / f"diversity_indicators_{timeframe}_{lookback_days}d_"
                f"{timestamp}.parquet"
            )
            combined_indicators = pd.concat(
                {symbol: df for symbol, df in indicators_data.items()},
                names=["symbol", "datetime"],
            ).reset_index()
            write_frame(combined_indicators, str(indicators_file))
            artifacts["indicators"] = str(indicators_file)

        # 3) Sauvegarde métriques diversité
        metrics_file = (
            output_path / f"diversity_metrics_{timeframe}_{lookback_days}d_"
            f"{timestamp}.parquet"
        )
        write_frame(diversity_metrics, str(metrics_file))
        artifacts["metrics"] = str(metrics_file)

        log.info("Artifacts sauvegardés: %s", output_path)
        return artifacts

    except Exception as e:
        log.error("Erreur sauvegarde artifacts: %s", e)
        return {}


def run_diversity_mode(args: argparse.Namespace) -> None:
    """
    Exécute le mode diversity du pipeline unifié.

    Args:
        args: Arguments CLI parsés
    """
    log.info("Démarrage mode diversity avec Option B")

    # Configuration logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Mapping des arguments CLI vers le pipeline
        pipeline_kwargs = {
            "timeframe": args.timeframe,
            "lookback_days": getattr(args, "lookback_days", 30),
            "save_artifacts": not args.no_persistence,
        }

        # Gestion des arguments optionnels
        if hasattr(args, "config") and args.config:
            pipeline_kwargs["config_path"] = args.config

        if args.indicators:
            pipeline_kwargs["indicators"] = (
                args.indicators.split(",")
                if isinstance(args.indicators, str)
                else args.indicators
            )

        if hasattr(args, "output_dir") and args.output_dir:
            pipeline_kwargs["output_dir"] = args.output_dir
        elif args.cache_dir:
            pipeline_kwargs["output_dir"] = args.cache_dir

        # Modes d'exécution
        if hasattr(args, "symbol") and args.symbol:
            # Mode symbole unique
            log.info("Mode symbole unique : %s@%s", args.symbol, args.timeframe)
            pipeline_kwargs["symbols"] = [args.symbol]

        elif hasattr(args, "group") and args.group:
            # Mode groupe
            log.info("Mode groupe : %s@%s", args.group, args.timeframe)
            pipeline_kwargs["groups"] = [args.group]

        else:
            # Mode complet (tous les groupes par défaut)
            log.info("Mode complet : tous groupes@%s", args.timeframe)
            default_groups = list(DEFAULT_DIVERSITY_CONFIG["groups"].keys())
            pipeline_kwargs["groups"] = default_groups

        # Exécution du pipeline unifié
        results = run_unified_diversity(**pipeline_kwargs)

        # Affichage résultats
        metadata = results["metadata"]
        log.info(
            "Pipeline terminé: %d symboles, %d indicateurs, %.1fs",
            metadata["symbols_count"],
            len(results["indicators_data"]),
            metadata["duration_seconds"],
        )

        if results["metadata"]["failed_symbols"]:
            log.warning(
                "Symboles en échec: %s",
                results["metadata"]["failed_symbols"][:3],
            )

        # Sauvegarde optionnelle de résultats spécifiques
        if hasattr(args, "output") and args.output:
            # Export des métriques de diversité
            diversity_metrics = results["diversity_metrics"]
            write_frame(diversity_metrics, args.output)
            log.info("Métriques diversité sauvées : %s", args.output)

    except Exception as e:
        log.error("Erreur pipeline diversity : %s", e)
        sys.exit(1)

    log.info("Mode diversity terminé avec succès")


def main():
    """Point d'entrée CLI pour le pipeline diversity unifié."""
    parser = argparse.ArgumentParser(
        description="ThreadX Unified Diversity Pipeline - Option B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Traitement symbole unique
  python -m threadx.data.unified_diversity_pipeline --mode diversity --symbol BTCUSDT --timeframe 1h

  # Traitement groupe DeFi
  python -m threadx.data.unified_diversity_pipeline --mode diversity --group DeFi --timeframe 4h

  # Traitement complet avec limite par groupe
  python -m threadx.data.unified_diversity_pipeline --mode diversity --timeframe 1d --limit 5

  # Indicateurs personnalisés
  python -m threadx.data.unified_diversity_pipeline --mode diversity --group L2 --indicators rsi macd sma_20
        """,
    )

    parser.add_argument(
        "--mode",
        default="diversity",
        choices=["diversity"],
        help="Mode de traitement (diversity uniquement pour ce module)",
    )

    parser.add_argument(
        "--symbol", type=str, help="Symbole unique à traiter (ex: BTCUSDT)"
    )

    parser.add_argument(
        "--group", type=str, help="Groupe de diversité à traiter (ex: L2, DeFi, AI)"
    )

    parser.add_argument(
        "--timeframe", default="1h", help="Timeframe à utiliser (défaut: 1h)"
    )

    parser.add_argument(
        "--indicators",
        nargs="*",
        help="Indicateurs à calculer (défaut: rsi macd bb sma_20 ema_50)",
    )

    parser.add_argument(
        "--limit", type=int, help="Limite nombre de symboles par groupe"
    )

    parser.add_argument(
        "--cache-dir", type=str, help="Répertoire de cache personnalisé"
    )

    parser.add_argument(
        "--no-persistence", action="store_true", help="Désactiver la persistance"
    )

    parser.add_argument(
        "--output", type=str, help="Fichier de sortie pour mode symbole unique"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Logging détaillé")

    args = parser.parse_args()

    # Validation arguments
    if args.symbol and args.group:
        parser.error("--symbol et --group sont mutuellement exclusifs")

    if args.output and not args.symbol:
        parser.error("--output ne peut être utilisé qu'avec --symbol")

    # Exécution mode diversity
    run_diversity_mode(args)


if __name__ == "__main__":
    main()
