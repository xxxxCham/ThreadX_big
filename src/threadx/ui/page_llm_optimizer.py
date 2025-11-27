"""
Page Streamlit: Optimisation Multi-LLM
======================================

Interface pour le système collaboratif d'agents LLM (Analyst + Strategist).
Workflow: Sweep → Analyse → Propositions → Tests → Visualisation
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Ajouter src au path si nécessaire
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os

from threadx.data_access import (
    discover_tokens_and_timeframes,
    get_timeframe_performance_estimate,
    get_top_tokens_by_volume,
    get_top_tokens_from_reference,
    load_ohlcv,
)
from threadx.indicators.bank import IndicatorBank, IndicatorSettings
from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.strategist import Strategist
from threadx.llm.model_router import ModelRouter, TaskType
from threadx.llm.ollama_manager import prepare_for_llm_run
from threadx.llm.run_report import (
    LLMRunReport,
    RunIndex,
    create_report_from_run,
)
from threadx.llm.winning_strategies import (
    WinningCriteria,
    WinningStrategiesManager,
    get_winning_manager,
    save_winning_strategy,
)
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec, generate_param_grid
from threadx.ui.backtest_bridge import run_backtest_gpu
from threadx.ui.strategy_registry import (
    SWEEP_PRESETS,
    get_sweep_preset,
    list_strategies,
    parameter_specs_for,
    resolve_range,
)
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def _normalize_param_type(raw_type: str | None) -> str:
    if raw_type in {"int", "integer"}:
        # Streamlit slider expects strict int/float typing consistency
        return "int"
    if raw_type in {"float", "number", None}:
        return "float"
    return raw_type


def _generate_sweep_values(
    min_val: float, max_val: float, n_values: int, param_type: str, step: float | None
) -> list[Any]:
    if n_values <= 1:
        values = [min_val]
    else:
        span = max_val - min_val
        values = [min_val + i * span / (n_values - 1) for i in range(n_values)]

    if step:
        snapped: list[float] = []
        for v in values:
            offset = v - min_val
            snapped_val = min_val + round(offset / step) * step
            snapped_val = max(min_val, min(snapped_val, max_val))
            snapped.append(snapped_val)
        values = snapped

    if param_type == "int":
        values = [int(round(v)) for v in values]

    return values


def unload_ollama_model(model_name: str) -> bool:
    """
    Décharge explicitement un modèle Ollama de la mémoire.

    Args:
        model_name: Nom du modèle à décharger (ex: "deepseek-r1:8b")

    Returns:
        True si succès, False sinon
    """
    import requests

    try:
        # Appeler l'API Ollama pour garder le modèle 0 secondes = déchargement immédiat
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": model_name,
                "keep_alive": 0,  # 0 = décharger immédiatement
                "prompt": "",
            },
            timeout=5,
        )
        success = response.status_code == 200
        response.close()  # Fermer la connexion explicitement
        return success
    except Exception as e:
        # Ignorer silencieusement les erreurs de déchargement
        return False


def render_page():
    """Affiche la page d'optimisation Multi-LLM."""

    st.title("🤖 Optimisation Multi-LLM")
    st.markdown(
        """
    **Système collaboratif d'agents LLM** pour optimisation automatique de stratégies.

    **Workflow**:
    1. 🔄 Sweep GPU → Test multiple configurations
    2. 🧠 Analyst (deepseek-r1:32b) → Analyse quantitative & patterns
    3. 🎨 Strategist (deepseek-r1:32b) → Propositions créatives (cohérence maximale)
    4. ✅ Tests automatiques → Validation performances
    5. 📊 Visualisation → Comparaison résultats

    💡 **Nouveau**: Les deux agents utilisent désormais le même modèle (DeepSeek-R1 32B) pour une cohérence optimale.
    """
    )

    # Vérifier prérequis
    with st.expander("⚙️ Prérequis & Configuration", expanded=False):
        check_prerequisites()

    # Aide optimisation Ollama GPU
    with st.expander(
        "🔧 Résoudre erreurs CUDA Ollama (saturation GPU)", expanded=False
    ):
        st.markdown(
            """
        **Si vous rencontrez une erreur `CUDA error` ou `llama runner process has terminated`**,
        cela indique généralement une saturation de la mémoire GPU lors du chargement/inférence du modèle.

        ### 📊 Diagnostic rapide
        """
        )

        st.code(
            """# Vérifier occupation GPU
nvidia-smi
nvidia-smi -q -d MEMORY""",
            language="powershell",
        )

        st.markdown(
            """
        ### ⚙️ Réglages recommandés (Ollama)

        **1. Localiser la configuration du modèle**
        - Dossier modèles Ollama : `%USERPROFILE%\\.ollama\\models\\`
        - Cherchez le fichier `config.json` ou `modelfile` du modèle utilisé

        **2. Appliquer ces paramètres (sauvegardez l'original avant)**
        """
        )

        config_snippet = {
            "use_mmap": True,
            "num_gpu_layers": 48,
            "batch_size": 256,
            "num_threads": 16,
            "main_gpu": 0,
        }

        st.json(config_snippet)

        st.markdown(
            """
        **Explications :**
        - `use_mmap: true` → Réduit les pics mémoire lors du chargement
        - `num_gpu_layers: 48` → Moins de layers GPU = plus de marge (au lieu de 63)
        - `batch_size: 256` → Réduit la taille batch (au lieu de 512)
        - `main_gpu: 0` → Utilisez le GPU avec le plus de mémoire (vérifiez avec `nvidia-smi`)

        **3. Redémarrer Ollama proprement**
        """
        )

        st.code(
            """# Arrêter Ollama
Stop-Process -Name ollama -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Vérifier GPU libéré
nvidia-smi

# Relancer Ollama
ollama serve""",
            language="powershell",
        )

        st.markdown(
            """
        **4. Tester avec une requête courte**
        """
        )

        st.code(
            """python -c \"from threadx.llm.client import LLMClient
from threadx.llm.ollama_manager import prepare_for_llm_run; c = LLMClient(model='deepseek-r1:8b'); print(c.complete('Test ok'))\" """,
            language="powershell",
        )

        st.warning(
            """
        **Si le problème persiste :**
        - Essayez un modèle plus petit : `deepseek-r1:1.5b` ou `gemma2:2b`
        - Réduisez encore `num_gpu_layers` : essayez 40, puis 32
        - Fermez autres applications GPU (browsers, jeux, Docker, etc.)
        - En dernier recours : utilisez un modèle CPU-only (perte de performance)
        """
        )

    st.divider()

    # Configuration de base
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Configuration Sweep")

        granularity_factor = st.slider(
            "Granularité globale",
            min_value=0.2,
            max_value=0.8,
            value=0.5,
            step=0.05,
            help="Multiplicateur de pas (cohérent avec la page Optimisation). 0.2 = très fin, 0.8 = plus grossier.",
        )

        strategy_name = st.selectbox(
            "Stratégie",
            options=list_strategies(),  # Garder en phase avec le registry
            index=0,  # MA_Crossover sélectionné par défaut
            help="Stratégie à optimiser",
        )

        # Récupérer specs de la stratégie
        param_specs = parameter_specs_for(strategy_name)

        st.markdown("**Paramètres du sweep:**")
        sweep_params = {}

        for param_name, spec in param_specs.items():
            param_type = _normalize_param_type(spec.get("type", "number"))
            is_preset = False  # Flag pour savoir si c'est un préréglage

            if param_type == "boolean":
                sweep_params[param_name] = [False, True]
                st.caption(f"✓ {param_name}: [False, True]")
                continue  # Passer au paramètre suivant

            # Utiliser préréglages globaux si disponibles (source unique)
            preset = get_sweep_preset(param_name)
            base_step = spec.get("step")
            n_values = 3  # min / mid / max par défaut
            local_granularity = st.slider(
                f"Granularité {param_name}",
                min_value=0.2,
                max_value=0.8,
                value=0.5,
                step=0.05,
                help="Affinage spécifique de ce paramètre (multiplicateur de pas, cohérent avec la page Optimisation).",
                key=f"gran_{param_name}",
            )
            if preset:
                min_val = preset["min"]
                max_val = preset["max"]
                base_step = preset.get("step", base_step)
                step = (
                    base_step * granularity_factor * local_granularity
                    if base_step is not None
                    else None
                )
                n_values = max(2, preset.get("n_values", 3))
                is_preset = True
                # Afficher la raison du préréglage
                reason = SWEEP_PRESETS[param_name]["reason"]
                st.caption(
                    f"🔒 {param_name}: {min_val} → {max_val} "
                    f"(3 points, pas~{step or 'auto'}) - {reason}"
                )
            else:
                opt_min, opt_max = resolve_range(spec)
                min_val = opt_min if opt_min is not None else spec.get("min", 0)
                max_val = opt_max if opt_max is not None else spec.get("max", 100)

                # Sélection de la plage (alignée sur la page Optimisation)
                if param_type == "int":
                    min_val = int(round(min_val))
                    max_val = int(round(max_val))
                    step_local = base_step if base_step is not None else 1
                    selected_min, selected_max = st.slider(
                        f"Plage {param_name}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        step=int(step_local),
                        key=f"range_{param_name}",
                    )
                else:
                    min_val = float(min_val)
                    max_val = float(max_val)
                    step_local = (
                        base_step
                        if base_step is not None
                        else (max_val - min_val) / 10 or 0.1
                    )
                    selected_min, selected_max = st.slider(
                        f"Plage {param_name}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        step=step_local,
                        key=f"range_{param_name}",
                    )

                # Remplacer min/max par la plage sélectionnée et appliquer la granularité globale sur le pas
                min_val, max_val = selected_min, selected_max
                base_step = step_local
                step = (
                    base_step * granularity_factor * local_granularity
                    if base_step is not None
                    else None
                )

            # Générer valeurs avec protection division par zéro et validation
            if min_val is None or max_val is None:
                st.warning(f"⚠️ Valeurs min/max invalides pour {param_name}, ignoré")
                continue
            if step is not None and step <= 0:
                step = None
            values = _generate_sweep_values(
                min_val, max_val, n_values, param_type, step
            )
            try:
                values = sorted(list(dict.fromkeys(values)))
            except Exception:
                pass

            sweep_params[param_name] = values

            # Afficher les valeurs générées
            st.caption(f"✓ {param_name}: {values}")

        total_configs = 1
        for vals in sweep_params.values():
            total_configs *= len(vals)

        st.info(f"**Total configurations**: {total_configs}")

        # === VISUALISATION DE L'ESPACE DE RECHERCHE ===
        if sweep_params:
            st.markdown("#### 📊 Visualisation de l'Espace de Recherche")

            # Sélecteur de type de visualisation
            viz_type = st.radio(
                "Type de visualisation",
                [
                    "🎯 Radar normalisé",
                    "📊 Barres horizontales",
                    "📈 Barres avec échelles individuelles",
                ],
                horizontal=True,
                help="Radar = vue d'ensemble, Barres = valeurs précises par paramètre",
                key="llm_viz_type",
            )

            # Préparer les données depuis sweep_params
            spans_raw = []
            labels = []
            min_values = []
            max_values = []

            for param_name, values in sweep_params.items():
                if len(values) > 0:
                    min_v = min(values)
                    max_v = max(values)
                    span = abs(max_v - min_v)
                    spans_raw.append(span)
                    labels.append(param_name)
                    min_values.append(min_v)
                    max_values.append(max_v)

            if viz_type == "🎯 Radar normalisé":
                # === RADAR CHART (normalisé) ===
                spans_normalized = []
                hover_texts = []

                # Normaliser chaque span à 100%
                max_span = max(spans_raw) if spans_raw else 1
                for i, (key, span) in enumerate(zip(labels, spans_raw)):
                    spans_normalized.append((span / max_span) * 100)
                    hover_texts.append(
                        f"{key}<br>Min: {min_values[i]:.4g}<br>Max: {max_values[i]:.4g}<br>Span: {span:.4g}"
                    )

                fig_dist = go.Figure(
                    data=go.Scatterpolar(
                        r=spans_normalized,
                        theta=labels,
                        fill="toself",
                        line=dict(color="#26a69a", width=2),
                        fillcolor="rgba(38, 166, 154, 0.3)",
                        hovertext=hover_texts,
                        hoverinfo="text",
                    )
                )
                fig_dist.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 110],
                            tickmode="linear",
                            tick0=0,
                            dtick=25,
                            ticksuffix="%",
                            showticklabels=True,
                        ),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    showlegend=False,
                    height=400,
                    margin=dict(l=60, r=60, t=40, b=40),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.caption("📊 Échelle normalisée 0-100% (hover pour valeurs réelles)")

            elif viz_type == "📊 Barres horizontales":
                # === BARRES HORIZONTALES (valeurs normalisées) ===
                max_span = max(spans_raw) if spans_raw else 1
                spans_normalized = [(s / max_span) * 100 for s in spans_raw]

                fig_dist = go.Figure(
                    data=go.Bar(
                        y=labels,
                        x=spans_normalized,
                        orientation="h",
                        marker=dict(
                            color=spans_normalized,
                            colorscale="Teal",
                            showscale=False,
                        ),
                        text=[
                            f"{s:.1f}%<br>({spans_raw[i]:.4g})"
                            for i, s in enumerate(spans_normalized)
                        ],
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>Min: %{customdata[0]:.4g}<br>Max: %{customdata[1]:.4g}<br>Span: %{customdata[2]:.4g}<extra></extra>",
                        customdata=list(zip(min_values, max_values, spans_raw)),
                    )
                )
                fig_dist.update_layout(
                    xaxis=dict(title="Étendue normalisée (%)", range=[0, 110]),
                    yaxis=dict(title=""),
                    height=max(300, len(labels) * 50),
                    margin=dict(l=150, r=80, t=20, b=40),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.caption(
                    "📊 Barres horizontales avec valeurs normalisées (hover pour détails)"
                )

            else:  # Barres avec échelles individuelles
                # === SUBPLOTS avec échelle par paramètre ===
                from plotly.subplots import make_subplots

                n_params = len(labels)
                fig_dist = make_subplots(
                    rows=n_params,
                    cols=1,
                    subplot_titles=labels,
                    vertical_spacing=0.08,
                    specs=[[{"type": "bar"}] for _ in range(n_params)],
                )

                for i, (label, min_v, max_v, span) in enumerate(
                    zip(labels, min_values, max_values, spans_raw), 1
                ):
                    fig_dist.add_trace(
                        go.Bar(
                            x=[span],
                            y=[label],
                            orientation="h",
                            marker=dict(color="#26a69a"),
                            text=f"{min_v:.4g} → {max_v:.4g}",
                            textposition="outside",
                            showlegend=False,
                            hovertemplate=f"<b>{label}</b><br>Min: {min_v:.4g}<br>Max: {max_v:.4g}<br>Span: {span:.4g}<extra></extra>",
                        ),
                        row=i,
                        col=1,
                    )
                    # Échelle individuelle par axe
                    fig_dist.update_xaxes(range=[0, span * 1.2], row=i, col=1)

                fig_dist.update_layout(
                    height=max(400, n_params * 80),
                    margin=dict(l=150, r=80, t=60, b=40),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.caption("📈 Chaque paramètre a sa propre échelle (valeurs réelles)")

            st.plotly_chart(
                fig_dist, use_container_width=True, key="llm_param_distribution"
            )

    with col2:
        st.subheader("🤖 Configuration LLM")

        # Mode économie mémoire pour éviter de charger plusieurs modèles simultanément
        memory_saver = st.checkbox(
            "💾 Mode économie mémoire",
            value=True,
            help="Utilise un seul modèle pour tous les agents et décharge la mémoire entre chaque étape. Recommandé pour éviter la saturation RAM.",
        )

        # Liste des modèles disponibles avec infos GPU (installés localement 26/11/2025)
        available_models_info = [
            ("deepseek-r1:70b", "DeepSeek-R1 70B (~42GB) - ARCHITECT PREMIUM"),
            ("deepseek-r1:32b", "DeepSeek-R1 32B (~19GB) - ARCHITECT & BUILDER"),
            ("gemma3:27b", "Gemma 3 27B (~17GB) - GUEST"),
            ("deepseek-r1-distill:14b", "DeepSeek-R1 Distill 14B (~9GB) - GUEST"),
            ("gemma3:12b", "Gemma 3 12B (~8GB) - GUEST"),
            ("deepseek-r1:8b", "DeepSeek-R1 8B (~5GB) - BUILDER RAPIDE"),
            ("mistral:7b-instruct", "Mistral 7B (~4GB) - ULTRA LÉGER"),
        ]

        model_names = [m[0] for m in available_models_info]
        model_labels = [f"{m[0]} - {m[1]}" for m in available_models_info]

        # Initialiser le routeur de modèles
        router = ModelRouter()

        st.info(
            "💡 **Stratégie Multi-LLM Active** : Architecte + Bâtisseur + Auditeurs (rotation)"
        )

        # Modèles installés localement (vérifiés 26/11/2025)
        available_models = [
            "deepseek-r1:70b",  # ~42GB - Meilleur raisonnement
            "deepseek-r1:32b",  # ~19GB - Bon compromis
            "qwen2.5:32b",  # ~19GB - Alibaba, polyvalent
            "gemma3:27b",  # ~17GB - Google, rapide
            "deepseek-r1-distill:14b",  # ~9GB - Distillé
            "gemma3:12b",  # ~8GB - Compact
            "deepseek-r1:8b",  # ~5GB - Très rapide
            "mistral:7b-instruct",  # ~4GB - Ultra léger
        ]

        # Nettoyer les valeurs de session obsolètes (modèles non installés)
        for key in ["llm_architect_model", "llm_builder_model", "llm_guest_models"]:
            if key in st.session_state:
                val = st.session_state[key]
                if isinstance(val, str) and val not in available_models:
                    del st.session_state[key]
                elif isinstance(val, list):
                    cleaned = [m for m in val if m in available_models]
                    if cleaned != val:
                        st.session_state[key] = cleaned if cleaned else ["gemma3:27b"]

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            architect_model_ui = st.selectbox(
                "🏗️ Architecte",
                options=available_models,
                index=(
                    available_models.index("deepseek-r1:32b")
                    if "deepseek-r1:32b" in available_models
                    else 0
                ),
                key="llm_architect_model",
                help="Modèle pour l'étape Analyst/Architecte (raisonnement approfondi)",
            )
        with col_m2:
            builder_model_ui = st.selectbox(
                "🔨 Bâtisseur",
                options=available_models,
                index=(
                    available_models.index("deepseek-r1:8b")
                    if "deepseek-r1:8b" in available_models
                    else 0
                ),
                key="llm_builder_model",
                help="Modèle pour l'étape Strategist/Bâtisseur (itérations rapides)",
            )
        with col_m3:
            guest_models_ui = st.multiselect(
                "👀 Auditeurs (rotation)",
                options=available_models,
                default=["gemma3:27b", "gemma3:12b"],
                key="llm_guest_models",
                help="Liste des modèles qui tourneront en rotation pour l'audit",
            )

        # Afficher les modèles qui seront utilisés
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("🏗️ Architecte", architect_model_ui, help="Initialisation robuste")
        with col_m2:
            st.metric("🔨 Bâtisseur", builder_model_ui, help="Optimisation itérative")
        with col_m3:
            st.metric(
                "👀 Auditeurs",
                "Rotation",
                help=f"Rotation: {', '.join(guest_models_ui) if guest_models_ui else 'aucun'}",
            )

        # Pour compatibilité avec le reste du code, on définit des valeurs par défaut
        # Mais le routeur sera prioritaire dans la boucle d'optimisation
        router = ModelRouter(
            architect_model=architect_model_ui,
            builder_model=builder_model_ui,
            guest_models=guest_models_ui,
        )

        analyst_model = router.get_model_for_task(TaskType.INITIALIZATION)
        strategist_model = router.get_model_for_task(
            TaskType.OPTIMIZATION, step_number=st.session_state.get("llm_step", 1)
        )

        n_proposals = st.slider(
            "Nombre de propositions",
            min_value=1,
            max_value=5,
            value=3,
            help="Propositions créatives générées par Strategist",
        )

        top_n_analysis = st.slider(
            "Top N configs à analyser",
            min_value=3,
            max_value=10,
            value=5,
            help="Nombre de meilleures configs analysées par Analyst",
        )

        # 🆕 Mode Boucle Continue
        st.markdown("#### 🔁 Mode d'Optimisation")

        # Mode continu vs limité
        continuous_mode = st.checkbox(
            "♾️ **Mode Continu Autonome**",
            value=False,
            key="llm_continuous_mode",
            help="La boucle s'exécute indéfiniment jusqu'à ce que vous cliquiez STOP. Les stratégies gagnantes sont automatiquement sauvegardées.",
        )

        if continuous_mode:
            st.info(
                "🔄 **Mode Continu actif**: L'optimisation tournera en boucle jusqu'à l'arrêt manuel. "
                "Les stratégies gagnantes (Sharpe > seuil) seront automatiquement sauvegardées dans `reports/winning_strategies/`."
            )
            max_iterations = 999  # Pseudo-infini

            # Critères pour sauvegarder une stratégie gagnante
            col_crit1, col_crit2 = st.columns(2)
            with col_crit1:
                min_sharpe_threshold = st.number_input(
                    "Seuil Sharpe minimum",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.3,
                    step=0.1,
                    key="llm_min_sharpe_threshold",
                    help="Une stratégie avec Sharpe >= ce seuil sera sauvegardée comme 'gagnante'",
                )
            with col_crit2:
                min_cross_sharpe = st.number_input(
                    "Sharpe moyen cross-test min",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.0,
                    step=0.05,
                    key="llm_min_cross_sharpe",
                    help="Sharpe moyen minimum en cross-testing pour validation (0 = désactivé)",
                )
        else:
            max_iterations = st.slider(
                "Nombre d'itérations max",
                min_value=1,
                max_value=10,
                value=3,
                key="llm_max_iterations",
                help="Boucle d'amélioration: Sweep→Analyst→Strategist→Test→(répéter si amélioration)",
            )
            min_sharpe_threshold = 0.3
            min_cross_sharpe = 0.0

        # Cross-testing multi-tokens/timeframes
        enable_cross_testing = st.checkbox(
            "🌍 Cross-testing multi-tokens/timeframes",
            value=False,
            key="llm_enable_cross_testing",
            help="Valider la meilleure config sur d'autres tokens et timeframes",
        )

        if enable_cross_testing:
            # Découvrir dynamiquement les tokens et timeframes disponibles
            available_tokens, available_timeframes = discover_tokens_and_timeframes()

            # Timeframes pertinents pour le cross-testing (court/moyen terme)
            target_timeframes = ["3m", "5m", "15m", "30m", "1h"]
            filtered_timeframes = [
                tf for tf in target_timeframes if tf in available_timeframes
            ]
            if not filtered_timeframes:
                filtered_timeframes = (
                    available_timeframes[:5] if available_timeframes else ["1h"]
                )

            # === Section Sélection Intelligente ===
            st.markdown("#### 🎯 Sélection Intelligente des Tokens")

            col_auto, col_manual = st.columns([1, 1])

            with col_auto:
                auto_select_tokens = st.checkbox(
                    "🤖 Auto-sélection Top 10 par Volume",
                    value=False,
                    key="llm_auto_select_tokens",
                    help="Sélectionne automatiquement les 10 tokens les plus tradés",
                )

                if auto_select_tokens:
                    # Initialiser ou charger le cache des top tokens
                    if "top_tokens_cache" not in st.session_state:
                        with st.spinner(
                            "📊 Chargement Top 10 tokens par capitalisation..."
                        ):
                            top_tokens_data = get_top_tokens_from_reference(
                                n=10, filter_available=True
                            )
                            st.session_state.top_tokens_cache = top_tokens_data
                    else:
                        top_tokens_data = st.session_state.top_tokens_cache

                    if top_tokens_data:
                        # Afficher le classement
                        st.success(
                            f"✅ Top {len(top_tokens_data)} tokens (par Market Cap):"
                        )

                        # Mini-tableau des tokens avec market cap
                        token_display = []
                        for t in top_tokens_data:
                            mcap = t.get(
                                "total_volume", 0
                            )  # market_cap stocké dans total_volume
                            if mcap and mcap > 1e9:
                                mcap_str = f"${mcap/1e9:.1f}B"
                            elif mcap and mcap > 1e6:
                                mcap_str = f"${mcap/1e6:.0f}M"
                            else:
                                mcap_str = "N/A"
                            name = t.get("name", t["symbol"])
                            token_display.append(
                                f"{t['rank']}. **{t['symbol']}** ({name}) - {mcap_str}"
                            )

                        # Afficher en 2 colonnes
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            for item in token_display[:5]:
                                st.markdown(item)
                        with col_t2:
                            for item in token_display[5:]:
                                st.markdown(item)

                        # Extraire les symboles
                        cross_tokens = [t["symbol"] for t in top_tokens_data]
                    else:
                        st.warning(
                            "⚠️ Fichier de référence non trouvé ou tokens non disponibles"
                        )
                        auto_select_tokens = False

                # Bouton pour rafraîchir le cache
                if auto_select_tokens:
                    if st.button("🔄 Rafraîchir classement", key="refresh_top_tokens"):
                        st.session_state.pop("top_tokens_cache", None)
                        st.rerun()

            with col_manual:
                auto_select_timeframes = st.checkbox(
                    "⏰ Auto-sélection Timeframes optimaux",
                    value=False,
                    key="llm_auto_select_timeframes",
                    help="Sélectionne les timeframes avec la meilleure qualité de données",
                )

                if auto_select_timeframes:
                    # Analyser les timeframes pour le token principal
                    primary_token = st.session_state.get("llm_token", "BTCUSDC")

                    if "timeframe_analysis_cache" not in st.session_state:
                        with st.spinner("⏳ Analyse des timeframes..."):
                            tf_analysis = get_timeframe_performance_estimate(
                                primary_token, filtered_timeframes
                            )
                            st.session_state.timeframe_analysis_cache = tf_analysis
                    else:
                        tf_analysis = st.session_state.timeframe_analysis_cache

                    if tf_analysis:
                        st.success(f"✅ Analyse timeframes pour {primary_token}:")

                        for tf_data in tf_analysis[:5]:
                            score_pct = tf_data["data_quality_score"] * 100
                            st.markdown(
                                f"• **{tf_data['timeframe']}**: "
                                f"📊 {tf_data['bars_count']} bars, "
                                f"✓ {tf_data['completeness']*100:.0f}% complet, "
                                f"⚡ Score: {score_pct:.0f}%"
                            )

                        # Sélectionner les 3 meilleurs timeframes
                        cross_timeframes = [tf["timeframe"] for tf in tf_analysis[:3]]
                    else:
                        st.warning("⚠️ Impossible d'analyser les timeframes")
                        auto_select_timeframes = False

            st.markdown("---")
            st.markdown("#### 📝 Sélection Manuelle")

            col_tokens, col_timeframes = st.columns(2)
            with col_tokens:
                if not auto_select_tokens:
                    # Tokens par défaut s'ils existent
                    default_tokens = [
                        t
                        for t in ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
                        if t in available_tokens
                    ]
                    if not default_tokens and available_tokens:
                        default_tokens = available_tokens[:3]

                    cross_tokens = st.multiselect(
                        "Tokens à tester",
                        options=available_tokens if available_tokens else ["BTCUSDC"],
                        default=default_tokens if default_tokens else ["BTCUSDC"],
                        key="llm_cross_tokens",
                        help=f"{len(available_tokens)} tokens disponibles dans les données",
                    )
                else:
                    st.info(f"🎯 {len(cross_tokens)} tokens auto-sélectionnés")
                    st.multiselect(
                        "Tokens sélectionnés (lecture seule)",
                        options=cross_tokens,
                        default=cross_tokens,
                        key="llm_cross_tokens_readonly",
                        disabled=True,
                    )

            with col_timeframes:
                if not auto_select_timeframes:
                    # Timeframes par défaut
                    default_tfs = [
                        tf for tf in ["15m", "30m", "1h"] if tf in filtered_timeframes
                    ]
                    if not default_tfs:
                        default_tfs = (
                            filtered_timeframes[:3] if filtered_timeframes else ["1h"]
                        )

                    cross_timeframes = st.multiselect(
                        "Timeframes à tester",
                        options=filtered_timeframes,
                        default=default_tfs,
                        key="llm_cross_timeframes",
                        help="3m, 5m, 15m, 30m, 1h recommandés pour cross-testing",
                    )
                else:
                    st.info(f"⏰ {len(cross_timeframes)} timeframes auto-sélectionnés")
                    st.multiselect(
                        "Timeframes sélectionnés (lecture seule)",
                        options=cross_timeframes,
                        default=cross_timeframes,
                        key="llm_cross_timeframes_readonly",
                        disabled=True,
                    )

            # Afficher le nombre total de tests
            n_cross_tests = len(cross_tokens) * len(cross_timeframes)
            st.info(
                f"🧪 **{n_cross_tests}** combinaisons token/timeframe seront testées"
            )
        else:
            cross_tokens = ["BTCUSDC"]
            cross_timeframes = ["1h"]

        # Checkbox analyse IA (cochée par défaut)
        enable_ai_analysis = st.checkbox(
            "⚡ Activer l'analyse IA pour la meilleure configuration",
            value=True,
            help="Les LLM analyseront les résultats pour proposer des optimisations",
        )

        # Checkbox streaming (afficher réflexion en temps réel)
        enable_streaming = st.checkbox(
            "🧠 Afficher réflexion LLM en temps réel",
            value=True,
            help="Voir le raisonnement Chain-of-Thought de DeepSeek-R1 pendant la génération",
        )

    # Configuration depuis la sidebar (valeurs globales)
    st.markdown("### ⚙️ Configuration Moteur de Sweep")

    # Récupérer les valeurs globales de la sidebar
    use_gpu = True  # Toujours activé
    use_multigpu = st.session_state.get("global_use_multigpu", True)
    max_workers = st.session_state.get("global_workers", 30)
    feeder_aggr = st.session_state.get("global_feeder_aggr", 16)

    # Afficher statut de configuration
    st.info(
        f"🎛️ Configuration depuis **sidebar** : {st.session_state.get('global_perf_profile', '⚖️ Équilibré')}"
    )

    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        st.metric("🖥️ GPU", "RTX 5080/5090")
    with col_status2:
        if use_multigpu:
            st.metric("🔥 Multi-GPU", "✅ Activé", help="5080/5090 (66%) + 2060 (34%)")
        else:
            st.metric("🔥 Multi-GPU", "❌ Désactivé", help="5080/5090 uniquement")
    with col_status3:
        st.metric("⚡ Workers", f"{max_workers}", help=f"Feeder: {feeder_aggr}")

    st.caption(
        "💡 Modifier la configuration dans la **sidebar** (section Configuration Globale)"
    )

    # Réglages avancés (ProcessPool seulement)
    st.markdown("#### Réglages avancés")
    col_processpool = st.columns(1)[0]

    with col_processpool:
        force_processpool = st.checkbox(
            "Forcer ProcessPool (CPU-bound)",
            value=st.session_state.get("llm_sweep_force_processpool", True),
            key="llm_sweep_force_processpool",
            help="Active un pool de processus (plus coûteux en mémoire) quand la stratégie est GIL-bound",
        )

    # Consignes pour les LLM
    if enable_ai_analysis:
        with st.expander("📋 Consignes pour les Agents LLM", expanded=False):
            st.markdown(
                """
            **Instructions système pour Analyst & Strategist** :

            🗂️ **Fichiers à consulter (lecture seule)** :
            - Registry paramètres : `src/threadx/ui/strategy_registry.py` (section REGISTRY)
            - Implémentation stratégie : `src/threadx/strategy/<strategy_name>.py`
              - `MA_Crossover` → `ma_crossover.py`
              - `EMA_Cross` → `ema_cross.py`
              - `ATR_Channel` → `atr_channel.py`
              - `Bollinger_Dual` → `bollinger_dual.py`
            - Paramètres en cours : ceux fournis dans le sweep et la baseline affichée dans l'UI

            🎯 **Objectifs prioritaires** :
            - Maximiser le Sharpe Ratio (risque/rendement)
            - Minimiser le drawdown maximum
            - Maintenir un win rate > 50%
            - Optimiser le nombre de trades (ni trop, ni trop peu)

            🧭 **Approche et itérations** :
            - Lire les meilleurs runs du sweep (top Sharpe) et repérer les paramètres communs
            - Proposer de petites variations incrémentales (±1 pas) sur 1-3 paramètres max
            - Justifier chaque changement par un pattern observé + impact attendu
            - Préparer un plan d'essai clair : param -> nouvelle valeur -> raison

            ⚠️ **Contraintes** :
            - `risk_per_trade` : rester dans [0.005, 0.02]
            - `max_hold_bars` : adapter à la volatilité détectée
            - Stop Loss / Take Profit : ratio min 1:1.5
            - Toujours respecter min/max/step du registry pour chaque paramètre

            💡 **Recommandations** :
            - Favoriser la robustesse à la performance brute
            - Tester des valeurs sur plusieurs régimes de marché (range wide vs range tight)
            - Documenter le raisonnement en 2 phrases max par proposition
            """
            )

    st.divider()

    # Initialiser le state pour la boucle continue
    if "continuous_running" not in st.session_state:
        st.session_state.continuous_running = False
    if "continuous_iteration" not in st.session_state:
        st.session_state.continuous_iteration = 0
    if "winning_strategies_count" not in st.session_state:
        st.session_state.winning_strategies_count = 0

    # Boutons de contrôle
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        # Bouton STOP visible seulement si en cours
        if st.session_state.continuous_running:
            if st.button(
                "🛑 ARRÊTER L'OPTIMISATION", type="secondary", use_container_width=True
            ):
                st.session_state.continuous_running = False
                st.warning("⏹️ Arrêt demandé... L'itération en cours va se terminer.")
                st.rerun()

    with col_btn2:
        # Afficher statistiques si mode continu actif
        if (
            st.session_state.get("llm_continuous_mode", False)
            and st.session_state.continuous_iteration > 0
        ):
            st.metric(
                "📊 Stratégies gagnantes",
                st.session_state.winning_strategies_count,
                delta=f"Itération {st.session_state.continuous_iteration}",
            )

    # Bouton de lancement principal
    launch_disabled = st.session_state.continuous_running
    if st.button(
        (
            "🚀 Lancer l'optimisation Multi-LLM"
            if not st.session_state.get("llm_continuous_mode")
            else "♾️ Démarrer la boucle continue"
        ),
        type="primary",
        use_container_width=True,
        disabled=launch_disabled,
    ):
        # Activer le mode continu si sélectionné
        if st.session_state.get("llm_continuous_mode", False):
            st.session_state.continuous_running = True
            st.session_state.continuous_iteration = 0
            st.session_state.winning_strategies_count = 0

        run_multi_llm_optimization(
            strategy_name=strategy_name,
            sweep_params=sweep_params,
            analyst_model=analyst_model,
            strategist_model=strategist_model,
            model_router=router,
            n_proposals=n_proposals,
            top_n_analysis=top_n_analysis,
            use_gpu=use_gpu,
            use_multigpu=use_multigpu,
            max_workers=max_workers,
            feeder_aggr=feeder_aggr,
            force_processpool=force_processpool,
            memory_saver=memory_saver,
            enable_streaming=enable_streaming,
            max_iterations=max_iterations,
            enable_cross_testing=enable_cross_testing,
            cross_tokens=cross_tokens,
            cross_timeframes=cross_timeframes,
            continuous_mode=st.session_state.get("llm_continuous_mode", False),
            min_sharpe_threshold=(
                min_sharpe_threshold
                if st.session_state.get("llm_continuous_mode")
                else 0.5
            ),
            min_cross_sharpe=(
                min_cross_sharpe if st.session_state.get("llm_continuous_mode") else 0.2
            ),
        )

    # ============================================================
    # 📊 SECTION RÉSULTATS PERSISTANTS
    # Cette section affiche les résultats même après un rerun
    # ============================================================
    _render_persistent_results()


def _render_persistent_results():
    """Affiche les résultats d'optimisation stockés en session_state (persistant après rerun)."""

    llm_results = st.session_state.get("llm_results", {})

    # Ne rien afficher si pas de résultats
    if not llm_results:
        return

    st.divider()
    st.markdown("## 📊 Résultats de la dernière optimisation")

    # Bouton pour effacer les résultats
    col_clear, col_info = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Effacer résultats", key="clear_llm_results"):
            for key in ["llm_results", "llm_best_test", "llm_cross_results"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col_info:
        iteration = st.session_state.get("continuous_iteration", 0)
        winning = st.session_state.get("winning_strategies_count", 0)
        if iteration > 0:
            st.info(f"🔄 Itération {iteration} | 🏆 {winning} stratégie(s) gagnante(s)")

    # 1. Résultats du Sweep
    sweep_data = llm_results.get("sweep", [])
    if sweep_data:
        with st.expander("📈 Sweep GPU - Top 50 configurations", expanded=False):
            df_sweep = pd.DataFrame(sweep_data)
            # Normaliser colonnes si nécessaire
            if "sharpe" in df_sweep.columns and "sharpe_ratio" not in df_sweep.columns:
                df_sweep["sharpe_ratio"] = df_sweep["sharpe"]
            if "sharpe_ratio" in df_sweep.columns:
                top_50 = df_sweep.nlargest(50, "sharpe_ratio")
                st.dataframe(top_50, use_container_width=True, hide_index=True)
            else:
                st.dataframe(
                    df_sweep.head(50), use_container_width=True, hide_index=True
                )

    # 2. Analyse de l'Analyst
    analysis_data = llm_results.get("analysis", {})
    if analysis_data:
        with st.expander("🧠 Analyse Analyst", expanded=False):
            if isinstance(analysis_data, dict):
                # Patterns
                patterns = analysis_data.get("patterns", [])
                if patterns:
                    st.markdown("**Patterns identifiés:**")
                    for p in patterns[:5]:
                        st.markdown(f"- {p}")

                # Recommandations
                recs = analysis_data.get("recommendations", [])
                if recs:
                    st.markdown("**Recommandations:**")
                    for r in recs[:5]:
                        st.markdown(f"- {r}")

                # Métriques clés
                metrics = analysis_data.get("key_metrics", {})
                if metrics:
                    st.markdown("**Métriques clés:**")
                    st.json(metrics)
            else:
                st.write(analysis_data)

    # 3. Propositions du Strategist
    proposals_data = llm_results.get("proposals", {})
    if proposals_data:
        with st.expander("🎨 Propositions Strategist", expanded=False):
            proposals_list = (
                proposals_data.get("proposals", [])
                if isinstance(proposals_data, dict)
                else []
            )
            if proposals_list:
                for i, prop in enumerate(proposals_list, 1):
                    st.markdown(f"**Proposition {i}:**")
                    params = prop.get("params", prop.get("parameters", {}))
                    rationale = prop.get("rationale", prop.get("reasoning", ""))
                    if params:
                        st.json(params)
                    if rationale:
                        st.caption(f"💡 {rationale}")
                    st.divider()
            else:
                st.write(proposals_data)

    # 4. Résultats des tests
    test_results = llm_results.get("tests", [])
    if test_results:
        with st.expander("✅ Résultats des tests", expanded=True):
            for i, test in enumerate(test_results, 1):
                metrics = test.get("metrics", test)
                sharpe = metrics.get("sharpe", metrics.get("sharpe_ratio")) or 0
                ret = metrics.get("total_return", metrics.get("pnl_pct")) or 0
                trades = metrics.get("trades", metrics.get("num_trades")) or 0

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"Test {i} - Sharpe", f"{sharpe:.3f}")
                with col2:
                    st.metric(
                        "Return", f"{ret:.2%}" if abs(ret) < 10 else f"{ret:.1f}%"
                    )
                with col3:
                    st.metric("Trades", f"{trades}")
                with col4:
                    win_rate = metrics.get("win_rate") or 0
                    st.metric(
                        "Win Rate",
                        f"{win_rate:.1%}" if win_rate < 1 else f"{win_rate:.1f}%",
                    )

    # 5. Meilleur test
    best_test = st.session_state.get("llm_best_test", {})
    if best_test:
        with st.expander("🏆 Meilleure configuration", expanded=True):
            col_params, col_metrics = st.columns(2)
            with col_params:
                st.markdown("**Paramètres:**")
                params = best_test.get("params", best_test.get("parameters", {}))
                if params:
                    st.json(params)
            with col_metrics:
                st.markdown("**Métriques:**")
                metrics = best_test.get("metrics", {})
                if metrics:
                    st.metric("Sharpe", f"{(metrics.get('sharpe') or 0):.3f}")
                    st.metric("Return", f"{(metrics.get('total_return') or 0):.2%}")
                    st.metric("Trades", f"{metrics.get('trades') or 0}")

    # 6. Cross-testing
    cross_results = st.session_state.get("llm_cross_results", [])
    if cross_results:
        with st.expander("🌍 Résultats Cross-Testing", expanded=True):
            df_cross = pd.DataFrame(cross_results)

            # Calculer moyenne
            if "sharpe" in df_cross.columns:
                avg_sharpe = df_cross["sharpe"].mean()
                st.metric("Sharpe moyen cross-test", f"{avg_sharpe:.3f}")

            st.dataframe(df_cross, use_container_width=True, hide_index=True)

            # Heatmap si assez de données
            if (
                len(cross_results) >= 4
                and "token" in df_cross.columns
                and "timeframe" in df_cross.columns
            ):
                try:
                    pivot = df_cross.pivot_table(
                        values="sharpe",
                        index="token",
                        columns="timeframe",
                        aggfunc="mean",
                    )
                    import plotly.express as px

                    fig = px.imshow(
                        pivot,
                        text_auto=".2f",
                        color_continuous_scale="RdYlGn",
                        title="Sharpe par Token/Timeframe",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass


def check_prerequisites():
    """Vérifie que les prérequis sont installés."""

    st.markdown("**Vérification des prérequis:**")

    col1, col2 = st.columns(2)

    with col1:
        # Vérifier Ollama
        try:
            from threadx.llm.client import LLMClient

            client = LLMClient(model="gemma3:27b", timeout=5.0)
            test = client.complete("Test", max_tokens=5)
            st.success("✅ Ollama actif")
        except Exception as e:
            st.error(f"❌ Ollama non accessible: {e}")
            st.code("ollama serve", language="bash")

    with col2:
        # Vérifier GPU
        try:
            import cupy as cp

            # Tenter de déterminer si GPU disponible (compatible toutes versions CuPy)
            try:
                # Méthode 1: is_available() (versions récentes)
                gpu_available = cp.cuda.is_available()
            except AttributeError:
                # Méthode 2: device_count (versions plus anciennes)
                try:
                    gpu_available = cp.cuda.runtime.getDeviceCount() > 0
                except:
                    # Méthode 3: Tenter allocation mémoire
                    try:
                        _ = cp.array([1])
                        gpu_available = True
                    except:
                        gpu_available = False

            if gpu_available:
                try:
                    cuda_version = cp.cuda.runtime.runtimeGetVersion()
                    st.success(f"✅ GPU disponible (CUDA {cuda_version})")
                except:
                    st.success("✅ GPU disponible")
            else:
                st.warning("⚠️ GPU non disponible (CPU sera utilisé)")
        except ImportError:
            st.warning("⚠️ CuPy non installé (CPU sera utilisé)")
        except Exception as e:
            st.warning(f"⚠️ Erreur vérification GPU: {e} (CPU sera utilisé)")


def render_llm_thinking_stream(
    prompt: str,
    system: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """
    Affiche la réflexion du LLM en streaming temps réel.

    Capture et affiche les balises <think>...</think> de DeepSeek-R1
    pendant que le modèle génère sa réponse.

    Args:
        prompt: Prompt utilisateur
        system: Prompt système
        model: Nom du modèle (ex: "deepseek-r1:32b")
        temperature: Température de génération
        max_tokens: Nombre max de tokens

    Returns:
        str: Réponse complète du LLM
    """
    import re

    from threadx.llm.client import LLMClient

    with st.expander("🧠 Réflexion du LLM en temps réel", expanded=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            thinking_placeholder = st.empty()
            json_placeholder = st.empty()

        with col2:
            st.caption("🔄 Streaming actif...")
            progress = st.progress(0)

        full_response = ""
        client = LLMClient(model=model, debug=False)

        try:
            chunk_count = 0
            for chunk in client.complete_streaming(
                prompt, system, temperature, max_tokens
            ):
                full_response += chunk
                chunk_count += 1

                # Extraire <think>...</think>
                think_match = re.search(
                    r"<think>(.*?)</think>", full_response, re.DOTALL
                )
                if think_match:
                    thinking = think_match.group(1).strip()
                    thinking_placeholder.markdown(
                        f"**💭 Raisonnement Chain-of-Thought:**\n\n{thinking}"
                    )

                # Afficher JSON si détecté
                json_match = re.search(r"\{.*\}", full_response, re.DOTALL)
                if json_match:
                    json_placeholder.code(json_match.group(0), language="json")

                # Mise à jour progrès (approximatif basé sur chunks)
                progress.progress(min(chunk_count / 100, 0.99))

            progress.progress(1.0)
            st.caption("✅ Génération terminée")

        except Exception as e:
            st.error(f"❌ Erreur streaming: {e}")
            return ""

    return full_response


def run_multi_llm_optimization(
    strategy_name: str,
    sweep_params: dict,
    analyst_model: str,
    strategist_model: str,
    n_proposals: int,
    top_n_analysis: int,
    use_gpu: bool,
    use_multigpu: bool,
    max_workers: int | None,
    feeder_aggr: int,
    force_processpool: bool,
    model_router: ModelRouter = None,
    memory_saver: bool = True,
    enable_streaming: bool = True,
    max_iterations: int = 1,
    enable_cross_testing: bool = False,
    cross_tokens: list[str] | None = None,
    cross_timeframes: list[str] | None = None,
    continuous_mode: bool = False,
    min_sharpe_threshold: float = 0.5,
    min_cross_sharpe: float = 0.2,
):
    """Exécute le workflow complet d'optimisation Multi-LLM.

    Args:
        memory_saver: Si True, décharge les modèles Ollama de la mémoire après chaque agent
        enable_streaming: Si True, affiche la réflexion Chain-of-Thought en temps réel
        max_iterations: Nombre maximum d'itérations d'amélioration
        enable_cross_testing: Si True, valide sur plusieurs tokens/timeframes
        cross_tokens: Liste de tokens pour cross-testing
        cross_timeframes: Liste de timeframes pour cross-testing
        continuous_mode: Si True, boucle continue jusqu'à arrêt manuel
        min_sharpe_threshold: Seuil Sharpe pour sauvegarder comme stratégie gagnante
        min_cross_sharpe: Seuil Sharpe moyen en cross-testing
    """

    # Valeurs par défaut
    cross_tokens = cross_tokens or ["BTCUSDC"]
    cross_timeframes = cross_timeframes or ["1h"]

    # Initialiser best_params pour éviter UnboundLocalError
    best_params = None

    # Incrémenteur d'itération globale pour piloter la rotation des modèles
    st.session_state["llm_step"] = st.session_state.get("llm_step", 0) + 1
    current_step = st.session_state["llm_step"]

    # Initialiser session state
    if "llm_results" not in st.session_state:
        st.session_state.llm_results = {}

    # Tracker le temps de début du run pour le rapport
    st.session_state["llm_run_start_time"] = time.time()

    logger.info(
        f"[Multi-LLM] Démarrage optimisation - "
        f"strategy:{strategy_name}, analyst:{analyst_model}, strategist:{strategist_model}, "
        f"n_proposals:{n_proposals}, gpu:{use_gpu}, multigpu:{use_multigpu}"
    )

    # Conteneurs pour affichage progressif
    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # ============================================================
        # ÉTAPE 1: SWEEP GPU
        # ============================================================
        status_text.markdown("### 🔄 Étape 1/5: Exécution du Sweep GPU...")
        progress_bar.progress(10)

        sweep_start = time.time()
        with st.spinner(
            f"Test de {len(list(_generate_combinations(sweep_params, strategy_name)))} configurations..."
        ):
            sweep_results = execute_sweep(
                strategy_name=strategy_name,
                sweep_params=sweep_params,
                use_gpu=use_gpu,
                use_multigpu=use_multigpu,
                max_workers=max_workers,
                feeder_aggr=feeder_aggr,
                force_processpool=force_processpool,
            )
        st.session_state["llm_sweep_duration"] = time.time() - sweep_start

        logger.info(
            f"[Multi-LLM] Étape 1/5 SWEEP terminé - "
            f"{len(sweep_results)} configs testées en {time.time() - sweep_start:.2f}s"
        )

        st.session_state.llm_results["sweep"] = sweep_results

        # Afficher résultats sweep
        with results_container:
            st.success(f"✅ Sweep terminé: {len(sweep_results)} configs testées")

            with st.expander("📊 Top 50 configurations", expanded=True):
                df_sweep = pd.DataFrame(sweep_results)

                # Normaliser les noms de colonnes (les métriques devraient déjà être extraites par execute_sweep)
                # Mais vérification au cas où
                if (
                    "sharpe" in df_sweep.columns
                    and "sharpe_ratio" not in df_sweep.columns
                ):
                    df_sweep["sharpe_ratio"] = df_sweep["sharpe"]
                if (
                    "pnl_pct" in df_sweep.columns
                    and "total_return" not in df_sweep.columns
                ):
                    df_sweep["total_return"] = df_sweep["pnl_pct"]

                # Filtrer les configs invalides avant tout tri pour MA/EMA (slow > fast)
                def _is_valid_ma_row(row: pd.Series) -> bool:
                    params = row.get("params", {})
                    if not params and isinstance(row, pd.Series):
                        # fallback si les valeurs sont à plat dans le row
                        params = {
                            k: row.get(k)
                            for k in ["fast_period", "slow_period"]
                            if k in row
                        }
                    fast = params.get("fast_period")
                    slow = params.get("slow_period")
                    if fast is None or slow is None:
                        return True
                    return slow > fast

                if strategy_name in ["MA_Crossover", "EMA_Cross"]:
                    before = len(df_sweep)
                    df_sweep = df_sweep[df_sweep.apply(_is_valid_ma_row, axis=1)]
                    dropped = before - len(df_sweep)
                    if dropped > 0:
                        st.warning(
                            f"⚠️ {dropped} configs invalides retirées (slow_period ≤ fast_period)."
                        )
                        logger.warning(
                            f"[Sweep] {dropped} configs MA/EMA filtrées (slow<=fast)"
                        )

                # Vérifier que sharpe_ratio existe avant de trier
                if "sharpe_ratio" not in df_sweep.columns:
                    st.error(
                        f"❌ Colonne 'sharpe_ratio' manquante dans df_sweep. Colonnes: {list(df_sweep.columns)}"
                    )
                    if len(df_sweep) > 0:
                        st.caption("Premier résultat:")
                        st.json(df_sweep.iloc[0].to_dict())
                    raise Exception(
                        "Impossible de trier par sharpe_ratio - colonne manquante"
                    )

                top_50 = df_sweep.nlargest(50, "sharpe_ratio")
                st.dataframe(
                    top_50,
                    width="stretch",
                    hide_index=True,
                )

        progress_bar.progress(30)

        # ============================================================
        # ÉTAPE 2: ANALYSE ANALYST
        # ============================================================

        # Sélection du modèle (Router ou manuel)
        current_analyst_model = analyst_model
        if model_router:
            current_analyst_model = model_router.get_model_for_task(
                TaskType.INITIALIZATION, step_number=current_step
            )

        status_text.markdown(
            f"### 🧠 Étape 2/5: Analyse Analyst ({current_analyst_model})..."
        )

        logger.info(
            f"[Multi-LLM] Étape 2/5 ANALYST démarré - modèle:{current_analyst_model}, top_n:{top_n_analysis}"
        )

        with st.spinner(
            f"Analyse des top {top_n_analysis} configs (peut prendre 30-60s)..."
        ):
            analyst = Analyst(model=current_analyst_model, debug=False)

            # Créer zone de streaming pour l'analyse
            analysis_container = st.container()

            with analysis_container:
                st.markdown("#### 🔍 Réflexions de l'Analyst")

                with st.chat_message("assistant", avatar="🧠"):
                    st.caption(f"Modèle: {analyst_model}")

                    # Placeholder pour streaming
                    analysis_placeholder = st.empty()

                    # Exécuter analyse (avec ou sans streaming)
                    start_time = time.time()

                    if enable_streaming:
                        # Mode streaming: afficher réflexion en temps réel
                        analysis_placeholder.info(
                            "🧠 Analyse en cours avec streaming Chain-of-Thought..."
                        )

                        # Construire le prompt manuellement (même logique que dans Analyst)
                        prompt = f"""
Analysez les résultats de backtest suivants (top {top_n_analysis} configurations) :

{df_sweep.head(top_n_analysis).to_string()}

Identifiez:
1. **Patterns** : Relations entre paramètres et performances
2. **Key metrics** : Moyennes des métriques importantes (Sharpe, Drawdown, Win Rate, etc.)
3. **Trade-offs** : Compromis (ex: Sharpe élevé mais drawdown important)
4. **Recommendations** : Suggestions pour améliorer les performances

Répondez en JSON structuré:
{{
  "patterns": ["pattern1", "pattern2", ...],
  "key_metrics": {{"avg_sharpe": X, "max_drawdown_avg": Y, "avg_win_rate": Z, ...}},
  "trade_offs": ["trade-off1", ...],
  "recommendations": ["rec1", "rec2", ...]
}}
"""
                        system_prompt = """Vous êtes un Analyst quantitatif expert en analyse de résultats de backtest.
Votre rôle: analyser les données factuelles, identifier patterns et trade-offs, et fournir des recommandations claires."""

                        # Streaming avec affichage temps réel
                        full_response = render_llm_thinking_stream(
                            prompt=prompt,
                            system=system_prompt,
                            model=current_analyst_model,
                            temperature=0.3,
                            max_tokens=2000,
                        )

                        # Parser le JSON depuis la réponse
                        import json
                        import re

                        json_match = re.search(r"\{.*\}", full_response, re.DOTALL)
                        if json_match:
                            try:
                                analysis_data = json.loads(json_match.group(0))
                            except json.JSONDecodeError:
                                st.error(
                                    "❌ Erreur parsing JSON - utilisation fallback"
                                )
                                analysis_data = {
                                    "patterns": [],
                                    "key_metrics": {},
                                    "trade_offs": [],
                                    "recommendations": [
                                        "Analyser patterns identifiés",
                                        "Tester configurations avec Sharpe > moyenne",
                                    ],
                                }
                        else:
                            st.error("❌ Pas de JSON détecté - utilisation fallback")
                            analysis_data = {
                                "patterns": [],
                                "key_metrics": {},
                                "trade_offs": [],
                                "recommendations": ["Analyser patterns identifiés"],
                            }

                        # Construire résultat compatible
                        analysis_result = {
                            "analysis": analysis_data,
                            "top_configs": df_sweep.head(top_n_analysis).to_dict(
                                "records"
                            ),
                        }

                    else:
                        # Mode normal: appel direct sans streaming
                        analysis_placeholder.info("⏳ Analyse en cours...")
                        analysis_result = analyst.analyze_sweep_results(
                            sweep_df=df_sweep, top_n=top_n_analysis
                        )

                    elapsed = time.time() - start_time
                    st.session_state["llm_analyst_duration"] = elapsed

                    logger.info(
                        f"[Multi-LLM] Étape 2/5 ANALYST terminé - "
                        f"{len(analysis_result.get('analysis', {}).get('recommendations', []))} recommandations en {elapsed:.2f}s"
                    )

                    # Afficher résultats formatés
                    analysis_placeholder.empty()
                    display_analyst_results(analysis_result, elapsed)

        st.session_state.llm_results["analysis"] = analysis_result

        # Décharger le modèle Analyst si mode économie mémoire activé
        if memory_saver:
            unload_ollama_model(analyst_model)
            st.caption(f"💾 Modèle {analyst_model} déchargé de la mémoire")

        progress_bar.progress(50)

        # ============================================================
        # ÉTAPE 3: PROPOSITIONS STRATEGIST
        # ============================================================

        # Sélection du modèle (Router ou manuel)
        current_strategist_model = strategist_model
        if model_router:
            current_strategist_model = model_router.get_model_for_task(
                TaskType.OPTIMIZATION, step_number=current_step
            )

        status_text.markdown(
            f"### 🎨 Étape 3/5: Propositions Strategist ({current_strategist_model})..."
        )

        # ─────────────────────────────────────────────────────────────────────
        # FONCTION DE VALIDATION STRATÉGIE-SPÉCIFIQUE
        # ─────────────────────────────────────────────────────────────────────
        def validate_baseline_coherence(
            config: dict, strat_name: str
        ) -> tuple[bool, list[str]]:
            """
            Valide la cohérence d'une config selon la stratégie.

            Returns:
                (is_valid, warnings) : (True/False, liste de warnings)
            """
            warnings = []
            params = config.get("params", {})

            # Fallback : si 'params' vide, chercher à la racine
            if not params:
                param_specs = parameter_specs_for(strat_name)
                params = {k: v for k, v in config.items() if k in param_specs.keys()}

            # Règle 1: MA_Crossover/EMA_Cross → slow DOIT être > fast
            if strat_name in ["MA_Crossover", "EMA_Cross"]:
                fast = params.get("fast_period")
                slow = params.get("slow_period")
                if fast and slow and slow <= fast:
                    warnings.append(
                        f"❌ BLOQUANT: slow_period ({slow}) ≤ fast_period ({fast}) → Invalide"
                    )
                    return False, warnings  # Invalide

            # Règle 2: Take Profit DOIT être > Stop Loss (ratio min 1.5:1)
            sl = params.get("stop_loss_pct")
            tp = params.get("take_profit_pct")
            if sl and tp and tp < sl * 1.5:
                warnings.append(
                    f"⚠️ Ratio TP/SL faible: {tp}/{sl} = {tp/sl:.2f}x (recommandé >1.5x)"
                )
                # Non-bloquant mais signalé

            # Règle 3: Leverage > 1 avec Sharpe faible (<0.5)
            leverage = params.get("leverage", 1.0)
            sharpe = config.get("sharpe_ratio", 0)
            if leverage > 1.0 and sharpe < 0.5:
                sharpe_display = (
                    f"{sharpe:.3f}" if isinstance(sharpe, (int, float)) else "N/A"
                )
                warnings.append(
                    f"⚠️ Risque: Leverage {leverage}x + Sharpe faible ({sharpe_display})"
                )

            # Règle 4: Bollinger Breakout → bb_std > 0
            if strat_name == "Bollinger_Breakout":
                bb_std = params.get("bb_std_dev")
                if bb_std and bb_std <= 0:
                    warnings.append(
                        f"❌ BLOQUANT: bb_std_dev ({bb_std}) ≤ 0 → Invalide"
                    )
                    return False, warnings

            return True, warnings

        # ─────────────────────────────────────────────────────────────────────
        # SÉLECTION BASELINE AVEC VALIDATION (skip si incohérent)
        # ─────────────────────────────────────────────────────────────────────
        baseline_config = None
        baseline_rank = 0
        max_candidates = min(20, len(df_sweep))  # Vérifier jusqu'à top 20

        for rank in range(max_candidates):
            top_n = df_sweep.nlargest(rank + 1, "sharpe_ratio")
            if len(top_n) <= rank:
                break  # Pas assez de données
            candidate = top_n.iloc[rank]
            is_valid, validation_warnings = validate_baseline_coherence(
                candidate.to_dict(), strategy_name
            )

            if is_valid:
                baseline_config = candidate.to_dict()
                baseline_rank = rank + 1

                # Afficher réussite
                st.success(
                    f"✅ Baseline valide sélectionnée (rang #{baseline_rank}, Sharpe: {candidate['sharpe_ratio']:.3f})"
                )

                # Afficher warnings non-bloquants si présents
                if validation_warnings:
                    st.info(
                        "ℹ️ **Remarques (non-bloquantes):**\n\n"
                        + "\n".join(validation_warnings)
                    )

                break
            else:
                # Config invalide → skip
                st.warning(
                    f"⚠️ Config rang #{rank + 1} ignorée (Sharpe: {candidate['sharpe_ratio']:.3f}):\n"
                    + "\n".join(validation_warnings)
                )

        # Si aucune baseline valide trouvée dans top 20
        if baseline_config is None:
            # Fallback auto-fix pour MA_Crossover / EMA_Cross : corriger slow<=fast
            if strategy_name in ["MA_Crossover", "EMA_Cross"] and len(df_sweep) > 0:
                top_candidate = df_sweep.nlargest(1, "sharpe_ratio").iloc[0].to_dict()
                param_specs_full = parameter_specs_for(strategy_name)
                params = top_candidate.get("params", {})
                if not params:
                    params = {
                        k: v
                        for k, v in top_candidate.items()
                        if k in param_specs_full.keys()
                    }

                fast = params.get("fast_period")
                slow = params.get("slow_period")
                if fast is not None and slow is not None and slow <= fast:
                    params["slow_period"] = fast + 1  # Corrige minimalement le ratio

                    top_candidate["params"] = params
                    baseline_config = top_candidate
                    baseline_rank = 1

                    st.warning(
                        "⚠️ Aucune baseline valide trouvée. "
                        "Fallback appliqué : correction auto slow_period>fast_period "
                        "sur la meilleure config disponible (rang #1)."
                    )
                else:
                    st.error(
                        f"❌ **Aucune config valide trouvée dans le top {max_candidates}**\n\n"
                        "Vérifiez les paramètres du sweep ou élargissez les plages."
                    )
                    st.stop()
            else:
                st.error(
                    f"❌ **Aucune config valide trouvée dans le top {max_candidates}**\n\n"
                    "Vérifiez les paramètres du sweep ou élargissez les plages."
                )
                st.stop()

        # Debug : vérifier que les métriques sont présentes
        baseline_sharpe_val = baseline_config.get("sharpe_ratio")
        baseline_return_val = baseline_config.get("total_return")
        baseline_sharpe_display = (
            f"{baseline_sharpe_val:.3f}"
            if isinstance(baseline_sharpe_val, (int, float))
            else "N/A"
        )
        baseline_return_display = (
            f"{baseline_return_val:.3f}"
            if isinstance(baseline_return_val, (int, float))
            else "N/A"
        )
        st.caption(
            f"📊 Baseline sélectionnée - Sharpe: {baseline_sharpe_display}, Return: {baseline_return_display}"
        )

        # Extraire les paramètres depuis baseline_config['params']
        # SweepRunner retourne un format avec 'params' comme colonne séparée
        param_specs_full = parameter_specs_for(strategy_name)
        baseline_params = baseline_config.get("params", {})

        # Si 'params' n'existe pas, essayer à la racine (fallback)
        if not baseline_params:
            baseline_params = {
                k: v for k, v in baseline_config.items() if k in param_specs_full.keys()
            }

        # Debug : afficher ce qui a été extrait
        st.caption(
            f"📊 Paramètres baseline extraits : {len(baseline_params)} sur {len(param_specs_full)}"
        )
        with st.expander("🔍 Détails baseline complète", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Paramètres:**")
                st.json(baseline_params)
            with col2:
                st.markdown("**Métriques:**")
                metrics = {
                    k: v
                    for k, v in baseline_config.items()
                    if k
                    in [
                        "sharpe_ratio",
                        "total_return",
                        "max_drawdown",
                        "win_rate",
                        "pnl",
                        "pnl_pct",
                    ]
                }
                st.json(metrics)

        with st.spinner(
            f"Génération de {n_proposals} propositions créatives (peut prendre 20-40s)..."
        ):
            strategist = Strategist(model=current_strategist_model, debug=False)

            # Créer zone de streaming pour propositions
            proposals_container = st.container()

            with proposals_container:
                st.markdown("#### 💡 Propositions du Strategist")

                with st.chat_message("assistant", avatar="🎨"):
                    st.caption(f"Modèle: {current_strategist_model}")

                    proposals_placeholder = st.empty()
                    proposals_placeholder.info("⏳ Génération créative en cours...")

                    # Générer propositions
                    start_time = time.time()

                    # Construire param_specs pour validation
                    param_specs_full = parameter_specs_for(strategy_name)

                    logger.info(
                        f"[Multi-LLM] Étape 3/5 STRATEGIST démarré - modèle:{current_strategist_model}, n_proposals:{n_proposals}"
                    )

                    proposals_result = strategist.propose_modifications(
                        analysis=analysis_result,
                        current_params=baseline_params,
                        param_specs=param_specs_full,
                        n_proposals=n_proposals,
                    )
                    elapsed = time.time() - start_time
                    st.session_state["llm_strategist_duration"] = elapsed

                    logger.info(
                        f"[Multi-LLM] Étape 3/5 STRATEGIST terminé - "
                        f"{len(proposals_result.get('proposals', []))} propositions en {elapsed:.2f}s"
                    )

                    # Afficher propositions formatées
                    proposals_placeholder.empty()
                    display_strategist_results(
                        proposals_result,
                        baseline_params,
                        baseline_config.get("sharpe_ratio", 0),
                        elapsed,
                    )

        st.session_state.llm_results["proposals"] = proposals_result

        # Décharger le modèle Strategist si mode économie mémoire activé
        if memory_saver:
            unload_ollama_model(strategist_model)
            st.caption(f"💾 Modèle {strategist_model} déchargé de la mémoire")

        progress_bar.progress(70)

        # ============================================================
        # ÉTAPE 4: TESTS AUTOMATIQUES
        # ============================================================
        status_text.markdown("### ✅ Étape 4/5: Tests automatiques des propositions...")

        logger.info(
            f"[Multi-LLM] Étape 4/5 TESTS démarré - {len(proposals_result['proposals'])} propositions à tester"
        )
        test_start = time.time()

        with st.spinner(
            f"Test de {len(proposals_result['proposals'])} propositions..."
        ):
            test_results = test_proposals(
                strategy_name=strategy_name,
                proposals=proposals_result["proposals"],
                baseline_config=baseline_config,
                use_gpu=use_gpu,
            )

        test_duration = time.time() - test_start
        successful_tests = sum(1 for r in test_results if r.get("sharpe_ratio"))
        logger.info(
            f"[Multi-LLM] Étape 4/5 TESTS terminé - "
            f"{successful_tests}/{len(proposals_result['proposals'])} propositions valides en {test_duration:.2f}s"
        )

        st.session_state.llm_results["tests"] = test_results

        # Vérifier si les tests ont réussi
        if len(test_results) == 0:
            st.warning(
                "⚠️ Aucune proposition n'a pu être testée avec succès. Vérifiez les erreurs ci-dessus."
            )
            st.info(
                "💡 Les propositions du Strategist nécessitent peut-être des paramètres manquants ou incompatibles avec la stratégie."
            )

        progress_bar.progress(90)

        # ============================================================
        # ÉTAPE 5: VISUALISATION & RAPPORT
        # ============================================================
        status_text.markdown("### 📊 Étape 5/5: Génération du rapport final...")

        with st.spinner("Création des visualisations..."):
            display_final_report(
                baseline=baseline_config,
                proposals=proposals_result["proposals"],
                test_results=test_results,
                analysis=analysis_result,
            )

        # ============================================================
        # GÉNÉRATION DU RAPPORT ET INDEXATION
        # ============================================================
        status_text.markdown("### 📁 Sauvegarde du rapport...")

        try:
            # Calculer durée totale du run
            run_end_time = time.time()
            total_run_duration = run_end_time - st.session_state.get(
                "llm_run_start_time", run_end_time
            )

            # Récupérer les durées stockées dans session_state
            sweep_duration = st.session_state.get("llm_sweep_duration", 0.0)
            analyst_duration = st.session_state.get("llm_analyst_duration", 0.0)
            strategist_duration = st.session_state.get("llm_strategist_duration", 0.0)

            # Baseline sharpe
            baseline_sharpe = baseline_config.get(
                "sharpe_ratio", baseline_config.get("sharpe", 0.0)
            )

            # Créer le rapport
            report = create_report_from_run(
                strategy_name=strategy_name,
                sweep_results=sweep_results,
                sweep_params=sweep_params,
                sweep_duration=sweep_duration,
                analysis_result=analysis_result,
                analyst_model=analyst_model,
                analyst_duration=analyst_duration,
                proposals_result=proposals_result,
                baseline_params=baseline_params,
                baseline_sharpe=baseline_sharpe,
                strategist_model=strategist_model,
                strategist_duration=strategist_duration,
                test_results=test_results,
                config={
                    "use_gpu": use_gpu,
                    "use_multigpu": use_multigpu,
                    "max_workers": max_workers,
                    "feeder_aggr": feeder_aggr,
                    "n_proposals": n_proposals,
                    "top_n_analysis": top_n_analysis,
                    "memory_saver": memory_saver,
                },
            )

            # Sauvegarder et indexer
            index = RunIndex()
            report_path = index.save_report(
                report, tags=[strategy_name, f"sharpe_{report.best_sharpe:.2f}"]
            )

            # Stocker dans session_state pour accès ultérieur
            st.session_state["last_report"] = report
            st.session_state["last_report_path"] = str(report_path)

            # Afficher succès avec lien
            with results_container:
                st.divider()
                st.success(f"📁 **Rapport sauvegardé:** `{report_path}`")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🆔 Run ID", report.run_id)
                with col2:
                    st.metric("📊 Best Sharpe", f"{report.best_sharpe:.3f}")
                with col3:
                    improvement = (
                        "✅ Oui"
                        if (report.tests and report.tests.improvement_found)
                        else "❌ Non"
                    )
                    st.metric("📈 Amélioration", improvement)

                # Bouton téléchargement JSON
                st.download_button(
                    label="📥 Télécharger le rapport (JSON)",
                    data=report.to_json(),
                    file_name=f"llm_report_{report.run_id}.json",
                    mime="application/json",
                )

                # Afficher résumé
                with st.expander("📋 Résumé du rapport", expanded=False):
                    st.markdown(report.summary)

        except Exception as e:
            st.warning(f"⚠️ Erreur lors de la sauvegarde du rapport: {e}")
            import traceback

            st.caption(traceback.format_exc())

        # ============================================================
        # ÉTAPE 6 (OPTIONNEL): CROSS-TESTING
        # ============================================================
        if enable_cross_testing and len(cross_tokens) > 1 or len(cross_timeframes) > 1:
            status_text.markdown(
                "### 🌍 Étape 6: Cross-testing multi-tokens/timeframes..."
            )

            # Récupérer la meilleure config des tests
            best_test = None
            best_sharpe = float("-inf")
            for r in test_results:
                sharpe = r.get("sharpe_ratio", 0) or 0
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_test = r

            if best_test:
                best_params = best_test.get("params", baseline_params)

                # Sauvegarder best_test pour la sauvegarde des stratégies gagnantes
                st.session_state["llm_best_test"] = best_test

                st.info(
                    f"🔬 Test de la meilleure config (Sharpe={best_sharpe:.3f}) sur {len(cross_tokens)} tokens × {len(cross_timeframes)} timeframes"
                )

                cross_results = []
                cross_progress = st.progress(0)
                total_tests = len(cross_tokens) * len(cross_timeframes)
                test_idx = 0

                for token in cross_tokens:
                    for tf in cross_timeframes:
                        test_idx += 1
                        cross_progress.progress(test_idx / total_tests)

                        try:
                            # Charger les données pour ce token/timeframe
                            from threadx.data_access import load_ohlcv

                            df_cross = load_ohlcv(
                                symbol=token,
                                timeframe=tf,
                            )

                            if df_cross is not None and len(df_cross) > 0:
                                # Backtest avec les meilleurs paramètres
                                from threadx.ui.backtest_bridge import run_backtest_gpu

                                result = run_backtest_gpu(
                                    df=df_cross,
                                    strategy=strategy_name,
                                    params=best_params,
                                )

                                cross_results.append(
                                    {
                                        "token": token,
                                        "timeframe": tf,
                                        "sharpe_ratio": result.metrics.get(
                                            "sharpe_ratio"
                                        )
                                        or 0,
                                        "total_return": result.metrics.get(
                                            "total_return"
                                        )
                                        or 0,
                                        "max_drawdown": result.metrics.get(
                                            "max_drawdown"
                                        )
                                        or 0,
                                        "win_rate": result.metrics.get("win_rate") or 0,
                                        "n_trades": result.metrics.get("n_trades") or 0,
                                        "status": "✅",
                                    }
                                )
                                sharpe_val = result.metrics.get("sharpe_ratio") or 0
                                logger.info(
                                    f"[Cross-test] {token}/{tf}: Sharpe={sharpe_val:.3f}"
                                )
                            else:
                                cross_results.append(
                                    {
                                        "token": token,
                                        "timeframe": tf,
                                        "sharpe_ratio": None,
                                        "status": "❌ Pas de données",
                                    }
                                )
                        except Exception as e:
                            cross_results.append(
                                {
                                    "token": token,
                                    "timeframe": tf,
                                    "sharpe_ratio": None,
                                    "status": f"❌ {str(e)[:30]}",
                                }
                            )
                            logger.warning(f"[Cross-test] {token}/{tf}: Erreur - {e}")

                # Afficher les résultats du cross-testing
                if cross_results:
                    st.markdown("#### 🌍 Résultats Cross-Testing")

                    df_cross_results = pd.DataFrame(cross_results)

                    # Créer un tableau pivot
                    if "sharpe_ratio" in df_cross_results.columns:
                        pivot = df_cross_results.pivot_table(
                            index="token",
                            columns="timeframe",
                            values="sharpe_ratio",
                            aggfunc="first",
                        )
                        st.dataframe(
                            pivot.style.format("{:.3f}").background_gradient(
                                cmap="RdYlGn", axis=None
                            ),
                            use_container_width=True,
                        )

                        # Statistiques
                        valid_sharpes = [
                            r["sharpe_ratio"]
                            for r in cross_results
                            if r.get("sharpe_ratio") is not None
                        ]
                        if valid_sharpes:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "📊 Sharpe Moyen",
                                    f"{sum(valid_sharpes)/len(valid_sharpes):.3f}",
                                )
                            with col2:
                                st.metric("📈 Sharpe Max", f"{max(valid_sharpes):.3f}")
                            with col3:
                                st.metric("📉 Sharpe Min", f"{min(valid_sharpes):.3f}")

                            # Verdict
                            avg_sharpe = sum(valid_sharpes) / len(valid_sharpes)
                            if avg_sharpe > 0.3:
                                st.success(
                                    "✅ **La stratégie est robuste** - Performance positive sur plusieurs tokens/timeframes"
                                )
                            elif avg_sharpe > 0:
                                st.warning(
                                    "⚠️ **Performance mitigée** - La stratégie fonctionne mais avec des résultats variables"
                                )
                            else:
                                st.error(
                                    "❌ **Attention overfitting possible** - La stratégie ne généralise pas bien"
                                )

                    st.session_state["llm_cross_results"] = cross_results

        # ========================================
        # 🏆 SAUVEGARDE AUTOMATIQUE STRATÉGIE GAGNANTE
        # ========================================
        logger.info(
            f"[Continuous] Mode continu={continuous_mode}, best_params={'OK' if best_params else 'None'}"
        )

        if continuous_mode and best_params is not None:
            try:
                logger.info(
                    "[Continuous] Évaluation de la stratégie pour sauvegarde..."
                )
                # Récupérer les métriques du meilleur résultat depuis best_test
                best_metrics = {}
                best_test_saved = st.session_state.get("llm_best_test", {})

                if best_test_saved:
                    # Utiliser les métriques du meilleur test
                    best_metrics = {
                        "sharpe": best_test_saved.get("sharpe_ratio", 0),
                        "total_return": best_test_saved.get("total_return", 0),
                        "win_rate": best_test_saved.get("win_rate", 0),
                        "max_drawdown": best_test_saved.get("max_drawdown", 0),
                        "profit_factor": best_test_saved.get("profit_factor", 1.0),
                        "num_trades": len(best_test_saved.get("trades", [])),
                    }
                    logger.info(
                        f"[Continuous] Métriques extraites de best_test: sharpe={best_metrics['sharpe']:.3f}"
                    )
                else:
                    # Fallback: chercher dans llm_results
                    sweep_data = st.session_state.get("llm_results", {}).get(
                        "sweep", []
                    )
                    if sweep_data:
                        best_result = sweep_data[0] if sweep_data else {}
                        best_metrics = {
                            "sharpe": best_result.get("sharpe", 0),
                            "total_return": best_result.get("total_return", 0),
                            "win_rate": best_result.get("win_rate", 0),
                            "max_drawdown": best_result.get("max_dd", 0),
                            "profit_factor": best_result.get("profit_factor", 1.0),
                            "num_trades": best_result.get("trades", 0),
                        }
                        logger.info(
                            f"[Continuous] Métriques extraites de sweep: sharpe={best_metrics['sharpe']:.3f}"
                        )
                    else:
                        logger.warning("[Continuous] Aucune métrique trouvée!")

                # Définir les critères personnalisés
                criteria = WinningCriteria(
                    min_sharpe=min_sharpe_threshold,
                    min_avg_cross_sharpe=min_cross_sharpe,
                    min_positive_cross_ratio=0.5,
                    min_trades=3,
                    max_drawdown_pct=-30.0,
                    min_profit_factor=1.0,
                )

                # Récupérer résultats cross-test
                cross_test_data = st.session_state.get("llm_cross_results", [])

                # Tenter de sauvegarder si gagnante
                manager = get_winning_manager()
                # Mettre à jour les critères personnalisés du manager
                manager.criteria = criteria

                # Calculer avg_cross_sharpe
                avg_cross_sharpe = 0.0
                if cross_test_data:
                    cross_sharpes = [
                        (r.get("sharpe") or r.get("sharpe_ratio") or 0)
                        for r in cross_test_data
                        if isinstance(r, dict)
                    ]
                    # Filtrer les None et calculer la moyenne
                    valid_sharpes = [s for s in cross_sharpes if s is not None]
                    if valid_sharpes:
                        avg_cross_sharpe = sum(valid_sharpes) / len(valid_sharpes)

                # Vérifier si c'est une stratégie gagnante
                logger.info(
                    f"[Continuous] Critères: sharpe>={criteria.min_sharpe}, "
                    f"cross_sharpe>={criteria.min_avg_cross_sharpe}, "
                    f"trades>={criteria.min_trades}"
                )
                logger.info(
                    f"[Continuous] Métriques: sharpe={best_metrics.get('sharpe', 0):.3f}, "
                    f"avg_cross={avg_cross_sharpe:.3f}, trades={best_metrics.get('num_trades', 0)}"
                )

                is_winning_result, winning_reason = manager.is_winning(
                    sharpe_ratio=best_metrics.get("sharpe", 0),
                    cross_results=cross_test_data,
                    total_trades=best_metrics.get("num_trades", 0),
                    max_drawdown=best_metrics.get("max_drawdown", 0),
                    profit_factor=best_metrics.get("profit_factor", 1.0),
                )

                logger.info(
                    f"[Continuous] Résultat: winning={is_winning_result}, raison={winning_reason}"
                )

                if is_winning_result:
                    # Récupérer contexte depuis session_state
                    current_token = st.session_state.get("symbol", "UNKNOWN")
                    current_timeframe = st.session_state.get("timeframe", "1h")

                    # Ajouter sharpe_ratio aux metrics pour la sauvegarde
                    save_metrics = best_metrics.copy()
                    save_metrics["sharpe_ratio"] = save_metrics.get("sharpe", 0)

                    # Sauvegarder la stratégie gagnante
                    saved_path = save_winning_strategy(
                        strategy_name=strategy_name,
                        params=best_params,
                        metrics=save_metrics,
                        primary_token=current_token,
                        primary_timeframe=current_timeframe,
                        cross_results=cross_test_data,
                        models_used={
                            "analyst": analyst_model,
                            "strategist": strategist_model,
                        },
                        notes=f"Continuous mode - Iteration {st.session_state.get('continuous_iteration', 1)}",
                        tags=["continuous_mode", current_token, current_timeframe],
                    )

                    if saved_path:
                        st.session_state["winning_strategies_count"] = (
                            st.session_state.get("winning_strategies_count", 0) + 1
                        )
                        st.success(
                            f"🏆 **STRATÉGIE GAGNANTE SAUVEGARDÉE !**\n"
                            f"- Sharpe: {best_metrics.get('sharpe', 0):.3f}\n"
                            f"- Cross-test Sharpe moyen: {avg_cross_sharpe:.3f}\n"
                            f"- Fichier: `{saved_path}`"
                        )
                        logger.info(
                            f"[Continuous] Stratégie gagnante #{st.session_state['winning_strategies_count']} sauvegardée: {saved_path}"
                        )
                else:
                    st.info(
                        f"📊 Stratégie non retenue: {winning_reason}\n"
                        f"(Sharpe: {best_metrics.get('sharpe', 0):.3f}, Cross-test avg: {avg_cross_sharpe:.3f})"
                    )

            except Exception as e:
                logger.warning(f"[Continuous] Erreur sauvegarde stratégie: {e}")
                st.warning(f"⚠️ Impossible de sauvegarder la stratégie: {e}")

        progress_bar.progress(100)
        status_text.success("### 🎉 Optimisation Multi-LLM terminée !")

        # ========================================
        # 🔄 MODE CONTINU - Relancer si actif
        # ========================================
        logger.info(
            f"[Continuous] Check relance: continuous_mode={continuous_mode}, "
            f"continuous_running={st.session_state.get('continuous_running', False)}"
        )

        if continuous_mode and st.session_state.get("continuous_running", False):
            iteration = st.session_state.get("continuous_iteration", 1)
            winning_count = st.session_state.get("winning_strategies_count", 0)

            st.info(
                f"🔄 **Mode Continu** - Itération {iteration} terminée | "
                f"🏆 {winning_count} stratégie(s) gagnante(s) trouvée(s)"
            )

            # Incrémenter le compteur d'itération
            st.session_state["continuous_iteration"] = iteration + 1

            # Petite pause avant relance
            time.sleep(2)

            # Relancer automatiquement
            st.rerun()

        total_duration = time.time() - st.session_state["llm_run_start_time"]
        logger.info(
            f"[Multi-LLM] Optimisation TERMINÉE - "
            f"Durée totale:{total_duration:.2f}s, "
            f"Sweep:{st.session_state.get('llm_sweep_duration', 0):.1f}s, "
            f"Analyst:{st.session_state.get('llm_analyst_duration', 0):.1f}s, "
            f"Strategist:{st.session_state.get('llm_strategist_duration', 0):.1f}s"
        )

    except RuntimeError as e:
        error_msg = str(e)

        # Détection erreur CUDA Ollama
        if "CUDA" in error_msg or "GPU" in error_msg:
            st.error("### ❌ Erreur GPU Ollama détectée")
            st.warning(
                "**Le processus Ollama a rencontré une erreur GPU (CUDA).**\n\n"
                "**Solutions recommandées:**\n"
                "1. 🔄 **Redémarrez Ollama**: Ouvrez un terminal et tapez `ollama serve`\n"
                "2. 🔍 **Vérifiez votre GPU**: Assurez-vous qu'il n'est pas utilisé par une autre application\n"
                "3. 📦 **Utilisez un modèle plus petit**: Essayez `deepseek-r1:1.5b` ou `gemma2:2b`\n"
                "4. 🚫 **Fermez autres applications GPU**: Libérez la mémoire GPU\n"
                "5. 💻 **Mode CPU**: Configurez Ollama pour utiliser le CPU si le GPU est instable"
            )
            with st.expander("🔍 Détails techniques de l'erreur"):
                st.code(error_msg)
        else:
            st.error(f"❌ Erreur d'exécution: {error_msg}")

    except Exception as e:
        logger.error(
            f"[Multi-LLM] ERREUR lors de l'optimisation: {type(e).__name__}: {str(e)}",
            exc_info=True,
        )
        st.error(f"❌ Erreur inattendue: {e}")
        import traceback

        with st.expander("🐛 Traceback complet"):
            st.code(traceback.format_exc())


# ===================================================================
# PATCH CRITIQUE : Conversion forcée des paramètres en entiers
# Évite ValueError: min_periods must be an integer dans pandas.rolling
# ===================================================================
INTEGER_PARAM_KEYS = {
    "fast_period",
    "slow_period",
    "rsi_period",
    "bb_period",
    "atr_period",
    "ema_period",
    "macd_fast",
    "macd_slow",
    "macd_signal",
    "max_hold_bars",
    "lookback_bars",
    "trend_period",
}

RISK_PER_TRADE_MIN = 0.005
RISK_PER_TRADE_MAX = 0.02


def _force_integer_params(combo: dict) -> dict:
    """Convertit les paramètres censés être entiers en int (évite float 5.0 → crash pandas)."""
    combo = combo.copy()
    for key in INTEGER_PARAM_KEYS:
        if key in combo and combo[key] is not None:
            try:
                combo[key] = int(round(float(combo[key])))
            except (ValueError, TypeError):
                # Fallback sécurisé si conversion impossible
                combo[key] = 20
    return combo


def _apply_strategy_constraints(
    params: dict, strategy_name: str, context: str = ""
) -> dict:
    """Applique les contraintes de stratégie (slow>fast, bornes risk_per_trade) et logge les ajustements."""
    adjusted = params.copy()
    notices: list[str] = []

    # MA/EMA: slow_period doit être > fast_period
    fast = adjusted.get("fast_period")
    slow = adjusted.get("slow_period")
    if fast is not None and slow is not None:
        try:
            if float(slow) <= float(fast):
                adjusted["slow_period"] = int(float(fast)) + 1
                notices.append(
                    f"slow_period ajusté à {adjusted['slow_period']} (> fast_period {fast})"
                )
        except Exception:
            pass

    # risk_per_trade: clamp sur [0.5%, 2%]
    if "risk_per_trade" in adjusted and adjusted["risk_per_trade"] is not None:
        try:
            r = float(adjusted["risk_per_trade"])
            clamped = max(RISK_PER_TRADE_MIN, min(RISK_PER_TRADE_MAX, r))
            if clamped != r:
                adjusted["risk_per_trade"] = clamped
                notices.append(
                    f"risk_per_trade bridé à {clamped:.3f} (bornes {RISK_PER_TRADE_MIN:.3f}-{RISK_PER_TRADE_MAX:.3f})"
                )
        except Exception:
            pass

    if notices:
        prefix = f"[{context}] " if context else ""
        st.caption(f"⚠️ Contraintes appliquées {prefix}: " + " | ".join(notices))
        logger.warning(f"[Constraints] {prefix}{'; '.join(notices)}")

    return adjusted


def _generate_combinations(sweep_params: dict, strategy_name: str | None = None):
    """Génère toutes les combinaisons de paramètres (avec contraintes basiques si MA/EMA)."""
    from itertools import product

    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    for combo in product(*param_values):
        combo_dict = dict(zip(param_names, combo))
        # CORRECTION: Forcer conversion int pour paramètres de période
        combo_dict = _force_integer_params(combo_dict)

        if strategy_name in {"MA_Crossover", "EMA_Cross"}:
            fast = combo_dict.get("fast_period")
            slow = combo_dict.get("slow_period")
            if fast is not None and slow is not None and slow <= fast:
                # Skip invalid MA/EMA combos
                continue

        yield combo_dict


def execute_sweep(
    strategy_name: str,
    sweep_params: dict,
    use_gpu: bool,
    use_multigpu: bool,
    max_workers: int | None,
    feeder_aggr: int,
    force_processpool: bool,
):
    """Exécute le sweep et retourne les résultats avec SweepRunner performant."""
    import threading

    logger.info(
        f"[Sweep] Démarrage sweep - strategy:{strategy_name}, gpu:{use_gpu}, workers:{max_workers}"
    )

    # Configuration optimisation pour LLM: appliquer feeder_aggr
    os.environ["THREADX_FEEDER_AGGR"] = str(feeder_aggr)

    # Charger les vraies données depuis session_state (chargées dans la première page)
    from threadx.data_access import load_ohlcv

    symbol = st.session_state.get("symbol", "BTC")
    timeframe = st.session_state.get("timeframe", "1h")
    start_date = st.session_state.get("start_date")
    end_date = st.session_state.get("end_date")

    # Vérifier si les données sont déjà en cache session_state
    if "data" in st.session_state and not st.session_state.data.empty:
        df_market = st.session_state.data
        st.caption(
            f"📊 Utilisation des données en cache: {len(df_market)} barres ({symbol}/{timeframe})"
        )
    else:
        # Charger depuis la base de données
        try:
            df_market = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)
            if df_market.empty:
                st.error(
                    f"❌ Aucune donnée disponible pour {symbol}/{timeframe} entre {start_date} et {end_date}"
                )
                st.warning(
                    "💡 Allez d'abord dans l'onglet 'Configuration' pour charger les données."
                )
                return []
            st.session_state.data = df_market  # Mettre en cache
            st.caption(
                f"📊 Données chargées: {len(df_market)} barres ({df_market.index[0].date()} → {df_market.index[-1].date()})"
            )
        except Exception as e:
            st.error(f"❌ Erreur chargement données: {e}")
            st.warning(
                "💡 Allez d'abord dans l'onglet 'Configuration' pour charger les données."
            )
            return []

    # 🔥 NOUVEAU: Utiliser SweepRunner au lieu de la boucle séquentielle
    # Créer IndicatorBank pour cache GPU partagé
    indicator_settings = IndicatorSettings(use_gpu=use_gpu)
    indicator_bank = IndicatorBank(indicator_settings)

    # Initialiser SweepRunner avec paramètres configurables
    runner = SweepRunner(
        indicator_bank=indicator_bank,
        max_workers=max_workers if max_workers is not None else 30,
        use_multigpu=use_multigpu,
        use_processes=force_processpool,
    )

    # Convertir sweep_params au format ScenarioSpec
    scenario_params = {}
    for param_name, param_values in sweep_params.items():
        scenario_params[param_name] = {"values": param_values}

    scenario_spec = ScenarioSpec(type="grid", params=scenario_params)

    # Calculer nombre total de combinaisons (avec contraintes appliquées par generate_param_grid)
    total_combinations = len(
        generate_param_grid({k: v for k, v in scenario_params.items()})
    )

    # 🔥 Monitoring temps réel avec threading
    progress_placeholder = st.empty()
    stats_cols = st.columns(4)

    shared_state = {
        "running": False,
        "error": None,
        "results": None,
    }

    # 🔧 AUTO-FIX: Préparer Ollama pour le run (démarrage + nettoyage cache)
    st.info("🔧 Préparation de l'environnement LLM...")
    ollama_ready, ollama_msg = prepare_for_llm_run()
    if ollama_ready:
        st.success(ollama_msg)
    else:
        st.error(f"❌ Impossible de préparer Ollama: {ollama_msg}")
        st.stop()

    def run_sweep_thread():
        """Thread qui exécute le sweep."""
        try:
            shared_state["running"] = True
            results_df = runner.run_grid(
                grid_spec=scenario_spec,
                real_data=df_market,
                symbol=symbol,  # Utiliser la variable au lieu de "BTC" hardcodé
                timeframe=timeframe,  # Utiliser la variable au lieu de "1h" hardcodé
                strategy_name=strategy_name,
                reuse_cache=True,
            )
            shared_state["results"] = results_df
            shared_state["error"] = None
        except Exception as e:
            shared_state["error"] = str(e)
            shared_state["results"] = None
        finally:
            shared_state["running"] = False

    # Lancer le sweep en thread
    if "sweep_speed_samples" not in st.session_state:
        st.session_state["sweep_speed_samples"] = []

    sweep_thread = threading.Thread(target=run_sweep_thread, daemon=True)
    sweep_thread.start()

    # Monitoring UI
    start_time = time.time()
    status_placeholder = stats_cols[0].empty()
    speed_placeholder = stats_cols[1].empty()
    eta_placeholder = stats_cols[2].empty()
    completed_placeholder = stats_cols[3].empty()

    last_current = -1
    last_ui_update = 0.0
    progress_placeholder.progress(0, text="🚀 Initialisation du Sweep GPU...")
    status_placeholder.metric("📊 Statut", "Initialisation...", delta=None)

    # Boucle de monitoring
    while shared_state["running"]:
        try:
            if runner.total_scenarios > 0:
                current = runner.current_scenario
                total = runner.total_scenarios
                progress = min(current / total, 0.99)
                elapsed = time.time() - start_time

                now = time.time()
                if (
                    current > 0
                    and elapsed > 0
                    and (current != last_current or (now - last_ui_update) >= 0.2)
                ):
                    # Lissage vitesse sur 2 secondes
                    samples = st.session_state.get("sweep_speed_samples", [])
                    samples.append((now, current))

                    cutoff = now - 2.0
                    samples = [(t, c) for (t, c) in samples if t >= cutoff]
                    st.session_state["sweep_speed_samples"] = samples

                    if len(samples) >= 2:
                        t0, c0 = samples[0]
                        t1, c1 = samples[-1]
                        time_delta = max(1e-6, t1 - t0)
                        speed = max(0.0, (c1 - c0) / time_delta)
                    else:
                        delta_c = (
                            (current - last_current) if last_current >= 0 else current
                        )
                        delta_t = (
                            elapsed if last_current < 0 else (now - last_ui_update)
                        )
                        speed = max(0.0, delta_c / max(1e-6, delta_t))

                    remaining = total - current
                    eta_seconds = remaining / speed if speed > 0 else 0
                    eta_hours, eta_remainder = divmod(eta_seconds, 3600)
                    eta_minutes, eta_secs = divmod(eta_remainder, 60)

                    if eta_hours >= 1:
                        eta_str = f"{int(eta_hours)}h {int(eta_minutes)}m"
                    else:
                        eta_str = f"{int(eta_minutes)}m {int(eta_secs)}s"

                    last_ui_update = now
                    last_current = current

                    # Mise à jour UI
                    progress_placeholder.progress(
                        progress,
                        text=f"⏳ {current:,}/{total:,} configs ({progress*100:.1f}%)",
                    )
                    status_placeholder.metric(
                        "📊 Statut",
                        "En cours ⚡",
                        delta=f"{speed:.1f} tests/sec (moy. 2s)",
                        delta_color="normal",
                    )
                    speed_placeholder.metric(
                        "🚀 Vitesse",
                        f"{speed:.1f}",
                        delta="tests/sec",
                        delta_color="off",
                    )
                    eta_placeholder.metric("⏱️ ETA", eta_str)
                    completed_placeholder.metric(
                        "✅ Complétés",
                        f"{current:,}",
                        delta=f"{(current/total*100):.1f}%",
                        delta_color="normal",
                    )

            time.sleep(0.2)
        except Exception:
            pass

    # Attendre fin du thread
    sweep_thread.join(timeout=5)

    # Résultats finaux
    elapsed_time = time.time() - start_time

    if shared_state["error"]:
        progress_placeholder.progress(0, text=f"❌ Erreur après {elapsed_time:.1f}s")
        status_placeholder.metric("📊 Statut", "Erreur ❌", delta=None)
        raise Exception(shared_state["error"])

    results_df = shared_state.get("results")
    if results_df is None:
        raise Exception("Aucun résultat retourné")

    if len(results_df) == 0:
        raise Exception(
            "Le sweep n'a produit aucun résultat valide. Vérifiez les paramètres et les données."
        )

    completed = len(results_df)
    tests_per_second = completed / elapsed_time if elapsed_time > 0 else 0
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes)}m {int(seconds)}s"

    # Stats finales
    progress_placeholder.progress(
        1.0, text=f"✅ Sweep terminé en {time_str} | {completed:,} résultats"
    )
    status_placeholder.metric(
        "📊 Statut", "✅ Terminé", delta="100%", delta_color="normal"
    )
    speed_placeholder.metric(
        "🚀 Vitesse Moyenne",
        f"{tests_per_second:.1f}",
        delta="tests/sec",
        delta_color="off",
    )
    eta_placeholder.metric("⏱️ Durée Totale", time_str)
    completed_placeholder.metric(
        "✅ Résultats", f"{completed:,}", delta="100%", delta_color="normal"
    )

    logger.debug(f"[Sweep] {total_combinations} combinaisons générées")

    # Normaliser le format des résultats
    # SweepRunner retourne un format avec colonnes ['params', 'stats', 'error']
    # où 'stats' contient les métriques imbriquées

    # Debug: afficher les colonnes disponibles
    st.caption(f"🔍 Colonnes retournées par le sweep: {list(results_df.columns)}")

    # Debug: compter les erreurs
    if "error" in results_df.columns:
        n_errors = results_df["error"].notna().sum()
        if n_errors > 0:
            st.warning(f"⚠️ {n_errors}/{len(results_df)} résultats ont des erreurs")
            with st.expander("🔍 Debug: Erreurs détectées", expanded=True):
                errors_sample = results_df[results_df["error"].notna()][
                    ["params", "error"]
                ].head(5)
                for idx, row in errors_sample.iterrows():
                    st.error(f"Erreur: {row['error']}")
                    st.json(row["params"])

    # Extraire les métriques depuis la colonne 'stats' si présente
    if "stats" in results_df.columns:
        st.caption("📊 Extraction des métriques depuis la colonne 'stats'...")

        # Fonction pour extraire les métriques d'un objet stats
        def extract_metrics(stats_obj):
            if stats_obj is None:
                return {
                    "sharpe": 0.0,
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                }

            # Si c'est déjà un dict
            if isinstance(stats_obj, dict):
                return {
                    "sharpe": stats_obj.get(
                        "sharpe_ratio", stats_obj.get("sharpe", 0.0)
                    ),
                    "pnl": stats_obj.get("total_pnl", stats_obj.get("pnl", 0.0)),
                    "pnl_pct": stats_obj.get(
                        "total_pnl_pct", stats_obj.get("pnl_pct", 0.0)
                    ),
                    "max_drawdown": stats_obj.get("max_drawdown", 0.0),
                    "win_rate": stats_obj.get(
                        "win_rate_pct", stats_obj.get("win_rate", 0.0)
                    ),
                    "total_trades": stats_obj.get("total_trades", 0),
                }

            # Si c'est un objet avec attributs
            try:
                return {
                    "sharpe": getattr(
                        stats_obj, "sharpe_ratio", getattr(stats_obj, "sharpe", 0.0)
                    ),
                    "pnl": getattr(
                        stats_obj, "total_pnl", getattr(stats_obj, "pnl", 0.0)
                    ),
                    "pnl_pct": getattr(
                        stats_obj, "total_pnl_pct", getattr(stats_obj, "pnl_pct", 0.0)
                    ),
                    "max_drawdown": getattr(stats_obj, "max_drawdown", 0.0),
                    "win_rate": getattr(stats_obj, "win_rate", 0.0),
                    "total_trades": getattr(stats_obj, "total_trades", 0),
                }
            except:
                return {
                    "sharpe": 0.0,
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                }

        # Extraire les métriques
        metrics_list = results_df["stats"].apply(extract_metrics)

        # Créer un DataFrame depuis les métriques
        metrics_df = pd.DataFrame(list(metrics_list))

        # Fusionner avec le DataFrame principal
        for col in metrics_df.columns:
            results_df[col] = metrics_df[col]

        st.caption(f"✅ Métriques extraites: {list(metrics_df.columns)}")

        # DEBUG: Afficher un échantillon pour diagnostiquer les 0.0
        with st.expander("🔍 Debug: Échantillon résultats", expanded=False):
            st.markdown("**Première ligne de stats:**")
            first_stats = results_df["stats"].iloc[0]
            st.json(first_stats if isinstance(first_stats, dict) else str(first_stats))

            st.markdown("**Métriques extraites (première ligne):**")
            st.json(metrics_df.iloc[0].to_dict())

            st.markdown(
                "**Stats détaillées (5 premières configs avec meilleur Sharpe):**"
            )
            debug_cols = ["sharpe", "pnl", "pnl_pct", "max_drawdown", "total_trades"]
            available_debug_cols = [c for c in debug_cols if c in metrics_df.columns]
            if available_debug_cols:
                st.dataframe(metrics_df[available_debug_cols].nlargest(5, "sharpe"))

    # Normaliser les noms de colonnes
    if "sharpe" in results_df.columns and "sharpe_ratio" not in results_df.columns:
        results_df["sharpe_ratio"] = results_df["sharpe"]

    if "pnl_pct" in results_df.columns and "total_return" not in results_df.columns:
        results_df["total_return"] = results_df["pnl_pct"]

    # Vérification finale
    if "sharpe_ratio" not in results_df.columns:
        st.error(
            f"❌ Impossible d'extraire 'sharpe_ratio'. Colonnes finales: {list(results_df.columns)}"
        )
        if len(results_df) > 0:
            st.caption("Exemple de ligne:")
            st.json(results_df.iloc[0].to_dict())
        raise Exception("Colonne 'sharpe_ratio' manquante après normalisation")

    # Convertir DataFrame en liste de dicts pour compatibilité avec le reste du code
    results = results_df.to_dict("records")

    best_sharpe = max(
        (r.get("sharpe_ratio", r.get("sharpe", 0)) for r in results), default=0
    )
    logger.info(
        f"[Sweep] Terminé - {len(results)} résultats, "
        f"meilleur sharpe:{best_sharpe:.3f}, durée:{elapsed_time:.1f}s"
    )

    return results


def test_proposals(
    strategy_name: str, proposals: list, baseline_config: dict, use_gpu: bool
):
    """Teste chaque proposition et retourne les résultats."""

    logger.info(f"[Test Proposals] Démarrage - {len(proposals)} propositions à tester")

    # Extraire les paramètres baseline pour compléter les propositions partielles
    param_specs = parameter_specs_for(strategy_name)
    baseline_params = baseline_config.get("params", {})
    if not baseline_params:
        baseline_params = {
            k: v for k, v in baseline_config.items() if k in param_specs.keys()
        }

    # Utiliser les mêmes données que le sweep (depuis session_state)
    if "data" in st.session_state and not st.session_state.data.empty:
        df_market = st.session_state.data
        st.caption(f"📊 Test des propositions sur {len(df_market)} barres réelles")
    else:
        st.error(
            "❌ Aucune donnée disponible en cache. Le sweep aurait dû charger les données."
        )
        return []

    # Utiliser run_backtest_gpu au lieu de BacktestEngine
    test_results = []

    for prop in proposals:
        try:
            st.caption(f"🧪 Test de '{prop['name']}' avec params: {prop['params']}")

            # Fusion baseline + proposition pour obtenir un set complet
            merged_params = {**baseline_params, **prop.get("params", {})}
            if len(merged_params) < len(param_specs):
                missing = set(param_specs.keys()) - set(merged_params.keys())
                logger.warning(
                    "[Test Proposals] %s: paramètres manquants -> complétés depuis baseline: %s",
                    prop.get("name", "Unknown"),
                    missing,
                )
                st.info(
                    f"ℹ️ Params manquants complétés depuis la baseline: {', '.join(sorted(missing))}"
                )

            # CORRECTION: Forcer conversion int + contraintes (slow>fast, risk bounds)
            cleaned_params = _force_integer_params(merged_params)
            cleaned_params = _apply_strategy_constraints(
                cleaned_params, strategy_name, context=prop.get("name", "proposal")
            )

            result = run_backtest_gpu(
                df=df_market,
                strategy=strategy_name,
                params=cleaned_params,
            )

            test_results.append(
                {
                    "name": prop["name"],
                    "params": merged_params,  # Params réellement testés (baseline + overrides)
                    "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
                    "total_return": result.metrics.get("total_return", 0.0),
                    "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                    "win_rate": result.metrics.get("win_rate", 0.0),
                    "trades": (
                        result.trades if hasattr(result, "trades") else []
                    ),  # ← Capture trades
                    "full_result": result,  # ← Capture résultat complet pour analyses futures
                }
            )

            sharpe = result.metrics.get("sharpe_ratio", 0.0)
            st.success(f"✅ '{prop['name']}' testé : Sharpe={sharpe:.3f}")
            logger.debug(f"[Test Proposals] {prop['name']}: sharpe={sharpe:.3f}")

        except Exception as e:
            logger.warning(f"[Test Proposals] Erreur sur {prop['name']}: {e}")
            st.error(f"❌ Erreur test '{prop['name']}': {str(e)}")
            st.caption(f"Paramètres reçus: {prop['params']}")
            continue

    logger.info(f"[Test Proposals] Terminé - {len(test_results)} résultats")
    return test_results


def display_analyst_results(analysis: dict, elapsed: float):
    """Affiche les résultats de l'Analyst de manière formatée."""

    st.markdown(f"*Temps d'analyse: {elapsed:.1f}s*")

    st.markdown("---")

    # Patterns
    st.markdown("**🎯 Patterns identifiés:**")
    for i, pattern in enumerate(analysis["analysis"]["patterns"], 1):
        st.markdown(f"{i}. {pattern}")

    # Métriques clés
    st.markdown("\n**📈 Métriques clés:**")
    key_metrics = analysis["analysis"].get("key_metrics", {})
    if key_metrics:
        cols = st.columns(len(key_metrics))
        for col, (metric, value) in zip(cols, key_metrics.items()):
            try:
                col.metric(metric.replace("_", " ").title(), f"{value:.3f}")
            except (TypeError, ValueError):
                col.metric(metric.replace("_", " ").title(), str(value))
    else:
        st.caption("Aucune métrique clé disponible")

    # Trade-offs
    st.markdown("\n**⚖️ Trade-offs observés:**")
    trade_offs = analysis["analysis"].get("trade_offs", [])
    if trade_offs:
        for i, tradeoff in enumerate(trade_offs, 1):
            st.markdown(f"{i}. {tradeoff}")
    else:
        st.info("Aucun trade-off identifié")

    # Recommandations
    st.markdown("\n**💡 Recommandations:**")
    recommendations = analysis["analysis"].get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.success(f"**Rec {i}:** {rec}")
    else:
        st.warning("❌ Le LLM n'a pas généré de recommandations structurées")

        # Fallback: Chercher dans le texte brut si disponible
        raw_response = analysis.get("_raw_response", "")
        if "recommand" in raw_response.lower():
            st.info(
                "💡 Recommandations détectées dans la réponse brute - voir logs pour détails"
            )
            with st.expander("📝 Réponse LLM brute"):
                st.text(raw_response[:2000])


def display_strategist_results(
    proposals: dict, baseline: dict, baseline_sharpe: float, elapsed: float
):
    """Affiche les propositions du Strategist."""

    st.markdown(f"*Temps de génération: {elapsed:.1f}s*")
    st.markdown(
        f"*Propositions valides: {proposals['total_valid']}/{proposals['total_generated']}*"
    )

    st.markdown("---")

    st.markdown(f"**📊 Baseline actuelle:** Sharpe = {baseline_sharpe:.3f}")
    st.caption(f"Params: {baseline}")

    st.markdown("\n**💡 Nouvelles propositions:**")

    for i, prop in enumerate(proposals["proposals"], 1):
        with st.expander(f"**Proposition {i}: {prop['name']}**", expanded=True):
            st.markdown(f"*{prop['rationale']}*")

            st.markdown("**Modifications:**")
            for param, new_val in prop["params"].items():
                old_val = baseline.get(param, "N/A")
                if old_val != new_val:
                    st.markdown(f"- `{param}`: {old_val} → **{new_val}**")


def display_final_report(
    baseline: dict, proposals: list, test_results: list, analysis: dict
):
    """Affiche le rapport final avec visualisations."""

    st.markdown("## 🏆 Rapport Final")

    # Vérifier si des tests ont réussi
    if len(test_results) == 0:
        st.warning(
            "⚠️ Aucune alternative n'a pu être testée - seule la baseline est disponible."
        )
        st.info(
            "🔍 Causes possibles : paramètres incompatibles, erreurs dans run_backtest_gpu(), ou stratégie non supportée."
        )
        return

    # Préparer données pour graphiques
    comparison_data = [
        {
            "Config": "BASELINE",
            "Sharpe": baseline.get("sharpe_ratio", 0.0),
            "Return": baseline.get("total_return", 0.0),
            "Drawdown": abs(baseline.get("max_drawdown", 0.0)),
        }
    ]

    for res in test_results:
        comparison_data.append(
            {
                "Config": res["name"],
                "Sharpe": res["sharpe_ratio"],
                "Return": res["total_return"],
                "Drawdown": abs(res["max_drawdown"]),
            }
        )

    df_comparison = pd.DataFrame(comparison_data)

    # Graphiques comparatifs
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Sharpe Ratio", "Total Return", "Max Drawdown"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
    )

    # Sharpe
    colors = ["gray"] + ["steelblue"] * len(test_results)
    fig.add_trace(
        go.Bar(
            x=df_comparison["Config"],
            y=df_comparison["Sharpe"],
            marker_color=colors,
            name="Sharpe",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Return
    fig.add_trace(
        go.Bar(
            x=df_comparison["Config"],
            y=df_comparison["Return"] * 100,
            marker_color=colors,
            name="Return",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Drawdown
    fig.add_trace(
        go.Bar(
            x=df_comparison["Config"],
            y=df_comparison["Drawdown"] * 100,
            marker_color=colors,
            name="Drawdown",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=3)

    fig.update_layout(height=400, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # Identifier meilleure config
    best_idx = df_comparison["Sharpe"].idxmax()
    best_config = df_comparison.iloc[best_idx]

    st.markdown("### 🏅 Meilleure Configuration")

    if best_idx == 0:
        st.info("⚠️ **Baseline** reste la meilleure config")
    else:
        best_proposal = test_results[best_idx - 1]
        improvement = (
            (best_config["Sharpe"] - baseline["sharpe_ratio"])
            / abs(baseline["sharpe_ratio"])
        ) * 100

        st.success(
            f"🎉 **{best_config['Config']}** améliore le Sharpe de **{improvement:+.1f}%** !"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Sharpe Ratio",
            f"{best_config['Sharpe']:.3f}",
            f"{best_config['Sharpe'] - baseline['sharpe_ratio']:+.3f}",
        )
        col2.metric(
            "Return",
            f"{best_config['Return']:.2%}",
            f"{best_config['Return'] - baseline['total_return']:+.2%}",
        )
        col3.metric("Drawdown", f"{best_config['Drawdown']:.2%}")

        st.markdown("**Paramètres:**")
        st.json(best_proposal["params"])

        # 📊 VISUALISATION CANDLESTICK + TRADES (meilleure config)
        st.markdown("---")
        with st.expander(
            "📈 Visualisation Graphique (Bougies + Trades)", expanded=True
        ):
            # Récupérer données OHLCV depuis session_state
            symbol = st.session_state.get("symbol", "BTCUSDC")
            timeframe = st.session_state.get("timeframe", "1h")
            start_date = st.session_state.get("start_date")
            end_date = st.session_state.get("end_date")

            # Charger données de marché
            try:
                df_ohlcv = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)

                # Récupérer trades de la meilleure proposition
                best_trades = best_proposal.get("trades", [])

                if not best_trades:
                    st.info("ℹ️ Aucun trade enregistré pour cette configuration")
                else:
                    render_candlestick_with_trades(
                        df_ohlcv=df_ohlcv,
                        trades=best_trades,
                        title=f"🏆 {best_config['Config']} - {len(best_trades)} trades",
                        key_suffix=f"best_{best_idx}",
                    )
            except Exception as e:
                st.error(f"❌ Erreur chargement données: {e}")

    # Tableau comparatif
    st.markdown("### 📋 Tableau Comparatif")
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    # 📊 VISUALISATIONS SUPPLÉMENTAIRES (toutes les configurations)
    st.markdown("---")
    st.markdown("### 📊 Graphiques Détaillés par Configuration")

    # Récupérer données OHLCV une seule fois
    symbol = st.session_state.get("symbol", "BTCUSDC")
    timeframe = st.session_state.get("timeframe", "1h")
    start_date = st.session_state.get("start_date")
    end_date = st.session_state.get("end_date")

    try:
        df_ohlcv = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)

        # Afficher baseline
        with st.expander("📊 BASELINE - Configuration initiale", expanded=False):
            st.info(
                "⚠️ Trades de baseline non enregistrés (seules les propositions sont testées)"
            )

        # Afficher chaque proposition
        for i, res in enumerate(test_results, 1):
            config_trades = res.get("trades", [])
            config_name = res["name"]
            sharpe = res.get("sharpe_ratio")
            sharpe_display = (
                f"{sharpe:.3f}" if isinstance(sharpe, (int, float)) else "N/A"
            )

            with st.expander(
                f"📊 Proposition {i}: {config_name} (Sharpe: {sharpe_display})",
                expanded=False,
            ):
                if not config_trades:
                    st.info("ℹ️ Aucun trade enregistré")
                else:
                    render_candlestick_with_trades(
                        df_ohlcv=df_ohlcv,
                        trades=config_trades,
                        title=f"{config_name} - {len(config_trades)} trades",
                        key_suffix=f"prop_{i}",
                    )
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger les données OHLCV: {e}")


def render_candlestick_with_trades(
    df_ohlcv: pd.DataFrame,
    trades: List[Dict[str, Any]],
    title: str = "📈 Graphique Prix + Trades",
    key_suffix: str = "default",
) -> None:
    """
    Affiche un graphique candlestick (bougies japonaises) avec les entrées/sorties de trades.

    Args:
        df_ohlcv: DataFrame avec colonnes [open, high, low, close] et index datetime
        trades: Liste de trades avec {entry_time, entry_price, exit_time, exit_price, side, pnl}
        title: Titre du graphique
        key_suffix: Suffixe unique pour éviter collisions de clés Streamlit
    """
    if not isinstance(df_ohlcv, pd.DataFrame) or df_ohlcv.empty:
        st.warning("⚠️ Données OHLCV indisponibles pour le tracé")
        return

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df_ohlcv.columns):
        st.warning(
            f"⚠️ Colonnes manquantes. Requis: {required_cols}, Disponible: {list(df_ohlcv.columns)}"
        )
        return

    st.markdown(f"#### {title}")

    # Créer figure Plotly
    fig = go.Figure()

    # Ajouter les bougies japonaises
    fig.add_trace(
        go.Candlestick(
            x=df_ohlcv.index,
            open=df_ohlcv["open"],
            high=df_ohlcv["high"],
            low=df_ohlcv["low"],
            close=df_ohlcv["close"],
            name="Prix",
            increasing_line_color="#26a69a",  # Vert pour hausse
            decreasing_line_color="#ef5350",  # Rouge pour baisse
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        )
    )

    # Préparer les données des trades
    entries_long_x, entries_long_y = [], []
    entries_short_x, entries_short_y = [], []
    exits_profit_x, exits_profit_y = [], []
    exits_loss_x, exits_loss_y = [], []

    for trade in trades or []:
        side = str(trade.get("side", "LONG")).upper()
        # FIX: Support both "pnl" (BacktestEngine) and "pnl_realized" (Strategy classes)
        pnl = trade.get("pnl", trade.get("pnl_realized", 0))

        # Entrées (différencier LONG/SHORT)
        # CRITICAL: Check values are not None to avoid Plotly format errors
        entry_time = trade.get("entry_time")
        entry_price = trade.get("entry_price")
        if entry_time is not None and entry_price is not None:
            if side == "LONG":
                entries_long_x.append(entry_time)
                entries_long_y.append(entry_price)
            else:
                entries_short_x.append(entry_time)
                entries_short_y.append(entry_price)

        # Sorties (différencier profit/perte)
        # CRITICAL: Check values are not None
        exit_time = trade.get("exit_time")
        exit_price = trade.get("exit_price")
        if exit_time is not None and exit_price is not None:
            if pnl > 0:
                exits_profit_x.append(exit_time)
                exits_profit_y.append(exit_price)
            else:
                exits_loss_x.append(exit_time)
                exits_loss_y.append(exit_price)

    # Ajouter traces pour entrées LONG
    if entries_long_x:
        fig.add_trace(
            go.Scatter(
                x=entries_long_x,
                y=entries_long_y,
                mode="markers",
                name="Entrée LONG",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#42a5f5",  # Bleu
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="<b>Entrée LONG</b><br>Prix: %{y:.2f}<br>%{x}<extra></extra>",
            )
        )

    # Ajouter traces pour entrées SHORT
    if entries_short_x:
        fig.add_trace(
            go.Scatter(
                x=entries_short_x,
                y=entries_short_y,
                mode="markers",
                name="Entrée SHORT",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#ab47bc",  # Violet
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="<b>Entrée SHORT</b><br>Prix: %{y:.2f}<br>%{x}<extra></extra>",
            )
        )

    # Ajouter traces pour sorties profitables
    if exits_profit_x:
        fig.add_trace(
            go.Scatter(
                x=exits_profit_x,
                y=exits_profit_y,
                mode="markers",
                name="Sortie Profit",
                marker=dict(
                    symbol="star",
                    size=10,
                    color="#66bb6a",  # Vert
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="<b>Sortie Profit</b><br>Prix: %{y:.2f}<br>%{x}<extra></extra>",
            )
        )

    # Ajouter traces pour sorties en perte
    if exits_loss_x:
        fig.add_trace(
            go.Scatter(
                x=exits_loss_x,
                y=exits_loss_y,
                mode="markers",
                name="Sortie Perte",
                marker=dict(
                    symbol="x",
                    size=10,
                    color="#ef5350",  # Rouge
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="<b>Sortie Perte</b><br>Prix: %{y:.2f}<br>%{x}<extra></extra>",
            )
        )

    # Mise en page
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_dark",
        xaxis_title="Temps",
        yaxis_title="Prix (USD)",
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="rgba(128,128,128,0.2)",
            type="date",
        ),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        font=dict(size=11),
    )

    # Affichage
    st.plotly_chart(fig, use_container_width=True, key=f"candlestick_{key_suffix}")

    # Stats rapides
    if trades:
        col1, col2, col3, col4 = st.columns(4)
        total_trades = len(trades)
        # FIX: Support both "pnl" (BacktestEngine) and "pnl_realized" (Strategy classes)
        profit_trades = sum(
            1 for t in trades if t.get("pnl", t.get("pnl_realized", 0)) > 0
        )
        loss_trades = total_trades - profit_trades
        win_rate = (profit_trades / total_trades * 100) if total_trades > 0 else 0

        col1.metric("📊 Total Trades", total_trades)
        col2.metric("✅ Profits", profit_trades)
        col3.metric("❌ Pertes", loss_trades)
        col4.metric("🎯 Win Rate", f"{win_rate:.1f}%")


if __name__ == "__main__":
    render_page()
