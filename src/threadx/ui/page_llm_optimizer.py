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

from threadx.data_access import load_ohlcv
from threadx.indicators.bank import IndicatorBank, IndicatorSettings
from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.strategist import Strategist
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec
from threadx.ui.backtest_bridge import run_backtest_gpu
from threadx.ui.strategy_registry import parameter_specs_for, get_sweep_preset, SWEEP_PRESETS
from threadx.llm.model_router import ModelRouter, TaskType
from threadx.llm.ollama_manager import prepare_for_llm_run


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
            timeout=5
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
    st.markdown("""
    **Système collaboratif d'agents LLM** pour optimisation automatique de stratégies.
    
    **Workflow**:
    1. 🔄 Sweep GPU → Test multiple configurations
    2. 🧠 Analyst (deepseek-r1:32b) → Analyse quantitative & patterns
    3. 🎨 Strategist (deepseek-r1:32b) → Propositions créatives (cohérence maximale)
    4. ✅ Tests automatiques → Validation performances
    5. 📊 Visualisation → Comparaison résultats
    
    💡 **Nouveau**: Les deux agents utilisent désormais le même modèle (DeepSeek-R1 32B) pour une cohérence optimale.
    """)

    # Vérifier prérequis
    with st.expander("⚙️ Prérequis & Configuration", expanded=False):
        check_prerequisites()

    # Aide optimisation Ollama GPU
    with st.expander("🔧 Résoudre erreurs CUDA Ollama (saturation GPU)", expanded=False):
        st.markdown("""
        **Si vous rencontrez une erreur `CUDA error` ou `llama runner process has terminated`**, 
        cela indique généralement une saturation de la mémoire GPU lors du chargement/inférence du modèle.
        
        ### 📊 Diagnostic rapide
        """)

        st.code("""# Vérifier occupation GPU
nvidia-smi
nvidia-smi -q -d MEMORY""", language="powershell")

        st.markdown("""
        ### ⚙️ Réglages recommandés (Ollama)
        
        **1. Localiser la configuration du modèle**
        - Dossier modèles Ollama : `%USERPROFILE%\\.ollama\\models\\`
        - Cherchez le fichier `config.json` ou `modelfile` du modèle utilisé
        
        **2. Appliquer ces paramètres (sauvegardez l'original avant)**
        """)

        config_snippet = {
            "use_mmap": True,
            "num_gpu_layers": 48,
            "batch_size": 256,
            "num_threads": 16,
            "main_gpu": 0
        }

        st.json(config_snippet)

        st.markdown("""
        **Explications :**
        - `use_mmap: true` → Réduit les pics mémoire lors du chargement
        - `num_gpu_layers: 48` → Moins de layers GPU = plus de marge (au lieu de 63)
        - `batch_size: 256` → Réduit la taille batch (au lieu de 512)
        - `main_gpu: 0` → Utilisez le GPU avec le plus de mémoire (vérifiez avec `nvidia-smi`)
        
        **3. Redémarrer Ollama proprement**
        """)

        st.code("""# Arrêter Ollama
Stop-Process -Name ollama -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Vérifier GPU libéré
nvidia-smi

# Relancer Ollama
ollama serve""", language="powershell")

        st.markdown("""
        **4. Tester avec une requête courte**
        """)

        st.code("""python -c \"from threadx.llm.client import LLMClient
from threadx.llm.ollama_manager import prepare_for_llm_run; c = LLMClient(model='deepseek-r1:8b'); print(c.complete('Test ok'))\" """, language="powershell")

        st.warning("""
        **Si le problème persiste :**
        - Essayez un modèle plus petit : `deepseek-r1:1.5b` ou `gemma2:2b`
        - Réduisez encore `num_gpu_layers` : essayez 40, puis 32
        - Fermez autres applications GPU (browsers, jeux, Docker, etc.)
        - En dernier recours : utilisez un modèle CPU-only (perte de performance)
        """)

    st.divider()

    # Configuration de base
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Configuration Sweep")

        strategy_name = st.selectbox(
            "Stratégie",
            options=["MA_Crossover", "Bollinger_Breakout", "EMA_Cross", "ATR_Channel"],
            index=0,  # MA_Crossover sélectionné par défaut
            help="Stratégie à optimiser"
        )

        # Récupérer specs de la stratégie
        param_specs = parameter_specs_for(strategy_name)

        st.markdown("**Paramètres du sweep:**")
        sweep_params = {}

        for param_name, spec in param_specs.items():
            param_type = spec.get("type", "number")
            is_preset = False  # Flag pour savoir si c'est un préréglage

            if param_type == "boolean":
                sweep_params[param_name] = [False, True]
                st.caption(f"✓ {param_name}: [False, True]")
                continue  # Passer au paramètre suivant

            # Utiliser préréglages globaux si disponibles (source unique)
            preset = get_sweep_preset(param_name)
            if preset:
                min_val = preset["min"]
                max_val = preset["max"]
                n_values = preset["n_values"]
                is_preset = True
                # Afficher la raison du préréglage
                reason = SWEEP_PRESETS[param_name]["reason"]
                st.caption(f"🔒 {param_name}: {min_val} (fixé - {reason})")
            else:
                min_val = spec.get("min", 0)
                max_val = spec.get("max", 100)
                step = spec.get("step", 1)

                # Générer 3-4 valeurs dans la plage
                n_values = st.slider(
                    f"Nombre valeurs {param_name}",
                    min_value=2,
                    max_value=6,
                    value=3,
                    key=f"n_{param_name}"
                )

            # Générer valeurs avec protection division par zéro et validation
            if min_val is None or max_val is None:
                st.warning(f"⚠️ Valeurs min/max invalides pour {param_name}, ignoré")
                continue
            if n_values == 1:
                values = [min_val]
            else:
                values = [min_val + i * (max_val - min_val) / (n_values - 1)
                         for i in range(n_values)]

            if param_type == "integer":
                values = [int(v) for v in values]

            sweep_params[param_name] = values

            # Afficher les valeurs générées uniquement si pas de préréglage
            if not is_preset:
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
                ["🎯 Radar normalisé", "📊 Barres horizontales", "📈 Barres avec échelles individuelles"],
                horizontal=True,
                help="Radar = vue d'ensemble, Barres = valeurs précises par paramètre",
                key="llm_viz_type"
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
                    hover_texts.append(f"{key}<br>Min: {min_values[i]:.4g}<br>Max: {max_values[i]:.4g}<br>Span: {span:.4g}")

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
                            tickmode='linear',
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
                        orientation='h',
                        marker=dict(
                            color=spans_normalized,
                            colorscale='Teal',
                            showscale=False,
                        ),
                        text=[f"{s:.1f}%<br>({spans_raw[i]:.4g})" for i, s in enumerate(spans_normalized)],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Min: %{customdata[0]:.4g}<br>Max: %{customdata[1]:.4g}<br>Span: %{customdata[2]:.4g}<extra></extra>',
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
                st.caption("📊 Barres horizontales avec valeurs normalisées (hover pour détails)")

            else:  # Barres avec échelles individuelles
                # === SUBPLOTS avec échelle par paramètre ===
                from plotly.subplots import make_subplots

                n_params = len(labels)
                fig_dist = make_subplots(
                    rows=n_params, cols=1,
                    subplot_titles=labels,
                    vertical_spacing=0.08,
                    specs=[[{"type": "bar"}] for _ in range(n_params)]
                )

                for i, (label, min_v, max_v, span) in enumerate(zip(labels, min_values, max_values, spans_raw), 1):
                    fig_dist.add_trace(
                        go.Bar(
                            x=[span],
                            y=[label],
                            orientation='h',
                            marker=dict(color='#26a69a'),
                            text=f"{min_v:.4g} → {max_v:.4g}",
                            textposition='outside',
                            showlegend=False,
                            hovertemplate=f'<b>{label}</b><br>Min: {min_v:.4g}<br>Max: {max_v:.4g}<br>Span: {span:.4g}<extra></extra>',
                        ),
                        row=i, col=1
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

            st.plotly_chart(fig_dist, use_container_width=True, key="llm_param_distribution")

    with col2:
        st.subheader("🤖 Configuration LLM")

        # Mode économie mémoire pour éviter de charger plusieurs modèles simultanément
        memory_saver = st.checkbox(
            "💾 Mode économie mémoire",
            value=True,
            help="Utilise un seul modèle pour tous les agents et décharge la mémoire entre chaque étape. Recommandé pour éviter la saturation RAM."
        )

        # Liste des modèles disponibles avec infos GPU
        available_models = [
            ("deepseek-r1:32b", "DeepSeek-R1 32B (~19GB) - ARCHITECT & BUILDER"),
            ("gemma3:27b", "Gemma 3 27B (~17GB) - GUEST"),
            ("qwen3-vl:30b", "Qwen 3 VL 30B (~19GB) - GUEST"),
            ("gpt-oss:20b", "GPT-OSS 20B (~13GB) - GUEST"),
            ("gemma3:12b", "Gemma 3 12B (~8GB)"),
            ("deepseek-r1-distill:14b", "DeepSeek-R1 14B (~9GB)"),
            ("deepseek-r1:8b", "DeepSeek-R1 8B (~5GB)"),
            ("mistral:7b-instruct", "Mistral 7B (~4GB)"),
        ]

        model_names = [m[0] for m in available_models]
        model_labels = [f"{m[0]} - {m[1]}" for m in available_models]

        # Initialiser le routeur de modèles
        router = ModelRouter()
        
        st.info("💡 **Stratégie Multi-LLM Active** : Architecte (32B) + Bâtisseur (32B) + Auditeurs")
        
        # Afficher les modèles qui seront utilisés
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("🏗️ Architecte", router.ARCHITECT_MODEL, help="Initialisation robuste")
        with col_m2:
            st.metric("🔨 Bâtisseur", router.BUILDER_MODEL, help="Optimisation itérative")
        with col_m3:
            st.metric("👀 Auditeurs", "Rotation", help=f"Rotation: {', '.join(router.GUEST_MODELS)}")
        
        # Pour compatibilité avec le reste du code, on définit des valeurs par défaut
        # Mais le routeur sera prioritaire dans la boucle d'optimisation
        analyst_model = router.get_model_for_task(TaskType.INITIALIZATION)
        strategist_model = router.get_model_for_task(TaskType.OPTIMIZATION)

        n_proposals = st.slider(
            "Nombre de propositions",
            min_value=1,
            max_value=5,
            value=3,
            help="Propositions créatives générées par Strategist"
        )

        top_n_analysis = st.slider(
            "Top N configs à analyser",
            min_value=3,
            max_value=10,
            value=5,
            help="Nombre de meilleures configs analysées par Analyst"
        )

        # Checkbox analyse IA (cochée par défaut)
        enable_ai_analysis = st.checkbox(
            "⚡ Activer l'analyse IA pour la meilleure configuration",
            value=True,
            help="Les LLM analyseront les résultats pour proposer des optimisations"
        )

        # Checkbox streaming (afficher réflexion en temps réel)
        enable_streaming = st.checkbox(
            "🧠 Afficher réflexion LLM en temps réel",
            value=True,
            help="Voir le raisonnement Chain-of-Thought de DeepSeek-R1 pendant la génération"
        )

    # Configuration depuis la sidebar (valeurs globales)
    st.markdown("### ⚙️ Configuration Moteur de Sweep")

    # Récupérer les valeurs globales de la sidebar
    use_gpu = True  # Toujours activé
    use_multigpu = st.session_state.get("global_use_multigpu", True)
    max_workers = st.session_state.get("global_workers", 30)
    feeder_aggr = st.session_state.get("global_feeder_aggr", 16)

    # Afficher statut de configuration
    st.info(f"🎛️ Configuration depuis **sidebar** : {st.session_state.get('global_perf_profile', '⚖️ Équilibré')}")

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

    st.caption("💡 Modifier la configuration dans la **sidebar** (section Configuration Globale)")

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
            st.markdown("""
            **Instructions système pour Analyst & Strategist** :
            
            🎯 **Objectifs prioritaires** :
            - Maximiser le Sharpe Ratio (risque/rendement)
            - Minimiser le drawdown maximum
            - Maintenir un win rate > 50%
            - Optimiser le nombre de trades (ni trop, ni trop peu)
            
            📊 **Approche d'analyse** :
            - Identifier les patterns dans les meilleures configurations
            - Détecter les corrélations entre paramètres
            - Proposer des modifications incrémentales (pas de changements brutaux)
            - Valider la cohérence des propositions avec les contraintes de risque
            
            ⚠️ **Contraintes** :
            - `risk_per_trade` : Rester dans [0.005, 0.02]
            - `max_hold_bars` : Adapter selon la volatilité détectée
            - Stop Loss / Take Profit : Ratio min 1:1.5
            - Toujours respecter les plages min/max des paramètres
            
            💡 **Recommandations** :
            - Privilégier la robustesse à la performance brute
            - Tester les propositions sur différents régimes de marché
            - Documenter clairement le raisonnement derrière chaque modification
            """)

    st.divider()

    # Bouton de lancement
    if st.button("🚀 Lancer l'optimisation Multi-LLM", type="primary", use_container_width=True):
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
        )


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
    max_tokens: int = 2000
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
    from threadx.llm.client import LLMClient
    import re
    
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
                think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    thinking_placeholder.markdown(
                        f"**💭 Raisonnement Chain-of-Thought:**\n\n{thinking}"
                    )
                
                # Afficher JSON si détecté
                json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
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
):
    """Exécute le workflow complet d'optimisation Multi-LLM.

    Args:
        memory_saver: Si True, décharge les modèles Ollama de la mémoire après chaque agent
        enable_streaming: Si True, affiche la réflexion Chain-of-Thought en temps réel
    """

    # Initialiser session state
    if "llm_results" not in st.session_state:
        st.session_state.llm_results = {}

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

        with st.spinner(f"Test de {len(list(_generate_combinations(sweep_params)))} configurations..."):
            sweep_results = execute_sweep(
                strategy_name=strategy_name,
                sweep_params=sweep_params,
                use_gpu=use_gpu,
                use_multigpu=use_multigpu,
                max_workers=max_workers,
                feeder_aggr=feeder_aggr,
                force_processpool=force_processpool,
            )

        st.session_state.llm_results["sweep"] = sweep_results

        # Afficher résultats sweep
        with results_container:
            st.success(f"✅ Sweep terminé: {len(sweep_results)} configs testées")

            with st.expander("📊 Top 10 configurations", expanded=True):
                df_sweep = pd.DataFrame(sweep_results)

                # Normaliser les noms de colonnes (les métriques devraient déjà être extraites par execute_sweep)
                # Mais vérification au cas où
                if "sharpe" in df_sweep.columns and "sharpe_ratio" not in df_sweep.columns:
                    df_sweep["sharpe_ratio"] = df_sweep["sharpe"]
                if "pnl_pct" in df_sweep.columns and "total_return" not in df_sweep.columns:
                    df_sweep["total_return"] = df_sweep["pnl_pct"]

                # Vérifier que sharpe_ratio existe avant de trier
                if "sharpe_ratio" not in df_sweep.columns:
                    st.error(f"❌ Colonne 'sharpe_ratio' manquante dans df_sweep. Colonnes: {list(df_sweep.columns)}")
                    if len(df_sweep) > 0:
                        st.caption("Premier résultat:")
                        st.json(df_sweep.iloc[0].to_dict())
                    raise Exception("Impossible de trier par sharpe_ratio - colonne manquante")

                top_10 = df_sweep.nlargest(10, "sharpe_ratio")
                st.dataframe(
                    top_10,
                    use_container_width=True,
                    hide_index=True,
                )

        progress_bar.progress(30)

        # ============================================================
        # ÉTAPE 2: ANALYSE ANALYST
        # ============================================================
        
        # Sélection du modèle (Router ou manuel)
        current_analyst_model = analyst_model
        if model_router:
            current_analyst_model = model_router.get_model_for_task(TaskType.INITIALIZATION)
            
        status_text.markdown(f"### 🧠 Étape 2/5: Analyse Analyst ({current_analyst_model})...")

        with st.spinner(f"Analyse des top {top_n_analysis} configs (peut prendre 30-60s)..."):
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
                        analysis_placeholder.info("🧠 Analyse en cours avec streaming Chain-of-Thought...")
                        
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
                            max_tokens=2000
                        )
                        
                        # Parser le JSON depuis la réponse
                        import json
                        import re
                        json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
                        if json_match:
                            try:
                                analysis_data = json.loads(json_match.group(0))
                            except json.JSONDecodeError:
                                st.error("❌ Erreur parsing JSON - utilisation fallback")
                                analysis_data = {
                                    "patterns": [],
                                    "key_metrics": {},
                                    "trade_offs": [],
                                    "recommendations": ["Analyser patterns identifiés", "Tester configurations avec Sharpe > moyenne"]
                                }
                        else:
                            st.error("❌ Pas de JSON détecté - utilisation fallback")
                            analysis_data = {
                                "patterns": [],
                                "key_metrics": {},
                                "trade_offs": [],
                                "recommendations": ["Analyser patterns identifiés"]
                            }
                        
                        # Construire résultat compatible
                        analysis_result = {
                            "analysis": analysis_data,
                            "top_configs": df_sweep.head(top_n_analysis).to_dict("records")
                        }
                        
                    else:
                        # Mode normal: appel direct sans streaming
                        analysis_placeholder.info("⏳ Analyse en cours...")
                        analysis_result = analyst.analyze_sweep_results(
                            sweep_df=df_sweep,
                            top_n=top_n_analysis
                        )
                    
                    elapsed = time.time() - start_time

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
            current_strategist_model = model_router.get_model_for_task(TaskType.OPTIMIZATION)
            
        status_text.markdown(f"### 🎨 Étape 3/5: Propositions Strategist ({current_strategist_model})...")

        # ─────────────────────────────────────────────────────────────────────
        # FONCTION DE VALIDATION STRATÉGIE-SPÉCIFIQUE
        # ─────────────────────────────────────────────────────────────────────
        def validate_baseline_coherence(config: dict, strat_name: str) -> tuple[bool, list[str]]:
            """
            Valide la cohérence d'une config selon la stratégie.
            
            Returns:
                (is_valid, warnings) : (True/False, liste de warnings)
            """
            warnings = []
            params = config.get('params', {})
            
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
                warnings.append(
                    f"⚠️ Risque: Leverage {leverage}x + Sharpe faible ({sharpe:.3f})"
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
                candidate.to_dict(), 
                strategy_name
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
                    st.info("ℹ️ **Remarques (non-bloquantes):**\n\n" + "\n".join(validation_warnings))
                
                break
            else:
                # Config invalide → skip
                st.warning(
                    f"⚠️ Config rang #{rank + 1} ignorée (Sharpe: {candidate['sharpe_ratio']:.3f}):\n"
                    + "\n".join(validation_warnings)
                )
        
        # Si aucune baseline valide trouvée dans top 20
        if baseline_config is None:
            st.error(
                f"❌ **Aucune config valide trouvée dans le top {max_candidates}**\n\n"
                "Vérifiez les paramètres du sweep ou élargissez les plages."
            )
            st.stop()

        # Debug : vérifier que les métriques sont présentes
        st.caption(f"📊 Baseline sélectionnée - Sharpe: {baseline_config.get('sharpe_ratio', 'N/A'):.3f}, Return: {baseline_config.get('total_return', 'N/A'):.3f}")

        # Extraire les paramètres depuis baseline_config['params']
        # SweepRunner retourne un format avec 'params' comme colonne séparée
        param_specs_full = parameter_specs_for(strategy_name)
        baseline_params = baseline_config.get('params', {})

        # Si 'params' n'existe pas, essayer à la racine (fallback)
        if not baseline_params:
            baseline_params = {k: v for k, v in baseline_config.items()
                              if k in param_specs_full.keys()}

        # Debug : afficher ce qui a été extrait
        st.caption(f"📊 Paramètres baseline extraits : {len(baseline_params)} sur {len(param_specs_full)}")
        with st.expander("🔍 Détails baseline complète", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Paramètres:**")
                st.json(baseline_params)
            with col2:
                st.markdown("**Métriques:**")
                metrics = {k: v for k, v in baseline_config.items()
                          if k in ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'pnl', 'pnl_pct']}
                st.json(metrics)

        with st.spinner(f"Génération de {n_proposals} propositions créatives (peut prendre 20-40s)..."):
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

                    proposals_result = strategist.propose_modifications(
                        analysis=analysis_result,
                        current_params=baseline_params,
                        param_specs=param_specs_full,
                        n_proposals=n_proposals
                    )
                    elapsed = time.time() - start_time

                    # Afficher propositions formatées
                    proposals_placeholder.empty()
                    display_strategist_results(
                        proposals_result,
                        baseline_params,
                        baseline_config.get("sharpe_ratio", 0),
                        elapsed
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

        with st.spinner(f"Test de {len(proposals_result['proposals'])} propositions..."):
            test_results = test_proposals(
                strategy_name=strategy_name,
                proposals=proposals_result["proposals"],
                baseline_config=baseline_config,
                use_gpu=use_gpu,
            )

        st.session_state.llm_results["tests"] = test_results

        # Vérifier si les tests ont réussi
        if len(test_results) == 0:
            st.warning("⚠️ Aucune proposition n'a pu être testée avec succès. Vérifiez les erreurs ci-dessus.")
            st.info("💡 Les propositions du Strategist nécessitent peut-être des paramètres manquants ou incompatibles avec la stratégie.")

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

        progress_bar.progress(100)
        status_text.success("### 🎉 Optimisation Multi-LLM terminée !")

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
        st.error(f"❌ Erreur inattendue: {e}")
        import traceback
        with st.expander("🐛 Traceback complet"):
            st.code(traceback.format_exc())


# ===================================================================
# PATCH CRITIQUE : Conversion forcée des paramètres en entiers
# Évite ValueError: min_periods must be an integer dans pandas.rolling
# ===================================================================
INTEGER_PARAM_KEYS = {
    "fast_period", "slow_period", "rsi_period", "bb_period",
    "atr_period", "ema_period", "macd_fast", "macd_slow", "macd_signal",
    "max_hold_bars", "lookback_bars", "trend_period"
}


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


def _generate_combinations(sweep_params: dict):
    """Génère toutes les combinaisons de paramètres."""
    from itertools import product

    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    for combo in product(*param_values):
        combo_dict = dict(zip(param_names, combo))
        # CORRECTION: Forcer conversion int pour paramètres de période
        yield _force_integer_params(combo_dict)


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
        st.caption(f"📊 Utilisation des données en cache: {len(df_market)} barres ({symbol}/{timeframe})")
    else:
        # Charger depuis la base de données
        try:
            df_market = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)
            if df_market.empty:
                st.error(f"❌ Aucune donnée disponible pour {symbol}/{timeframe} entre {start_date} et {end_date}")
                st.warning("💡 Allez d'abord dans l'onglet 'Configuration' pour charger les données.")
                return []
            st.session_state.data = df_market  # Mettre en cache
            st.caption(f"📊 Données chargées: {len(df_market)} barres ({df_market.index[0].date()} → {df_market.index[-1].date()})")
        except Exception as e:
            st.error(f"❌ Erreur chargement données: {e}")
            st.warning("💡 Allez d'abord dans l'onglet 'Configuration' pour charger les données.")
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

    # Calculer nombre total de combinaisons
    total_combinations = 1
    for values in sweep_params.values():
        total_combinations *= len(values)

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
                if current > 0 and elapsed > 0 and (current != last_current or (now - last_ui_update) >= 0.2):
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
                        delta_c = (current - last_current) if last_current >= 0 else current
                        delta_t = elapsed if last_current < 0 else (now - last_ui_update)
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
                        progress, text=f"⏳ {current:,}/{total:,} configs ({progress*100:.1f}%)"
                    )
                    status_placeholder.metric(
                        "📊 Statut",
                        "En cours ⚡",
                        delta=f"{speed:.1f} tests/sec (moy. 2s)",
                        delta_color="normal"
                    )
                    speed_placeholder.metric(
                        "🚀 Vitesse",
                        f"{speed:.1f}",
                        delta="tests/sec",
                        delta_color="off"
                    )
                    eta_placeholder.metric("⏱️ ETA", eta_str)
                    completed_placeholder.metric(
                        "✅ Complétés",
                        f"{current:,}",
                        delta=f"{(current/total*100):.1f}%",
                        delta_color="normal"
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
        raise Exception("Le sweep n'a produit aucun résultat valide. Vérifiez les paramètres et les données.")

    completed = len(results_df)
    tests_per_second = completed / elapsed_time if elapsed_time > 0 else 0
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes)}m {int(seconds)}s"

    # Stats finales
    progress_placeholder.progress(1.0, text=f"✅ Sweep terminé en {time_str} | {completed:,} résultats")
    status_placeholder.metric("📊 Statut", "✅ Terminé", delta="100%", delta_color="normal")
    speed_placeholder.metric(
        "🚀 Vitesse Moyenne",
        f"{tests_per_second:.1f}",
        delta="tests/sec",
        delta_color="off"
    )
    eta_placeholder.metric("⏱️ Durée Totale", time_str)
    completed_placeholder.metric("✅ Résultats", f"{completed:,}", delta="100%", delta_color="normal")

    # Normaliser le format des résultats
    # SweepRunner retourne un format avec colonnes ['params', 'stats', 'error']
    # où 'stats' contient les métriques imbriquées

    # Debug: afficher les colonnes disponibles
    st.caption(f"🔍 Colonnes retournées par le sweep: {list(results_df.columns)}")

    # Debug: compter les erreurs
    if 'error' in results_df.columns:
        n_errors = results_df['error'].notna().sum()
        if n_errors > 0:
            st.warning(f"⚠️ {n_errors}/{len(results_df)} résultats ont des erreurs")
            with st.expander("🔍 Debug: Erreurs détectées", expanded=True):
                errors_sample = results_df[results_df['error'].notna()][['params', 'error']].head(5)
                for idx, row in errors_sample.iterrows():
                    st.error(f"Erreur: {row['error']}")
                    st.json(row['params'])

    # Extraire les métriques depuis la colonne 'stats' si présente
    if 'stats' in results_df.columns:
        st.caption("📊 Extraction des métriques depuis la colonne 'stats'...")

        # Fonction pour extraire les métriques d'un objet stats
        def extract_metrics(stats_obj):
            if stats_obj is None:
                return {
                    'sharpe': 0.0,
                    'pnl': 0.0,
                    'pnl_pct': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0,
                }

            # Si c'est déjà un dict
            if isinstance(stats_obj, dict):
                return {
                    'sharpe': stats_obj.get('sharpe_ratio', stats_obj.get('sharpe', 0.0)),
                    'pnl': stats_obj.get('total_pnl', stats_obj.get('pnl', 0.0)),
                    'pnl_pct': stats_obj.get('total_pnl_pct', stats_obj.get('pnl_pct', 0.0)),
                    'max_drawdown': stats_obj.get('max_drawdown', 0.0),
                    'win_rate': stats_obj.get('win_rate_pct', stats_obj.get('win_rate', 0.0)),
                    'total_trades': stats_obj.get('total_trades', 0),
                }

            # Si c'est un objet avec attributs
            try:
                return {
                    'sharpe': getattr(stats_obj, 'sharpe_ratio', getattr(stats_obj, 'sharpe', 0.0)),
                    'pnl': getattr(stats_obj, 'total_pnl', getattr(stats_obj, 'pnl', 0.0)),
                    'pnl_pct': getattr(stats_obj, 'total_pnl_pct', getattr(stats_obj, 'pnl_pct', 0.0)),
                    'max_drawdown': getattr(stats_obj, 'max_drawdown', 0.0),
                    'win_rate': getattr(stats_obj, 'win_rate', 0.0),
                    'total_trades': getattr(stats_obj, 'total_trades', 0),
                }
            except:
                return {
                    'sharpe': 0.0,
                    'pnl': 0.0,
                    'pnl_pct': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0,
                }

        # Extraire les métriques
        metrics_list = results_df['stats'].apply(extract_metrics)

        # Créer un DataFrame depuis les métriques
        metrics_df = pd.DataFrame(list(metrics_list))

        # Fusionner avec le DataFrame principal
        for col in metrics_df.columns:
            results_df[col] = metrics_df[col]

        st.caption(f"✅ Métriques extraites: {list(metrics_df.columns)}")

        # DEBUG: Afficher un échantillon pour diagnostiquer les 0.0
        with st.expander("🔍 Debug: Échantillon résultats", expanded=False):
            st.markdown("**Première ligne de stats:**")
            first_stats = results_df['stats'].iloc[0]
            st.json(first_stats if isinstance(first_stats, dict) else str(first_stats))

            st.markdown("**Métriques extraites (première ligne):**")
            st.json(metrics_df.iloc[0].to_dict())

            st.markdown("**Stats détaillées (5 premières configs avec meilleur Sharpe):**")
            debug_cols = ['sharpe', 'pnl', 'pnl_pct', 'max_drawdown', 'total_trades']
            available_debug_cols = [c for c in debug_cols if c in metrics_df.columns]
            if available_debug_cols:
                st.dataframe(metrics_df[available_debug_cols].nlargest(5, 'sharpe'))

    # Normaliser les noms de colonnes
    if "sharpe" in results_df.columns and "sharpe_ratio" not in results_df.columns:
        results_df["sharpe_ratio"] = results_df["sharpe"]

    if "pnl_pct" in results_df.columns and "total_return" not in results_df.columns:
        results_df["total_return"] = results_df["pnl_pct"]

    # Vérification finale
    if "sharpe_ratio" not in results_df.columns:
        st.error(f"❌ Impossible d'extraire 'sharpe_ratio'. Colonnes finales: {list(results_df.columns)}")
        if len(results_df) > 0:
            st.caption("Exemple de ligne:")
            st.json(results_df.iloc[0].to_dict())
        raise Exception("Colonne 'sharpe_ratio' manquante après normalisation")

    # Convertir DataFrame en liste de dicts pour compatibilité avec le reste du code
    results = results_df.to_dict('records')

    return results


def test_proposals(strategy_name: str, proposals: list, baseline_config: dict, use_gpu: bool):
    """Teste chaque proposition et retourne les résultats."""

    # Utiliser les mêmes données que le sweep (depuis session_state)
    if "data" in st.session_state and not st.session_state.data.empty:
        df_market = st.session_state.data
        st.caption(f"📊 Test des propositions sur {len(df_market)} barres réelles")
    else:
        st.error("❌ Aucune donnée disponible en cache. Le sweep aurait dû charger les données.")
        return []

    # Utiliser run_backtest_gpu au lieu de BacktestEngine
    test_results = []

    for prop in proposals:
        try:
            st.caption(f"🧪 Test de '{prop['name']}' avec params: {prop['params']}")

            # CORRECTION: Forcer conversion int pour paramètres de période
            cleaned_params = _force_integer_params(prop["params"])

            result = run_backtest_gpu(
                df=df_market,
                strategy=strategy_name,
                params=cleaned_params,
            )

            test_results.append({
                "name": prop["name"],
                "params": prop["params"],
                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
                "total_return": result.metrics.get("total_return", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                "win_rate": result.metrics.get("win_rate", 0.0),
                "trades": result.trades if hasattr(result, 'trades') else [],  # ← Capture trades
                "full_result": result,  # ← Capture résultat complet pour analyses futures
            })

            st.success(f"✅ '{prop['name']}' testé : Sharpe={result.metrics.get('sharpe_ratio', 0.0):.3f}")

        except Exception as e:
            st.error(f"❌ Erreur test '{prop['name']}': {str(e)}")
            st.caption(f"Paramètres reçus: {prop['params']}")
            continue

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
            st.info("💡 Recommandations détectées dans la réponse brute - voir logs pour détails")
            with st.expander("📝 Réponse LLM brute"):
                st.text(raw_response[:2000])


def display_strategist_results(proposals: dict, baseline: dict, baseline_sharpe: float, elapsed: float):
    """Affiche les propositions du Strategist."""

    st.markdown(f"*Temps de génération: {elapsed:.1f}s*")
    st.markdown(f"*Propositions valides: {proposals['total_valid']}/{proposals['total_generated']}*")

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


def display_final_report(baseline: dict, proposals: list, test_results: list, analysis: dict):
    """Affiche le rapport final avec visualisations."""

    st.markdown("## 🏆 Rapport Final")

    # Vérifier si des tests ont réussi
    if len(test_results) == 0:
        st.warning("⚠️ Aucune alternative n'a pu être testée - seule la baseline est disponible.")
        st.info("🔍 Causes possibles : paramètres incompatibles, erreurs dans run_backtest_gpu(), ou stratégie non supportée.")
        return

    # Préparer données pour graphiques
    comparison_data = [{
        "Config": "BASELINE",
        "Sharpe": baseline.get("sharpe_ratio", 0.0),
        "Return": baseline.get("total_return", 0.0),
        "Drawdown": abs(baseline.get("max_drawdown", 0.0)),
    }]

    for res in test_results:
        comparison_data.append({
            "Config": res["name"],
            "Sharpe": res["sharpe_ratio"],
            "Return": res["total_return"],
            "Drawdown": abs(res["max_drawdown"]),
        })

    df_comparison = pd.DataFrame(comparison_data)

    # Graphiques comparatifs
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Sharpe Ratio", "Total Return", "Max Drawdown"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
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
        row=1, col=1
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
        row=1, col=2
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
        row=1, col=3
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
        improvement = ((best_config["Sharpe"] - baseline["sharpe_ratio"]) / abs(baseline["sharpe_ratio"])) * 100

        st.success(f"🎉 **{best_config['Config']}** améliore le Sharpe de **{improvement:+.1f}%** !")

        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe Ratio", f"{best_config['Sharpe']:.3f}",
                   f"{best_config['Sharpe'] - baseline['sharpe_ratio']:+.3f}")
        col2.metric("Return", f"{best_config['Return']:.2%}",
                   f"{best_config['Return'] - baseline['total_return']:+.2%}")
        col3.metric("Drawdown", f"{best_config['Drawdown']:.2%}")

        st.markdown("**Paramètres:**")
        st.json(best_proposal["params"])
        
        # 📊 VISUALISATION CANDLESTICK + TRADES (meilleure config)
        st.markdown("---")
        with st.expander("📈 Visualisation Graphique (Bougies + Trades)", expanded=True):
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
                        key_suffix=f"best_{best_idx}"
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
            st.info("⚠️ Trades de baseline non enregistrés (seules les propositions sont testées)")
        
        # Afficher chaque proposition
        for i, res in enumerate(test_results, 1):
            config_trades = res.get("trades", [])
            config_name = res["name"]
            sharpe = res["sharpe_ratio"]
            
            with st.expander(f"📊 Proposition {i}: {config_name} (Sharpe: {sharpe:.3f})", expanded=False):
                if not config_trades:
                    st.info("ℹ️ Aucun trade enregistré")
                else:
                    render_candlestick_with_trades(
                        df_ohlcv=df_ohlcv,
                        trades=config_trades,
                        title=f"{config_name} - {len(config_trades)} trades",
                        key_suffix=f"prop_{i}"
                    )
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger les données OHLCV: {e}")


def render_candlestick_with_trades(
    df_ohlcv: pd.DataFrame,
    trades: List[Dict[str, Any]],
    title: str = "📈 Graphique Prix + Trades",
    key_suffix: str = "default"
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
        st.warning(f"⚠️ Colonnes manquantes. Requis: {required_cols}, Disponible: {list(df_ohlcv.columns)}")
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
    
    for trade in (trades or []):
        side = str(trade.get("side", "LONG")).upper()
        # FIX: Support both "pnl" (BacktestEngine) and "pnl_realized" (Strategy classes)
        pnl = trade.get("pnl", trade.get("pnl_realized", 0))
        
        # Entrées (différencier LONG/SHORT)
        if "entry_time" in trade and "entry_price" in trade:
            if side == "LONG":
                entries_long_x.append(trade["entry_time"])
                entries_long_y.append(trade["entry_price"])
            else:
                entries_short_x.append(trade["entry_time"])
                entries_short_y.append(trade["entry_price"])
        
        # Sorties (différencier profit/perte)
        if "exit_time" in trade and "exit_price" in trade:
            if pnl > 0:
                exits_profit_x.append(trade["exit_time"])
                exits_profit_y.append(trade["exit_price"])
            else:
                exits_loss_x.append(trade["exit_time"])
                exits_loss_y.append(trade["exit_price"])
    
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
                    line=dict(width=1, color="white")
                ),
                hovertemplate="<b>Entrée LONG</b><br>Prix: %{y:.2f}<br>%{x}<extra></extra>"
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
                    line=dict(width=1, color="white")
                ),
                hovertemplate="<b>Entrée SHORT</b><br>Prix: %{y:.2f}<br>%{x}<extra></extra>"
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
                    line=dict(width=1, color="white")
                ),
                hovertemplate="<b>Sortie Profit</b><br>Prix: %{y:.2f}<br>%{x}<extra></extra>"
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
                    line=dict(width=1, color="white")
                ),
                hovertemplate="<b>Sortie Perte</b><br>Prix: %{y:.2f}<br>%{x}<extra></extra>"
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
            type="date"
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.2)"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        font=dict(size=11)
    )
    
    # Affichage
    st.plotly_chart(fig, use_container_width=True, key=f"candlestick_{key_suffix}")
    
    # Stats rapides
    if trades:
        col1, col2, col3, col4 = st.columns(4)
        total_trades = len(trades)
        # FIX: Support both "pnl" (BacktestEngine) and "pnl_realized" (Strategy classes)
        profit_trades = sum(1 for t in trades if t.get("pnl", t.get("pnl_realized", 0)) > 0)
        loss_trades = total_trades - profit_trades
        win_rate = (profit_trades / total_trades * 100) if total_trades > 0 else 0
        
        col1.metric("📊 Total Trades", total_trades)
        col2.metric("✅ Profits", profit_trades)
        col3.metric("❌ Pertes", loss_trades)
        col4.metric("🎯 Win Rate", f"{win_rate:.1f}%")


if __name__ == "__main__":
    render_page()
