"""
Page Streamlit: Optimisation Multi-LLM
======================================

Interface pour le système collaboratif d'agents LLM (Analyst + Strategist).
Workflow: Sweep → Analyse → Propositions → Tests → Visualisation
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import sys

# Ajouter src au path si nécessaire
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.strategist import Strategist
from threadx.ui.backtest_bridge import run_backtest_gpu
from threadx.ui.strategy_registry import parameter_specs_for


def render_page():
    """Affiche la page d'optimisation Multi-LLM."""
    
    st.title("🤖 Optimisation Multi-LLM")
    st.markdown("""
    **Système collaboratif d'agents LLM** pour optimisation automatique de stratégies.
    
    **Workflow**:
    1. 🔄 Sweep GPU → Test multiple configurations
    2. 🧠 Analyst (deepseek-r1:70b) → Analyse quantitative & patterns
    3. 🎨 Strategist (gpt-oss:20b) → Propositions créatives
    4. ✅ Tests automatiques → Validation performances
    5. 📊 Visualisation → Comparaison résultats
    """)
    
    # Vérifier prérequis
    with st.expander("⚙️ Prérequis & Configuration", expanded=False):
        check_prerequisites()
    
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
        
        # Préprogrammer valeurs spécifiques pour MA_Crossover
        ma_crossover_presets = {
            "max_hold_bars": {"min": 300, "max": 300, "n_values": 1},  # Fixé à 20 (via override)
            "risk_per_trade": {"min": 0.02, "max": 0.02, "n_values": 1}  # Fixé à 0.005 (via override)
        }
        
        st.markdown("**Paramètres du sweep:**")
        sweep_params = {}
        
        for param_name, spec in param_specs.items():
            param_type = spec.get("type", "number")
            
            if param_type == "boolean":
                sweep_params[param_name] = [False, True]
                st.caption(f"✓ {param_name}: [False, True]")
            else:
                # Utiliser presets si disponible pour MA_Crossover
                if strategy_name == "MA_Crossover" and param_name in ma_crossover_presets:
                    preset = ma_crossover_presets[param_name]
                    min_val = preset["min"]
                    max_val = preset["max"]
                    n_values = preset["n_values"]
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
            
            # Générer valeurs avec protection division par zéro
            if n_values == 1:
                values = [min_val]
            else:
                values = [min_val + i * (max_val - min_val) / (n_values - 1)
                         for i in range(n_values)]

            if param_type == "integer":
                values = [int(v) for v in values]

            sweep_params[param_name] = values
            st.caption(f"✓ {param_name}: {values}")
        
        total_configs = 1
        for vals in sweep_params.values():
            total_configs *= len(vals)
        
        st.info(f"**Total configurations**: {total_configs}")
    
    with col2:
        st.subheader("🤖 Configuration LLM")
        
        analyst_model = st.selectbox(
            "Modèle Analyst",
            options=["deepseek-r1:70b", "gemma3:27b", "qwen3-vl:30b"],
            help="Modèle pour analyse quantitative"
        )
        
        strategist_model = st.selectbox(
            "Modèle Strategist",
            options=["gpt-oss:20b", "gpt-oss:120b-cloud", "gemma3:27b"],
            help="Modèle pour propositions créatives"
        )
        
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
        
        use_gpu = st.checkbox("Utiliser GPU", value=True)
        
        # Checkbox analyse IA (cochée par défaut)
        enable_ai_analysis = st.checkbox(
            "⚡ Activer l'analyse IA pour la meilleure configuration",
            value=True,
            help="Les LLM analyseront les résultats pour proposer des optimisations"
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
            n_proposals=n_proposals,
            top_n_analysis=top_n_analysis,
            use_gpu=use_gpu,
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


def run_multi_llm_optimization(
    strategy_name: str,
    sweep_params: dict,
    analyst_model: str,
    strategist_model: str,
    n_proposals: int,
    top_n_analysis: int,
    use_gpu: bool,
):
    """Exécute le workflow complet d'optimisation Multi-LLM."""
    
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
            sweep_results = execute_sweep(strategy_name, sweep_params, use_gpu)
        
        st.session_state.llm_results["sweep"] = sweep_results
        
        # Afficher résultats sweep
        with results_container:
            st.success(f"✅ Sweep terminé: {len(sweep_results)} configs testées")
            
            with st.expander("📊 Top 10 configurations", expanded=True):
                df_sweep = pd.DataFrame(sweep_results)
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
        status_text.markdown(f"### 🧠 Étape 2/5: Analyse Analyst ({analyst_model})...")
        
        with st.spinner(f"Analyse des top {top_n_analysis} configs (peut prendre 30-60s)..."):
            analyst = Analyst(model=analyst_model, debug=False)
            
            # Créer zone de streaming pour l'analyse
            analysis_container = st.container()
            
            with analysis_container:
                st.markdown("#### 🔍 Réflexions de l'Analyst")
                
                with st.chat_message("assistant", avatar="🧠"):
                    st.caption(f"Modèle: {analyst_model}")
                    
                    # Placeholder pour streaming
                    analysis_placeholder = st.empty()
                    analysis_placeholder.info("⏳ Analyse en cours...")
                    
                    # Exécuter analyse
                    start_time = time.time()
                    analysis_result = analyst.analyze_sweep_results(
                        sweep_df=df_sweep,
                        top_n=top_n_analysis
                    )
                    elapsed = time.time() - start_time
                    
                    # Afficher résultats formatés
                    analysis_placeholder.empty()
                    display_analyst_results(analysis_result, elapsed)
        
        st.session_state.llm_results["analysis"] = analysis_result
        progress_bar.progress(50)
        
        # ============================================================
        # ÉTAPE 3: PROPOSITIONS STRATEGIST
        # ============================================================
        status_text.markdown(f"### 🎨 Étape 3/5: Propositions Strategist ({strategist_model})...")
        
        # Récupérer baseline (meilleure config actuelle)
        baseline_config = df_sweep.nlargest(1, "sharpe_ratio").iloc[0].to_dict()
        baseline_params = {k: v for k, v in baseline_config.items() 
                          if k in sweep_params.keys()}
        
        with st.spinner(f"Génération de {n_proposals} propositions créatives (peut prendre 20-40s)..."):
            strategist = Strategist(model=strategist_model, debug=False)
            
            # Créer zone de streaming pour propositions
            proposals_container = st.container()
            
            with proposals_container:
                st.markdown("#### 💡 Propositions du Strategist")
                
                with st.chat_message("assistant", avatar="🎨"):
                    st.caption(f"Modèle: {strategist_model}")
                    
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
        
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        import traceback
        st.code(traceback.format_exc())


def _generate_combinations(sweep_params: dict):
    """Génère toutes les combinaisons de paramètres."""
    from itertools import product
    
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    for combo in product(*param_values):
        yield dict(zip(param_names, combo))


def execute_sweep(strategy_name: str, sweep_params: dict, use_gpu: bool):
    """Exécute le sweep et retourne les résultats."""
    
    # Charger données (simulées pour démo - à remplacer par vraies données)
    import numpy as np
    
    n_candles = 8760
    dates = pd.date_range("2024-01-01", periods=n_candles, freq="1h")
    close_prices = 40000 + np.cumsum(np.random.randn(n_candles) * 100)
    
    df_market = pd.DataFrame({
        "timestamp": dates,
        "open": close_prices + np.random.randn(n_candles) * 50,
        "high": close_prices + abs(np.random.randn(n_candles) * 100),
        "low": close_prices - abs(np.random.randn(n_candles) * 100),
        "close": close_prices,
        "volume": np.random.randint(100, 1000, n_candles),
    })
    
    # Exécuter tous les backtests avec run_backtest_gpu
    results = []
    
    for params in _generate_combinations(sweep_params):
        try:
            result = run_backtest_gpu(
                df=df_market,
                strategy=strategy_name,
                params=params,
            )
            
            results.append({
                **params,
                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
                "total_return": result.metrics.get("total_return", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                "win_rate": result.metrics.get("win_rate", 0.0),
                "total_trades": len(result.trades),
            })
        except Exception:
            continue
    
    return results


def test_proposals(strategy_name: str, proposals: list, baseline_config: dict, use_gpu: bool):
    """Teste chaque proposition et retourne les résultats."""
    
    # Recharger données (mêmes que sweep)
    import numpy as np
    
    np.random.seed(42)  # Même seed pour comparaison
    n_candles = 8760
    dates = pd.date_range("2024-01-01", periods=n_candles, freq="1h")
    close_prices = 40000 + np.cumsum(np.random.randn(n_candles) * 100)
    
    df_market = pd.DataFrame({
        "timestamp": dates,
        "open": close_prices + np.random.randn(n_candles) * 50,
        "high": close_prices + abs(np.random.randn(n_candles) * 100),
        "low": close_prices - abs(np.random.randn(n_candles) * 100),
        "close": close_prices,
        "volume": np.random.randint(100, 1000, n_candles),
    })
    
    # Utiliser run_backtest_gpu au lieu de BacktestEngine
    test_results = []
    
    for prop in proposals:
        try:
            result = run_backtest_gpu(
                df=df_market,
                strategy=strategy_name,
                params=prop["params"],
            )
            
            test_results.append({
                "name": prop["name"],
                "params": prop["params"],
                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
                "total_return": result.metrics.get("total_return", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                "win_rate": result.metrics.get("win_rate", 0.0),
            })
        except Exception:
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
    cols = st.columns(len(analysis["analysis"]["key_metrics"]))
    for col, (metric, value) in zip(cols, analysis["analysis"]["key_metrics"].items()):
        col.metric(metric.replace("_", " ").title(), f"{value:.3f}")
    
    # Trade-offs
    st.markdown("\n**⚖️ Trade-offs observés:**")
    for i, tradeoff in enumerate(analysis["analysis"]["trade_offs"], 1):
        st.markdown(f"{i}. {tradeoff}")
    
    # Recommandations
    st.markdown("\n**💡 Recommandations:**")
    for i, rec in enumerate(analysis["analysis"]["recommendations"], 1):
        st.success(f"**Rec {i}:** {rec}")


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
    
    # Préparer données pour graphiques
    comparison_data = [{
        "Config": "BASELINE",
        "Sharpe": baseline["sharpe_ratio"],
        "Return": baseline["total_return"],
        "Drawdown": abs(baseline["max_drawdown"]),
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
    
    # Tableau comparatif
    st.markdown("### 📋 Tableau Comparatif")
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_page()
