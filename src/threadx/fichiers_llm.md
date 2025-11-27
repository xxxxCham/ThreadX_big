## Sommaire

1. `streamlit_app.py`
2. `llm\client.py`
3. `llm\interpreters.py`
4. `llm\model_router.py`
5. `llm\ollama_manager.py`
6. `llm\prompts.py`
7. `llm\run_report.py`
8. `llm\__init__.py`
9. `llm\agents\analyst.py`
10. `llm\agents\base_agent.py`
11. `llm\agents\codewriter.py`
12. `llm\agents\critic.py`
13. `llm\agents\strategist.py`
14. `llm\agents\__init__.py`
15. `strategy\ma_crossover.py`
16. `ui\page_llm_optimizer.py`


<!-- MODULE-START: streamlit_app.py -->
```json
{
  "name": "streamlit_app.py",
  "path": "streamlit_app.py",
  "ext": ".py",
  "anchor": "streamlit_app_py"
}
```
## streamlit_app_py
*Chemin* : `streamlit_app.py`  
*Type* : `.py`  

```python
"""
ThreadX v2.0 - Interface Streamlit Moderne
===========================================

Application de trading quantitatif avec interface fusionnée et moderne.

Architecture:
- Page 1: Configuration & Stratégie (fusion anciennes pages 1+2)
- Page 2: Backtest & Optimisation (fusion anciennes pages 3+4)

Author: ThreadX Framework
Version: 2.0.0 - UI Redesign
"""

from __future__ import annotations
import threadx_gpu_init  # ⚡ CRITICAL: Force RTX 5080 as default GPU

import gc
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import streamlit as st

# Optionally silence all logs early if requested (for performance profiling)
if os.getenv("THREADX_SILENCE_LOGS", "0") == "1":
    logging.disable(logging.CRITICAL)

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threadx.data_access import DATA_DIR
from threadx.ui.page_backtest_optimization import main as backtest_page_main
from threadx.ui.page_config_strategy import main as config_page_main
from threadx.ui.page_llm_optimizer import render_page as llm_optimizer_page
from threadx.ui.page_reports import render_page as reports_page
from threadx.ui.system_monitor import get_global_monitor
import subprocess
import platform

# Configuration
st.set_page_config(
    page_title="ThreadX v2.0 - Trading Quantitatif",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles CSS Modernes + Script JS pour molette sur sliders
st.markdown(
    """
<style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%);
    }
    h1 {
        color: #4fc3f7 !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        text-shadow: 0 0 20px rgba(79, 195, 247, 0.3);
    }
    h2 {
        color: #81c784 !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    h3 { color: #a8b2d1 !important; font-weight: 500 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #4fc3f7 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a8b2d1 !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%) !important;
        border-right: 1px solid rgba(79, 195, 247, 0.1) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a8b2d1;
        padding: 12px 24px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    hr { margin: 2rem 0 !important; border-color: rgba(79, 195, 247, 0.2) !important; }
    /* Curseur personnalisé pour les sliders */
    .stSlider:hover { cursor: ew-resize !important; }
</style>

<script>
// ===================================
// ThreadX - Contrôle Molette Sliders
// ===================================
// Active l'ajustement des sliders avec la molette de la souris
// Fonctionne sur tous les sliders de l'application (partout)

(function() {
    'use strict';

    // Configuration
    const WHEEL_SENSITIVITY = 0.1; // Sensibilité de la molette (10% par cran)
    const UPDATE_DELAY = 10; // Délai anti-rebond en ms

    let updateTimeout = null;
    let processedSliders = new WeakSet();

    /**
     * Calcule le nouveau pas d'un slider en fonction de son range
     */
    function calculateStep(slider) {
        const min = parseFloat(slider.min) || 0;
        const max = parseFloat(slider.max) || 100;
        let step = parseFloat(slider.step) || 1;
        const range = max - min;

        // Si step déjà défini et cohérent, l'utiliser
        if (step > 0 && step <= range) {
            return step;
        }

        // Sinon, calculer un pas intelligent basé sur le range
        if (range <= 1) {
            return 0.01; // Valeurs décimales fines
        } else if (range <= 10) {
            return 0.1;
        } else if (range <= 100) {
            return 1;
        } else if (range <= 1000) {
            return 10;
        } else {
            return Math.max(1, range / 100);
        }
    }

    /**
     * Ajoute le contrôle molette à un slider
     */
    function addWheelControl(slider) {
        // Éviter de traiter plusieurs fois le même slider
        if (processedSliders.has(slider)) {
            return;
        }

        processedSliders.add(slider);

        // Récupérer les bornes
        const min = parseFloat(slider.min) || 0;
        const max = parseFloat(slider.max) || 100;
        const step = calculateStep(slider);

        // Trouver le conteneur parent du slider
        const sliderContainer = slider.closest('[data-testid="stSlider"]') ||
                               slider.closest('.stSlider') ||
                               slider.parentElement;

        if (!sliderContainer) {
            console.warn('[ThreadX] Conteneur slider non trouvé');
            return;
        }

        // Fonction de mise à jour avec gestion d'erreurs
        function updateSlider(event) {
            try {
                event.preventDefault();
                event.stopPropagation();

                const currentValue = parseFloat(slider.value) || 0;
                const delta = -Math.sign(event.deltaY);
                const increment = delta * step;

                let newValue = currentValue + increment;

                // Clamper entre min et max
                newValue = Math.max(min, Math.min(max, newValue));

                // Arrondir selon le step
                newValue = Math.round(newValue / step) * step;

                // Limiter la précision pour éviter les erreurs d'arrondi
                const decimals = Math.max(0, (step.toString().split('.')[1] || '').length);
                newValue = parseFloat(newValue.toFixed(decimals));

                // Mettre à jour la valeur
                if (Math.abs(newValue - currentValue) >= step * 0.01) {
                    slider.value = newValue;

                    // Déclencher les événements pour que Streamlit détecte le changement
                    const inputEvent = new Event('input', { bubbles: true });
                    const changeEvent = new Event('change', { bubbles: true });

                    slider.dispatchEvent(inputEvent);
                    setTimeout(() => slider.dispatchEvent(changeEvent), 5);

                    // Visual feedback
                    sliderContainer.style.transition = 'transform 0.1s ease';
                    sliderContainer.style.transform = 'scale(1.02)';
                    setTimeout(() => {
                        sliderContainer.style.transform = 'scale(1)';
                    }, 100);

                    console.log(`[ThreadX] Slider mis à jour: ${currentValue} -> ${newValue}`);
                }
            } catch (error) {
                console.error('[ThreadX] Erreur lors de la mise à jour du slider:', error);
            }
        }

        // Ajouter l'event listener directement sur le slider
        slider.addEventListener('wheel', function(event) {
            clearTimeout(updateTimeout);
            updateTimeout = setTimeout(() => {
                updateSlider(event);
            }, UPDATE_DELAY);
        }, { passive: false });

        // Ajouter aussi sur le conteneur pour une meilleure détection
        sliderContainer.addEventListener('wheel', function(event) {
            // Vérifier si la souris est bien sur le slider
            const rect = sliderContainer.getBoundingClientRect();
            const mouseX = event.clientX;
            const mouseY = event.clientY;

            if (mouseX >= rect.left && mouseX <= rect.right &&
                mouseY >= rect.top && mouseY <= rect.bottom) {

                clearTimeout(updateTimeout);
                updateTimeout = setTimeout(() => {
                    updateSlider(event);
                }, UPDATE_DELAY);
            }
        }, { passive: false });

        // Changer le curseur au survol
        sliderContainer.addEventListener('mouseenter', () => {
            sliderContainer.style.cursor = 'ew-resize';
        });

        sliderContainer.addEventListener('mouseleave', () => {
            sliderContainer.style.cursor = 'default';
        });

        console.log(`[ThreadX] Slider wheel control activé: min=${min}, max=${max}, step=${step}`);
    }

    /**
     * Scanne et active tous les sliders de la page
     */
    function activateAllSliders() {
        // Sélecteurs multiples pour couvrir tous les types de sliders Streamlit
        const selectors = [
            'input[type="range"]',
            '[data-baseweb="slider"] input[type="range"]',
            '.stSlider input[type="range"]',
            '[data-testid="stSlider"] input[type="range"]',
            '[class*="slider"] input[type="range"]'
        ];

        let activatedCount = 0;

        selectors.forEach(selector => {
            try {
                const sliders = document.querySelectorAll(selector);
                sliders.forEach(slider => {
                    if (slider && slider.type === 'range') {
                        addWheelControl(slider);
                        activatedCount++;
                    }
                });
            } catch (error) {
                console.error(`[ThreadX] Erreur avec le sélecteur ${selector}:`, error);
            }
        });

        if (activatedCount > 0) {
            console.log(`[ThreadX] ${activatedCount} slider(s) activé(s)`);
        }
    }

    /**
     * Observer pour détecter les nouveaux sliders ajoutés dynamiquement
     */
    function setupMutationObserver() {
        try {
            const observer = new MutationObserver(function(mutations) {
                let shouldReactivate = false;

                mutations.forEach(function(mutation) {
                    if (mutation.addedNodes) {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.nodeType === 1) { // Element node
                                // Vérifier si c'est un slider ou contient des sliders
                                if (node.matches && (
                                    node.matches('input[type="range"]') ||
                                    node.matches('[data-testid="stSlider"]') ||
                                    node.matches('.stSlider')
                                )) {
                                    shouldReactivate = true;
                                } else if (node.querySelector) {
                                    try {
                                        const hasSlider = node.querySelector('input[type="range"]');
                                        if (hasSlider) {
                                            shouldReactivate = true;
                                        }
                                    } catch (e) {
                                        // Ignore querySelector errors
                                    }
                                }
                            }
                        });
                    }
                });

                if (shouldReactivate) {
                    setTimeout(activateAllSliders, 200);
                }
            });

            observer.observe(document.body, {
                childList: true,
                subtree: true,
                attributes: false,
                attributeOldValue: false,
                characterData: false,
                characterDataOldValue: false
            });

            console.log('[ThreadX] MutationObserver activé pour sliders dynamiques');
            return observer;
        } catch (error) {
            console.error('[ThreadX] Erreur lors de la création du MutationObserver:', error);
            return null;
        }
    }

    /**
     * Initialisation au chargement
     */
    function init() {
        console.log('[ThreadX] Initialisation contrôle molette sliders...');

        try {
            // Première activation
            activateAllSliders();

            // Observer pour nouveaux sliders
            setupMutationObserver();

            // Re-scanner périodiquement (fallback)
            setInterval(activateAllSliders, 3000);

            console.log('[ThreadX] ✅ Contrôle molette sliders activé globalement');
        } catch (error) {
            console.error('[ThreadX] Erreur lors de l\'initialisation:', error);
        }
    }

    // Démarrer quand le DOM est prêt
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // DOM déjà chargé, initialiser immédiatement
        setTimeout(init, 100);
    }

    // Re-scanner après les transitions Streamlit
    window.addEventListener('load', function() {
        setTimeout(activateAllSliders, 1000);
    });

    // Observer les changements de hash/URL pour Streamlit
    window.addEventListener('hashchange', function() {
        setTimeout(activateAllSliders, 500);
    });

    // Observer les événements Streamlit spécifiques
    document.addEventListener('streamlit:render', function() {
        setTimeout(activateAllSliders, 300);
    });

})();
</script>
""",
    unsafe_allow_html=True,
)

PAGE_TITLES = {
    "config": "📊 Chargement des Données",
    "backtest": "⚡ Optimisation",
    "llm": "🤖 Multi-LLM Optimizer",
    "reports": "📚 Historique Rapports",
    "monitor": "🖥️ Monitoring Système",
}


def render_monitor_page() -> None:
    """Page dédiée au monitoring temps réel CPU/RAM/GPU."""
    st.markdown("# 🖥️ Monitoring Système Temps Réel")
    st.caption(
        "Affiche l'utilisation CPU, mémoire et GPU pendant les backtests/optimisations."
    )

    monitor = get_global_monitor()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        enabled = st.toggle(
            "Activer le monitoring",
            value=st.session_state.get("monitor_enabled", True),
            key="monitor_enabled",
        )
    with col2:
        auto_refresh = st.toggle(
            "Auto-refresh",
            value=st.session_state.get("monitor_autorefresh", True),
            key="monitor_autorefresh",
        )
    with col3:
        refresh_secs = st.slider(
            "Intervalle (s)",
            0.25,
            5.0,
            st.session_state.get("monitor_interval", 0.5),
            0.25,
            key="monitor_interval",
        )

    # Start/Stop en fonction de l'état
    if enabled and not monitor.is_running():
        monitor.start()
    elif not enabled and monitor.is_running():
        monitor.stop()

    # Actions
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🧹 Vider l'historique", use_container_width=True):
            monitor.clear_history()
            st.toast("Historique vidé", icon="🧹")
    with c2:
        st.write("")

    # Données et graphiques
    df = monitor.get_history_df()
    if df.empty:
        st.info("Aucune donnée pour l'instant. Activez le monitoring.")
    else:
        # Mise en forme
        df_time = df.set_index("time")
        st.markdown("### CPU & Mémoire")
        st.line_chart(df_time[["cpu", "memory"]])

        st.markdown("### GPU Utilisation (%)")
        if (df[["gpu1", "gpu2"]].max() > 0).any():
            st.line_chart(df_time[["gpu1", "gpu2"]])
        else:
            st.caption("GPU inactif ou non détecté (pynvml non disponible)")

        st.markdown("### GPU Mémoire (%)")
        st.line_chart(df_time[["gpu1_mem", "gpu2_mem"]])

        # Statistiques
        with st.expander("Résumé Statistiques", expanded=False):
            stats = monitor.get_stats_summary()
            if stats:
                cols = st.columns(4)
                items = list(stats.items())
                for i in range(0, len(items), 4):
                    row = items[i : i + 4]
                    for (k, v), col in zip(row, cols):
                        with col:
                            st.metric(k, f"{v:.2f}" if isinstance(v, float) else v)
            else:
                st.write("Pas de statistiques disponibles.")

    # Auto-refresh non bloquant: on relance le script après une petite pause
    if enabled and auto_refresh:
        time.sleep(float(refresh_secs))
        st.rerun()


PAGE_RENDERERS = {
    "config": config_page_main,
    "backtest": backtest_page_main,
    "llm": llm_optimizer_page,
    "reports": reports_page,
    "monitor": render_monitor_page,
}


def init_session() -> None:
    """
    Initialise la session avec les réglages par défaut.
    Force l'application des paramètres BTC préréglés UNIQUEMENT à la première ouverture.
    Les modifications de l'utilisateur sont conservées entre les pages.
    """
    # Vérifier si c'est la première initialisation
    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = False

    defaults = {
        "page": "config",
        "symbol": "BTCUSDC",  # Bitcoin préréglé - OBLIGATOIRE
        "timeframe": "15m",  # 15 minutes préréglé - OBLIGATOIRE
        "start_date": date(2024, 12, 1),  # 1er décembre 2024 - OBLIGATOIRE
        "end_date": date(2025, 1, 31),  # 31 janvier 2025 - OBLIGATOIRE
        "strategy": "Bollinger_Breakout",  # Stratégie Bollinger+ATR préréglée
        "indicators": {},
        # Paramètres de stratégie préréglés selon le tableau classique
        "strategy_params": {
            "bb_period": 20,  # Milieu de la plage 10→50
            "bb_std": 2.0,  # Milieu de la plage 1.5→3.0
            "entry_z": 1.0,  # Seuil Z-score standard
            "entry_logic": "AND",  # Logique d'entrée standard
            "atr_period": 14,  # Milieu de la plage 7→21 (classique)
            "atr_multiplier": 1.5,  # Milieu de la plage 1.0→3.0
            "trailing_stop": True,  # Activer trailing stop
            "risk_per_trade": 0.02,  # 2% de risque par trade (préréglé)
            "min_pnl_pct": 0.01,  # Filtre minimum 0.01%
            "leverage": 1.0,  # Sans levier
            "max_hold_bars": 72,  # 3 jours en 1h (72 barres de 1h)
            "spacing_bars": 6,  # 6 barres minimum entre trades
            "trend_period": 0,  # Sans filtre tendance EMA
        },
        "data": None,
        "backtest_results": None,
        "sweep_results": None,
        "data_dir": str(DATA_DIR),
    }

    # Initialiser les clés manquantes
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # FORCER les paramètres par défaut UNIQUEMENT lors de la première initialisation
    # Après, les modifications utilisateur sont conservées
    if not st.session_state.session_initialized:
        st.session_state.symbol = "BTCUSDC"
        st.session_state.timeframe = "15m"
        st.session_state.start_date = date(2024, 12, 1)
        st.session_state.end_date = date(2025, 1, 31)

        # FORCER le risque par trade à 2% (0.02) - ne jamais le laisser à 0.01
        if "strategy_params" in st.session_state:
            st.session_state.strategy_params["risk_per_trade"] = 0.02

        # Marquer comme initialisé pour ne plus forcer les valeurs
        st.session_state.session_initialized = True


def reset_ollama() -> tuple[bool, str]:
    """
    Arrête et redémarre Ollama pour éviter les blocages.

    Returns:
        tuple[bool, str]: (succès, message)
    """
    try:
        is_windows = platform.system() == "Windows"

        # Étape 1: Arrêter Ollama
        if is_windows:
            # Utiliser PowerShell pour arrêter proprement
            stop_cmd = ["powershell", "-Command", "Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue"]
        else:
            stop_cmd = ["pkill", "-9", "ollama"]

        try:
            subprocess.run(stop_cmd, capture_output=True, timeout=5)
            time.sleep(1)  # Laisser le temps au processus de se terminer
        except Exception:
            pass  # Si aucun processus Ollama n'est en cours, c'est OK

        # Étape 2: Vérifier que Ollama est bien arrêté
        if is_windows:
            check_cmd = ["tasklist", "/FI", "IMAGENAME eq ollama.exe"]
            try:
                # Fix encoding Windows : utiliser errors='ignore' pour éviter UnicodeDecodeError
                result = subprocess.run(
                    check_cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',  # Ignore les caractères non-UTF8
                    timeout=3
                )
                # Fix NoneType : vérifier que stdout existe avant le 'in'
                if result.stdout and "ollama.exe" in result.stdout:
                    return False, "❌ Impossible d'arrêter Ollama (processus toujours actif)"
            except Exception as e:
                # Si la vérification échoue, on continue quand même
                pass

        # Étape 3: Redémarrer Ollama en arrière-plan
        if is_windows:
            # Démarrer Ollama en arrière-plan avec Start-Process
            start_cmd = ["powershell", "-Command", "Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden"]
            # Windows : CREATE_NEW_CONSOLE pour éviter les problèmes d'encoding
            subprocess.Popen(
                start_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_CONSOLE if is_windows else 0
            )
        else:
            start_cmd = ["ollama", "serve"]
            subprocess.Popen(
                start_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

        time.sleep(2)  # Laisser le temps à Ollama de démarrer

        return True, "✅ Ollama réinitialisé avec succès"

    except subprocess.TimeoutExpired:
        return False, "⏱️ Timeout lors de la réinitialisation d'Ollama"
    except FileNotFoundError:
        return False, "❌ Ollama non trouvé (vérifiez l'installation)"
    except Exception as e:
        return False, f"❌ Erreur: {str(e)}"


def clean_all_memory() -> tuple[bool, str]:
    """
    Nettoyage complet de toute la mémoire (RAM système, VRAM GPU, caches).

    Returns:
        tuple[bool, str]: (succès, message détaillé)
    """
    messages = []
    success = True

    try:
        # 1. Nettoyer la VRAM GPU (NVIDIA)
        try:
            import cupy as cp
            # Vider tous les memory pools CuPy
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

            freed_vram = mempool.used_bytes()
            freed_pinned = pinned_mempool.n_free_blocks()

            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

            messages.append(f"✅ VRAM GPU vidée: {freed_vram / (1024**3):.2f} GB libérés")
        except ImportError:
            messages.append("⚠️ CuPy non disponible (VRAM GPU non vidée)")
        except Exception as e:
            messages.append(f"⚠️ Erreur vidage VRAM: {str(e)}")
            success = False

        # 2. Vider le cache IndicatorBank (si disponible)
        try:
            from threadx.indicators.bank import IndicatorBank
            # Réinitialiser singleton IndicatorBank
            if hasattr(IndicatorBank, '_instance'):
                IndicatorBank._instance = None
            messages.append("✅ Cache IndicatorBank vidé")
        except Exception as e:
            messages.append(f"⚠️ IndicatorBank: {str(e)}")

        # 3. Vider les caches Streamlit
        st.cache_data.clear()
        st.cache_resource.clear()
        messages.append("✅ Caches Streamlit vidés")

        # 4. Nettoyer le session_state (conserver clés système)
        keys_to_keep = [k for k in st.session_state.keys() if k.startswith('_')]
        keys_to_delete = [k for k in st.session_state.keys() if not k.startswith('_')]

        for key in keys_to_delete:
            del st.session_state[key]

        messages.append(f"✅ Session_state nettoyé ({len(keys_to_delete)} clés supprimées)")

        # 5. Forcer le garbage collector Python (RAM système)
        collected = gc.collect()
        messages.append(f"✅ RAM système: {collected} objets collectés par GC")

        # 6. Reset Ollama
        ollama_success, ollama_msg = reset_ollama()
        if ollama_success:
            messages.append(f"✅ {ollama_msg}")
        else:
            messages.append(f"⚠️ Ollama: {ollama_msg}")
            success = False

        # Message final
        final_msg = "\n".join(messages)
        return success, final_msg

    except Exception as e:
        return False, f"❌ Erreur critique lors du nettoyage: {str(e)}"


def shutdown_app() -> None:
    """
    Arrête l'application Streamlit proprement.

    Nettoie toute la mémoire (GPU, cache, session) avant de quitter.
    """
    try:
        st.info("⏳ Arrêt en cours...")

        # 1. Arrêter le monitoring si actif
        try:
            monitor = get_global_monitor()
            if monitor.is_running():
                monitor.stop()
                st.caption("✅ Monitoring arrêté")
        except Exception as e:
            logging.debug(f"Monitor stop failed (ignoré): {e}")

        # 2. Nettoyer GPU Manager si actif
        try:
            from threadx.gpu.multi_gpu import get_default_manager
            manager = get_default_manager()
            if hasattr(manager, 'stop'):
                manager.stop()
                st.caption("✅ GPU Manager arrêté")
        except Exception as e:
            logging.debug(f"GPU Manager stop failed (ignoré): {e}")

        # 3. Nettoyer toute la mémoire
        success, msg = clean_all_memory()
        if success:
            st.success("✅ Mémoire nettoyée")
        else:
            st.warning(f"⚠️ Nettoyage partiel: {msg}")

        # Message utilisateur final
        st.success("✅ Application arrêtée proprement. Vous pouvez fermer cet onglet.")
        st.info("💡 Pour redémarrer : `streamlit run src/threadx/streamlit_app.py`")

        # Arrêter Streamlit (force rerun puis stop)
        time.sleep(1.5)
        st.stop()

    except Exception as e:
        st.error(f"❌ Erreur lors de l'arrêt: {str(e)}")


def restart_app() -> None:
    """
    Redémarre l'application en réinitialisant tout (cache, GPU, session).

    Équivalent à un premier démarrage : tous les caches sont vidés,
    la session est réinitialisée, les GPUs sont nettoyés.
    """
    try:
        st.info("🔄 Redémarrage en cours...")

        # 1. Arrêter le monitoring
        try:
            monitor = get_global_monitor()
            if monitor.is_running():
                monitor.stop()
                st.caption("✅ Monitoring arrêté")
        except Exception as e:
            logging.debug(f"Monitor stop failed (ignoré): {e}")

        # 2. Nettoyer GPU Manager
        try:
            from threadx.gpu.multi_gpu import get_default_manager
            manager = get_default_manager()
            if hasattr(manager, 'stop'):
                manager.stop()
            # Réinitialiser le singleton
            if hasattr(manager.__class__, '_default_manager'):
                manager.__class__._default_manager = None
            st.caption("✅ GPU Manager réinitialisé")
        except Exception as e:
            logging.debug(f"GPU Manager reset failed (ignoré): {e}")

        # 3. Nettoyage COMPLET mémoire
        success, msg = clean_all_memory()
        st.caption("✅ Mémoire nettoyée")

        # 4. Réinitialiser TOUTES les clés session_state (même système)
        all_keys = list(st.session_state.keys())
        for key in all_keys:
            del st.session_state[key]
        st.caption(f"✅ Session réinitialisée ({len(all_keys)} clés supprimées)")

        # 5. Vider tous les caches fichiers IndicatorBank
        try:
            from pathlib import Path
            cache_dir = Path("cache/indicators")
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                st.caption("✅ Cache fichiers vidé")
        except Exception as e:
            st.caption(f"⚠️ Cache fichiers: {e}")

        # Message final
        st.success("✅ Application prête à redémarrer !")
        st.info("🔄 La page va se recharger dans 2 secondes...")

        # Force rerun pour redémarrage complet
        time.sleep(2)
        st.rerun()

    except Exception as e:
        st.error(f"❌ Erreur lors du redémarrage: {str(e)}")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("# ThreadX v2.0")
        st.markdown("*Trading Quantitatif Haute Performance*")
        st.markdown("---")

        # Barre de progression du workflow
        st.markdown("### 📍 Progression")
        steps_total = 4
        page_to_step = {"config": 1, "backtest": 2, "llm": 3, "monitor": 4}
        current_page = st.session_state.get("page", "config")
        current_step = page_to_step.get(current_page, 1)

        # Afficher la barre de progression
        st.progress(current_step / steps_total)

        # Afficher les étapes avec statut
        st.caption(
            f"Étape 1/4 : Configuration données "
            f"{'✅' if current_step > 1 else '⏳' if current_step == 1 else '⭕'}"
        )
        st.caption(
            f"Étape 2/4 : Optimisation "
            f"{'✅' if current_step > 2 else '⏳' if current_step == 2 else '⭕'}"
        )
        st.caption(
            f"Étape 3/4 : Multi-LLM Optimizer "
            f"{'✅' if current_step > 3 else '⏳' if current_step == 3 else '⭕'}"
        )
        st.caption(
            f"Étape 4/4 : Monitoring système "
            f"{'⏳' if current_step == 4 else '⭕'}"
        )

        # Bouton "Suivant" selon l'étape actuelle
        if current_page == "config":
            if st.button("➡️ Passer à l'Optimisation", type="primary", use_container_width=True):
                st.session_state.page = "backtest"
                st.rerun()
        elif current_page == "backtest":
            col_bt1, col_bt2 = st.columns(2)
            with col_bt1:
                if st.button("🤖 Multi-LLM", use_container_width=True):
                    st.session_state.page = "llm"
                    st.rerun()
            with col_bt2:
                if st.button("📊 Monitoring", use_container_width=True):
                    st.session_state.page = "monitor"
                    st.rerun()
        elif current_page == "llm":
            col_bt1, col_bt2 = st.columns(2)
            with col_bt1:
                if st.button("📚 Historique", use_container_width=True):
                    st.session_state.page = "reports"
                    st.rerun()
            with col_bt2:
                if st.button("📊 Monitoring", use_container_width=True):
                    st.session_state.page = "monitor"
                    st.rerun()
        elif current_page == "reports":
            if st.button("🤖 Retour Multi-LLM", type="primary", use_container_width=True):
                st.session_state.page = "llm"
                st.rerun()

        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        labels = list(PAGE_TITLES.values())
        current_key = st.session_state.get("page", "config")
        current_label = PAGE_TITLES.get(current_key, labels[0])
        selected_label = st.radio(
            "Navigation",
            labels,
            index=labels.index(current_label),
            key="nav_radio",
            label_visibility="collapsed",
        )
        selected_key = next(k for k, v in PAGE_TITLES.items() if v == selected_label)
        if selected_key != current_key:
            st.session_state.page = selected_key
            st.rerun()
        st.markdown("---")
        st.markdown("### ⚙️ Système")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Backend", "NumPy")
        with col2:
            st.metric("Config", "TOML")

        # === CONFIGURATION GLOBALE ===
        st.markdown("---")
        st.markdown("### 🎛️ Configuration Globale")

        # GPU & Calcul - Détection dynamique
        with st.expander("🖥️ GPU & Calcul", expanded=True):
            # Détection GPUs disponibles
            try:
                from threadx.gpu.device_manager import list_devices
                devices = list_devices(use_cache=True)
                gpu_devices = [d for d in devices if d.device_id >= 0]

                if len(gpu_devices) == 0:
                    st.warning("⚠️ Aucun GPU détecté - Mode CPU uniquement")
                elif len(gpu_devices) == 1:
                    gpu = gpu_devices[0]
                    st.success(f"✅ GPU détecté : {gpu.name}")
                    st.caption(f"   └─ {gpu.memory_total_gb:.1f} GB VRAM")
                else:
                    # Multi-GPU disponible
                    gpu_names = [d.name for d in gpu_devices]
                    st.success(f"✅ {len(gpu_devices)} GPUs détectés : {', '.join(gpu_names)}")
                    for gpu in gpu_devices:
                        st.caption(f"   • GPU {gpu.device_id} ({gpu.name}): {gpu.memory_total_gb:.1f} GB VRAM")

                # Checkbox Multi-GPU (seulement si 2+ GPUs)
                if len(gpu_devices) >= 2:
                    use_multigpu = st.checkbox(
                        "🔥 Activer Multi-GPU",
                        value=st.session_state.get("global_use_multigpu", True),
                        key="global_use_multigpu",
                        help=f"Répartit les calculs sur {len(gpu_devices)} GPUs. Décocher si un GPU est utilisé ailleurs (Ollama, autre app)"
                    )

                    if use_multigpu:
                        # Afficher la balance depuis le manager
                        try:
                            from threadx.gpu.multi_gpu import get_default_manager
                            manager = get_default_manager()
                            balance_str = " + ".join([f"{name} ({ratio:.0%})" for name, ratio in manager.device_balance.items() if name != "cpu"])
                            st.caption(f"⚖️  Répartition : {balance_str}")
                        except Exception:
                            st.caption("⚖️  Répartition automatique selon VRAM")
                    else:
                        st.caption(f"⚡ GPU principal uniquement ({gpu_devices[0].name})")
                else:
                    # Single GPU : pas de checkbox multi-GPU
                    st.session_state["global_use_multigpu"] = False

            except Exception as e:
                st.error(f"❌ Erreur détection GPU : {e}")
                st.caption("Mode CPU sera utilisé")

        # Profil de Performances
        with st.expander("🔧 Profil de Performances", expanded=True):
            # Préréglages disponibles
            profiles = {
                "🚀 Optimisé": {"workers": 40, "feeder": 24, "desc": "Performance maximale (CPU ~35-40%)"},
                "⚖️ Équilibré": {"workers": 30, "feeder": 16, "desc": "Bon compromis (CPU ~25-30%)"},
                "💾 Économique": {"workers": 16, "feeder": 8, "desc": "Multitâche (CPU ~15-20%)"},
                "🔧 Personnalisé": {"workers": None, "feeder": None, "desc": "Réglages manuels"},
            }

            # Sélection du profil (Équilibré par défaut)
            default_profile = st.session_state.get("global_perf_profile", "⚖️ Équilibré")
            selected_profile = st.selectbox(
                "Choisir un profil",
                list(profiles.keys()),
                index=list(profiles.keys()).index(default_profile),
                key="global_perf_profile",
                help="Préréglages optimisés pour différents usages"
            )

            # Afficher description du profil
            st.caption(f"📋 {profiles[selected_profile]['desc']}")

            # Afficher ou permettre réglages manuels
            if selected_profile == "🔧 Personnalisé":
                manual_workers = st.number_input(
                    "Workers (parallélisme)",
                    min_value=2,
                    max_value=64,
                    value=st.session_state.get("global_manual_workers", 30),
                    step=2,
                    key="global_manual_workers",
                    help="Nombre de workers parallèles"
                )
                manual_feeder = st.select_slider(
                    "Feeder aggression (pipeline CPU)",
                    options=[1, 2, 4, 6, 8, 10, 12, 16, 24, 32],
                    value=st.session_state.get("global_manual_feeder", 16),
                    key="global_manual_feeder",
                    help="Contrôle la fenêtre de tâches en vol"
                )
                # Stocker valeurs manuelles
                st.session_state.global_workers = manual_workers
                st.session_state.global_feeder_aggr = manual_feeder
            else:
                # Utiliser valeurs du profil
                st.session_state.global_workers = profiles[selected_profile]["workers"]
                st.session_state.global_feeder_aggr = profiles[selected_profile]["feeder"]

                # Afficher valeurs en lecture seule
                col_w, col_f = st.columns(2)
                with col_w:
                    st.metric("Workers", profiles[selected_profile]["workers"])
                with col_f:
                    st.metric("Feeder", profiles[selected_profile]["feeder"])

        # Monitoring
        with st.expander("📊 Monitoring Temps Réel", expanded=False):
            enable_monitoring = st.checkbox(
                "Afficher stats CPU/GPU/RAM",
                value=st.session_state.get("global_monitoring", True),
                key="global_monitoring",
                help="Monitoring système pendant les calculs lourds"
            )

            if enable_monitoring:
                st.caption("✅ Stats temps réel activées")
            else:
                st.caption("⚠️ Monitoring désactivé (léger gain perfs)")

        # Intelligence Artificielle (LLM)
        with st.expander("🤖 Intelligence Artificielle", expanded=False):
            enable_llm = st.checkbox(
                "Activer analyse LLM (quand disponible)",
                value=st.session_state.get("global_enable_llm", False),
                key="global_enable_llm",
                help="Analyse IA des résultats de backtest (Ollama requis)"
            )

            if enable_llm:
                st.caption("🤖 Analyse LLM activée pour toutes les pages")
            else:
                st.caption("💡 Désactivé (gain mémoire/vitesse)")

        # Bouton Nettoyage Complet (fusion des 2 anciens boutons)
        st.markdown("---")
        if st.button("🧹 Nettoyage Complet (RAM + VRAM + Cache)", type="secondary", use_container_width=True, help="Vide la RAM système, VRAM GPU, tous les caches et reset Ollama"):
            with st.spinner("⏳ Nettoyage mémoire en cours..."):
                success, message = clean_all_memory()

                # Afficher résultat détaillé
                if success:
                    st.success("✅ Nettoyage terminé avec succès!")
                else:
                    st.warning("⚠️ Nettoyage partiel (voir détails)")

                # Afficher tous les messages dans un expander
                with st.expander("📋 Détails du nettoyage", expanded=True):
                    st.text(message)

                # Marquer pour réinitialisation
                st.session_state.session_initialized = False

                # Rerun après 1.5 secondes
                st.caption("🔄 Rechargement de l'application dans 1.5s...")
                time.sleep(1.5)
                st.rerun()

        # Panneau monitoring compact (activable à la demande)
        st.markdown("---")
        with st.expander("📡 Monitoring (sidebar)", expanded=False):
            monitor = get_global_monitor()
            sidebar_visible = st.checkbox(
                "Afficher le panneau",
                value=st.session_state.get("monitor_sidebar_visible", False),
                key="monitor_sidebar_visible",
            )
            auto_refresh_sb = st.checkbox(
                "Auto-refresh",
                value=st.session_state.get("monitor_autorefresh_sb", False),
                key="monitor_autorefresh_sb",
            )
            interval_sb = st.slider(
                "Intervalle (s)",
                0.25,
                5.0,
                st.session_state.get("monitor_interval_sb", 1.0),
                0.25,
                key="monitor_interval_sb",
            )

            # Gestion start/stop: actif si panneau visible OU page monitor sélectionnée
            page_key = st.session_state.get("page", "config")
            should_run = bool(sidebar_visible or page_key == "monitor")
            if should_run and not monitor.is_running():
                monitor.start()
            elif not should_run and monitor.is_running():
                monitor.stop()

            if sidebar_visible:
                df = monitor.get_history_df(n_last=180)
                if df.empty:
                    st.caption("Aucune donnée (activez l'auto-refresh)")
                else:
                    df_t = df.set_index("time")
                    st.line_chart(df_t[["cpu", "memory"]])
                    if (df[["gpu1", "gpu2"]].max() > 0).any():
                        st.line_chart(df_t[["gpu1", "gpu2"]])
                    else:
                        st.caption("GPU inactif ou non détecté")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(
                        "Vider", key="clear_hist_sb", use_container_width=True
                    ):
                        monitor.clear_history()
                with col_b:
                    st.caption("")

                if auto_refresh_sb:
                    time.sleep(float(interval_sb))
                    st.rerun()

        # Actions Système : Redémarrage et Arrêt
        st.markdown("---")
        st.markdown("### 🔧 Actions Système")

        col_restart, col_shutdown = st.columns(2)

        with col_restart:
            if st.button(
                "🔄 Redémarrer",
                type="secondary",
                use_container_width=True,
                help="Redémarre l'application en réinitialisant tout (cache, GPU, session)\nÉquivalent à un premier démarrage"
            ):
                restart_app()

        with col_shutdown:
            if st.button(
                "🛑 Arrêter",
                type="primary",
                use_container_width=True,
                help="Nettoie la mémoire et arrête l'application proprement"
            ):
                shutdown_app()

        st.caption("💡 **Redémarrer** : réinitialise tout (cache, GPU, session)")
        st.caption("💡 **Arrêter** : ferme l'application proprement")

        st.markdown("---")
        st.caption("**ThreadX v2.0** | © 2025")


def main() -> None:
    # NOTE: Reset Ollama au startup DÉSACTIVÉ par défaut (source de bugs)
    # L'utilisateur peut utiliser le bouton "Reset Ollama" dans la sidebar si nécessaire
    #
    # Si vous voulez réactiver le reset automatique, décommentez ce bloc:
    # if "ollama_reset_on_startup" not in st.session_state:
    #     st.session_state.ollama_reset_on_startup = True
    #     with st.spinner("⏳ Réinitialisation d'Ollama au démarrage..."):
    #         success, message = reset_ollama()
    #         if not success:
    #             logging.warning(f"Ollama reset failed on startup: {message}")

    init_session()
    render_sidebar()
    page_key = st.session_state.get("page", "config")
    renderer = PAGE_RENDERERS.get(page_key, config_page_main)
    renderer()


if __name__ == "__main__":
    main()
```
<!-- MODULE-END: streamlit_app.py -->

<!-- MODULE-START: client.py -->
```json
{
  "name": "client.py",
  "path": "llm\\client.py",
  "ext": ".py",
  "anchor": "client_py"
}
```
## client_py
*Chemin* : `llm\client.py`  
*Type* : `.py`  

```python
"""
ThreadX LLM Client
==================

Interface unifiée pour interagir avec des modèles LLM locaux via Ollama.

Features:
- Support multi-modèles (DeepSeek-R1, Gemma, Qwen, etc.)
- Timeout configurable et fallback gracieux
- Validation des réponses avec retry automatique
- Mode debug pour logging détaillé

Notes:
- UnicodeDecodeError dans thread Ollama: Erreur connue non-bloquante
  (thread interne de subprocess, n'affecte pas les résultats)

Usage:
    >>> from threadx.llm.client import LLMClient
    >>> client = LLMClient(model="deepseek-r1:8b")
    >>> response = client.complete("Analyse ces résultats...")
    >>> print(response)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

try:
    import ollama

    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    logging.warning("ollama package not installed. LLM features will be disabled.")


class LLMNotConfiguredError(Exception):
    """Exception levée quand un agent sans LLM tente d'appeler le LLM."""

    pass


class LLMClient:
    """
    Client LLM pour Ollama avec gestion d'erreurs robuste.

    Attributes:
        model: Nom du modèle Ollama (e.g., "deepseek-r1:8b", "deepseek-r1:32b")
        endpoint: URL de l'API Ollama (default: "http://localhost:11434")
        timeout: Timeout en secondes pour les requêtes (default: 60)
        max_retries: Nombre de tentatives en cas d'échec (default: 2)
        debug: Active le logging détaillé (default: False)
    """

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        endpoint: str = "http://localhost:11434",
        timeout: float = 60.0,
        max_retries: int = 2,
        debug: bool = False,
    ):
        if not HAS_OLLAMA:
            raise RuntimeError(
                "ollama package not installed. Install with: pip install ollama"
            )

        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug
        self.logger = logging.getLogger(__name__)

        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        # Vérifier que le modèle est disponible
        self._verify_model()

    def _verify_model(self) -> None:
        """Vérifie que le modèle Ollama est disponible."""
        try:
            models = ollama.list()
            available_models = [m.model for m in models.models]

            if self.model not in available_models:
                self.logger.warning(
                    f"Model {self.model} not found locally. "
                    f"Available: {available_models}. "
                    f"Ollama will attempt to download it on first use."
                )
        except Exception as e:
            self.logger.error(f"Failed to verify Ollama models: {e}")

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Génère une complétion simple.

        Args:
            prompt: Prompt utilisateur
            system: Message système optionnel (instructions)
            temperature: Température de sampling (0.0 = déterministe, 1.0 = créatif)
            max_tokens: Nombre maximum de tokens générés

        Returns:
            Réponse textuelle du LLM

        Raises:
            RuntimeError: Si la requête échoue après max_retries tentatives
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                start = time.time()

                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                )

                elapsed = time.time() - start
                content = response["message"]["content"]

                if self.debug:
                    self.logger.debug(
                        f"LLM completion successful in {elapsed:.2f}s "
                        f"(model={self.model}, tokens≈{len(content)//4})"
                    )

                return content

            except Exception as e:
                error_msg = str(e).lower()

                # Détection erreur CUDA - arrêt immédiat sans retry
                if "cuda error" in error_msg or "llama runner process has terminated" in error_msg:
                    self.logger.error(
                        f"Erreur CUDA détectée avec Ollama. Le GPU n'est pas disponible ou surchargé. "
                        f"Redémarrez Ollama ou utilisez un modèle CPU: {e}"
                    )
                    raise RuntimeError(
                        f"Erreur GPU Ollama (CUDA): {e}\n"
                        f"Solutions:\n"
                        f"1. Redémarrez Ollama: 'ollama serve'\n"
                        f"2. Vérifiez disponibilité GPU\n"
                        f"3. Utilisez un modèle plus petit\n"
                        f"4. Fermez autres applications GPU"
                    )

                self.logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {e}")
                time.sleep(2)  # Backoff augmenté pour stabilité

        return ""  # Unreachable but for type checker

    def complete_streaming(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """
        Génère une complétion en streaming (génère des chunks au fur et à mesure).

        Args:
            prompt: Prompt utilisateur
            system: Message système optionnel
            temperature: Température de sampling
            max_tokens: Nombre maximum de tokens

        Yields:
            str: Chunks de texte générés progressivement

        Raises:
            RuntimeError: Si la requête échoue
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                stream=True,
            )

            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except Exception as e:
            error_msg = str(e).lower()

            if "cuda error" in error_msg or "llama runner process has terminated" in error_msg:
                self.logger.error(f"Erreur CUDA détectée: {e}")
                raise RuntimeError(f"Erreur GPU Ollama (CUDA): {e}")

            raise RuntimeError(f"LLM streaming failed: {e}")

    def complete_structured(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """
        Génère une complétion avec réponse structurée JSON.

        Args:
            prompt: Prompt utilisateur (doit demander un JSON)
            system: Message système optionnel
            temperature: Température de sampling
            max_tokens: Nombre maximum de tokens

        Returns:
            Dict parsé depuis la réponse JSON

        Raises:
            RuntimeError: Si le parsing JSON échoue après tentatives
        """
        # Modifier le prompt pour forcer JSON
        json_prompt = f"{prompt}\n\nIMPORTANT: Réponds UNIQUEMENT avec du JSON valide, sans texte avant ou après."

        response_text = self.complete(json_prompt, system, temperature, max_tokens)

        # Stocker pour retry si besoin
        self._last_raw_response = response_text

        # Parsing JSON ultra-tolérant avec multiples stratégies
        parsed = self._parse_json_tolerant(response_text)

        if self.debug:
            self.logger.debug(f"Structured JSON parsed successfully: {list(parsed.keys())}")

        return parsed

    def _parse_json_tolerant(self, response_text: str) -> dict[str, Any]:
        """
        Parsing JSON ultra-tolérant avec fallbacks multiples.

        Stratégies:
        1. Extraction blocs markdown ```json...```
        2. Extraction objets JSON nus {...}
        3. Recherche première ligne commençant par {
        4. Correction échappements invalides

        Raises:
            RuntimeError: Si toutes stratégies échouent
        """
        import re

        json_candidates = []

        # STRATÉGIE 1: Chercher blocs markdown ```json ... ```
        for match in re.finditer(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL):
            json_candidates.append(("markdown_block", match.group(1)))

        # STRATÉGIE 2: Chercher objets JSON nus {...} (avec nested support)
        for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL):
            candidate = match.group(0)
            # Filtrer faux positifs (trop courts)
            if len(candidate) > 20:
                json_candidates.append(("naked_object", candidate))

        # STRATÉGIE 3: Chercher première ligne commençant par {
        lines = response_text.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_text = '\n'.join(lines[i:])
                json_candidates.append(("first_brace_line", json_text))
                break

        # Tenter parsing sur chaque candidat
        for strategy, candidate in json_candidates:
            try:
                # Nettoyer et corriger échappements
                cleaned = self._fix_json_escapes(candidate.strip())

                # Parser avec tolérance
                parsed = json.loads(cleaned, strict=False)

                if isinstance(parsed, dict) and len(parsed) > 0:
                    if self.debug:
                        self.logger.debug(
                            f"✅ JSON parsed via strategy '{strategy}' "
                            f"(keys: {list(parsed.keys())})"
                        )
                    return parsed

            except json.JSONDecodeError as e:
                if self.debug:
                    self.logger.debug(
                        f"Strategy '{strategy}' failed: {e} "
                        f"(pos {e.lineno}:{e.colno})"
                    )
                continue

        # ÉCHEC FINAL: Log détaillé et erreur explicite
        self.logger.error(
            f"❌ JSON parsing failed after all strategies.\n"
            f"Response (first 1000 chars):\n{response_text[:1000]}\n"
            f"Candidates tried: {len(json_candidates)} "
            f"({[s for s, _ in json_candidates]})"
        )
        raise RuntimeError(
            f"LLM returned invalid JSON after exhaustive parsing. "
            f"Response length: {len(response_text)} chars. "
            f"Tried {len(json_candidates)} extraction strategies."
        )

    def _fix_json_escapes(self, text: str) -> str:
        """
        Corrige les séquences d'échappement invalides dans le JSON.

        Échappements valides en JSON: quote, backslash, slash, b, f, n, r, t, uXXXX
        Tout autre backslash-X est invalide et sera converti en X.
        """
        import re

        def replace_escape(match):
            escaped_char = match.group(1)
            # Garder les échappements valides
            if escaped_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                return match.group(0)  # Garder tel quel
            else:
                # Remplacer \X par X (enlever le backslash)
                return escaped_char

        # Remplacer les échappements invalides
        fixed = re.sub(r'\\(.)', replace_escape, text)
        return fixed

    def complete_structured_with_retry(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_json_retries: int = 2,
    ) -> dict[str, Any]:
        """
        Appel LLM structuré avec retry intelligent si JSON invalide.

        Si parsing échoue:
        1. Extraire fragment JSON cassé
        2. Redemander au LLM de corriger avec feedback
        3. Parser à nouveau

        Args:
            prompt: Prompt utilisateur
            system: Message système optionnel
            temperature: Température de sampling
            max_tokens: Nombre max de tokens
            max_json_retries: Nombre de tentatives de réparation JSON

        Returns:
            Dict parsé depuis la réponse JSON

        Raises:
            RuntimeError: Si échec après toutes tentatives
        """
        current_prompt = prompt

        for attempt in range(max_json_retries):
            try:
                return self.complete_structured(
                    current_prompt, system, temperature, max_tokens
                )

            except RuntimeError as e:
                if "invalid JSON" in str(e) and attempt < max_json_retries - 1:
                    # Extraire réponse brute
                    last_response = getattr(self, '_last_raw_response', 'N/A')

                    # Prompt de réparation
                    repair_prompt = f"""Ta dernière réponse contenait du JSON invalide.

Réponse brute (premiers 500 chars):
{last_response[:500]}

Erreur de parsing: {str(e)}

Réponds UNIQUEMENT avec du JSON valide, sans texte avant ou après.
Format attendu: {{"key": "value", ...}}

JSON corrigé:
"""
                    self.logger.warning(
                        f"⚠️ JSON parse failed (attempt {attempt+1}/{max_json_retries}), "
                        f"asking LLM to repair..."
                    )

                    # Redemander avec prompt réparation
                    current_prompt = repair_prompt
                    continue

                # Échec final ou erreur non-JSON
                raise

        raise RuntimeError(
            f"JSON parsing failed after {max_json_retries} repair attempts"
        )

    def interpret_backtest_results(
        self,
        summary: dict[str, Any],
        params: dict[str, Any],
        trades_df: Any = None,
    ) -> dict[str, Any]:
        """
        Interprète des résultats de backtest avec analyse intelligente.

        Args:
            summary: Dict de métriques (sharpe, drawdown, win_rate, etc.)
            params: Dict de paramètres de stratégie testés
            trades_df: DataFrame de trades optionnel (pour analyse approfondie)

        Returns:
            Dict avec clés:
            - interpretation: str (résumé global)
            - strengths: list[str] (forces)
            - weaknesses: list[str] (faiblesses)
            - recommendations: list[str] (actions concrètes)
            - risk_level: str (LOW/MODERATE/HIGH)
            - suitability: str (profil investisseur)
        """
        from threadx.llm.prompts import BACKTEST_INTERPRETATION_PROMPT

        # Formater les métriques pour le prompt
        metrics_str = "\n".join([f"  - {k}: {v}" for k, v in summary.items() if v is not None])
        params_str = "\n".join([f"  - {k}: {v}" for k, v in params.items()])

        # Contexte additionnel sur les trades
        trades_context = ""
        if trades_df is not None and hasattr(trades_df, "__len__") and len(trades_df) > 0:
            trades_context = f"\n  - Nombre de trades: {len(trades_df)}"

        prompt = BACKTEST_INTERPRETATION_PROMPT.format(
            metrics=metrics_str, params=params_str, trades_context=trades_context
        )

        system = (
            "Tu es un analyste quantitatif expert avec 10+ ans d'expérience "
            "en trading algorithmique. Analyse les résultats avec rigueur et pragmatisme."
        )

        try:
            result = self.complete_structured(prompt, system=system, temperature=0.6, max_tokens=1500)

            # Valider les clés attendues
            required_keys = [
                "interpretation",
                "strengths",
                "weaknesses",
                "recommendations",
                "risk_level",
                "suitability",
            ]
            for key in required_keys:
                if key not in result:
                    self.logger.warning(f"Missing key '{key}' in LLM response, using default")
                    result[key] = [] if key in ["strengths", "weaknesses", "recommendations"] else "UNKNOWN"

            return result

        except Exception as e:
            self.logger.error(f"Backtest interpretation failed: {e}", exc_info=True)
            # Fallback gracieux
            return {
                "interpretation": f"Erreur d'interprétation LLM: {e}",
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "risk_level": "UNKNOWN",
                "suitability": "Non analysé (erreur LLM)",
            }
```
<!-- MODULE-END: client.py -->

<!-- MODULE-START: interpreters.py -->
```json
{
  "name": "interpreters.py",
  "path": "llm\\interpreters.py",
  "ext": ".py",
  "anchor": "interpreters_py"
}
```
## interpreters_py
*Chemin* : `llm\interpreters.py`  
*Type* : `.py`  

```python
"""
ThreadX LLM Response Interpreters
==================================

Parsers et validateurs pour structurer les réponses LLM en objets Python.

Features:
- Validation des clés requises
- Coercition de types (str → list, etc.)
- Fallback values par défaut
- Logging des erreurs de parsing
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_backtest_interpretation(response: dict[str, Any]) -> dict[str, Any]:
    """
    Parse et valide une réponse d'interprétation de backtest.

    Args:
        response: Dict brut depuis LLM

    Returns:
        Dict validé avec structure garantie:
        {
            "interpretation": str,
            "strengths": list[str],
            "weaknesses": list[str],
            "recommendations": list[str],
            "risk_level": str,
            "suitability": str
        }
    """
    # Valeurs par défaut
    defaults = {
        "interpretation": "Analyse non disponible",
        "strengths": [],
        "weaknesses": [],
        "recommendations": [],
        "risk_level": "UNKNOWN",
        "suitability": "Non analysé",
    }

    # Fusionner avec la réponse
    result = defaults.copy()

    for key in defaults:
        if key in response:
            value = response[key]

            # Validation des types
            if key in ["strengths", "weaknesses", "recommendations"]:
                # Doit être une liste
                if isinstance(value, str):
                    # Convertir string → liste
                    result[key] = [value] if value else []
                elif isinstance(value, list):
                    result[key] = [str(item) for item in value if item]
                else:
                    logger.warning(f"Invalid type for {key}: {type(value)}, using default")
                    result[key] = defaults[key]

            elif key == "risk_level":
                # Valider le niveau de risque
                valid_levels = ["LOW", "MODERATE", "HIGH", "UNKNOWN"]
                if isinstance(value, str):
                    upper_val = value.upper().strip()
                    if upper_val in valid_levels:
                        result[key] = upper_val
                    else:
                        logger.warning(f"Invalid risk_level '{value}', defaulting to UNKNOWN")
                        result[key] = "UNKNOWN"

            else:
                # Autres champs (str)
                result[key] = str(value) if value else defaults[key]

    # Log validation
    logger.debug(
        f"Parsed interpretation: {len(result['strengths'])} strengths, "
        f"{len(result['weaknesses'])} weaknesses, "
        f"{len(result['recommendations'])} recommendations"
    )

    return result


def parse_param_recommendation(response: dict[str, Any]) -> dict[str, Any]:
    """
    Parse une réponse de recommandation de paramètres.

    Args:
        response: Dict brut depuis LLM

    Returns:
        Dict validé avec structure:
        {
            "recommended_params": dict,
            "reasoning": dict,
            "confidence": float,
            "alternatives": list[dict]
        }
    """
    defaults = {
        "recommended_params": {},
        "reasoning": {},
        "confidence": 0.5,
        "alternatives": [],
    }

    result = defaults.copy()

    # recommended_params
    if "recommended_params" in response and isinstance(response["recommended_params"], dict):
        result["recommended_params"] = response["recommended_params"]

    # reasoning
    if "reasoning" in response and isinstance(response["reasoning"], dict):
        result["reasoning"] = response["reasoning"]

    # confidence
    if "confidence" in response:
        try:
            conf = float(response["confidence"])
            result["confidence"] = max(0.0, min(1.0, conf))  # Clamp [0, 1]
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence value: {response['confidence']}")
            result["confidence"] = 0.5

    # alternatives
    if "alternatives" in response and isinstance(response["alternatives"], list):
        result["alternatives"] = response["alternatives"]

    return result


def parse_anomaly_detection(response: dict[str, Any]) -> dict[str, Any]:
    """
    Parse une réponse de détection d'anomalies.

    Args:
        response: Dict brut depuis LLM

    Returns:
        Dict validé avec structure:
        {
            "anomalies_detected": bool,
            "suspicious_results": list[dict],
            "overall_quality": str,
            "warnings": list[str]
        }
    """
    defaults = {
        "anomalies_detected": False,
        "suspicious_results": [],
        "overall_quality": "UNKNOWN",
        "warnings": [],
    }

    result = defaults.copy()

    # anomalies_detected
    if "anomalies_detected" in response:
        result["anomalies_detected"] = bool(response["anomalies_detected"])

    # suspicious_results
    if "suspicious_results" in response and isinstance(response["suspicious_results"], list):
        result["suspicious_results"] = response["suspicious_results"]

    # overall_quality
    valid_qualities = ["EXCELLENT", "GOOD", "SUSPICIOUS", "POOR", "UNKNOWN"]
    if "overall_quality" in response:
        quality = str(response["overall_quality"]).upper().strip()
        if quality in valid_qualities:
            result["overall_quality"] = quality

    # warnings
    if "warnings" in response:
        if isinstance(response["warnings"], list):
            result["warnings"] = [str(w) for w in response["warnings"] if w]
        elif isinstance(response["warnings"], str):
            result["warnings"] = [response["warnings"]]

    return result


def parse_strategy_debug(response: dict[str, Any]) -> dict[str, Any]:
    """
    Parse une réponse de debugging de stratégie.

    Args:
        response: Dict brut depuis LLM

    Returns:
        Dict validé avec structure:
        {
            "diagnosis": str,
            "root_cause": str,
            "fix": str,
            "preventive_measures": list[str],
            "confidence": float
        }
    """
    defaults = {
        "diagnosis": "Non diagnostiqué",
        "root_cause": "Non identifié",
        "fix": "Aucune solution proposée",
        "preventive_measures": [],
        "confidence": 0.0,
    }

    result = defaults.copy()

    # Champs texte
    for field in ["diagnosis", "root_cause", "fix"]:
        if field in response:
            result[field] = str(response[field]) if response[field] else defaults[field]

    # preventive_measures
    if "preventive_measures" in response:
        if isinstance(response["preventive_measures"], list):
            result["preventive_measures"] = [str(m) for m in response["preventive_measures"] if m]
        elif isinstance(response["preventive_measures"], str):
            result["preventive_measures"] = [response["preventive_measures"]]

    # confidence
    if "confidence" in response:
        try:
            conf = float(response["confidence"])
            result["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            result["confidence"] = 0.0

    return result
```
<!-- MODULE-END: interpreters.py -->

<!-- MODULE-START: model_router.py -->
```json
{
  "name": "model_router.py",
  "path": "llm\\model_router.py",
  "ext": ".py",
  "anchor": "model_router_py"
}
```
## model_router_py
*Chemin* : `llm\model_router.py`  
*Type* : `.py`  

```python
import random
from enum import Enum
from typing import Optional, List, Dict

class TaskType(Enum):
    INITIALIZATION = "initialization"
    OPTIMIZATION = "optimization"
    AUDIT = "audit"
    GENERAL = "general"

class ModelRouter:
    """
    Handles the selection of LLM models based on the task type and iteration context.
    Implements the "Architect & Builder" strategy with Guest Auditors.
    """

    DEFAULT_ARCHITECT_MODEL = "deepseek-r1:32b"
    DEFAULT_BUILDER_MODEL = "deepseek-r1:14b"
    DEFAULT_GUEST_MODELS = [
        "qwen2.5:32b",
        "qwen2.5:14b",
        "mistral:22b",
        "mistral:7b",
    ]

    def __init__(
        self,
        audit_interval: int = 5,
        architect_model: str | None = None,
        builder_model: str | None = None,
        guest_models: list[str] | None = None,
    ):
        """
        Args:
            audit_interval: Number of optimization steps between guest audits.
            architect_model: Override for architect.
            builder_model: Override for builder.
            guest_models: Override list for auditors (round-robin).
        """
        self.audit_interval = audit_interval
        self._step_counter = 0
        self._guest_index = 0
        self.architect_model = architect_model or self.DEFAULT_ARCHITECT_MODEL
        self.builder_model = builder_model or self.DEFAULT_BUILDER_MODEL
        self.guest_models = guest_models or list(self.DEFAULT_GUEST_MODELS)

    def get_model_for_task(self, task_type: TaskType, step_number: Optional[int] = None) -> str:
        """
        Determines the best model to use for a given task.

        Args:
            task_type: The type of task (INITIALIZATION, OPTIMIZATION, etc.)
            step_number: The current generation/step number (for optimization loops).

        Returns:
            The name of the ollama model to use.
        """
        if task_type == TaskType.INITIALIZATION:
            return self.architect_model

        if task_type == TaskType.OPTIMIZATION:
            # If step_number is provided, check for audit rotation
            if step_number is not None and step_number > 0:
                if step_number % self.audit_interval == 0:
                    return self._get_next_guest_model()

            # Default builder for optimization
            return self.builder_model

        if task_type == TaskType.AUDIT:
            return self._get_random_guest_model()

        # Default fallback
        return self.builder_model

    def _get_next_guest_model(self) -> str:
        """Selects the next guest model in round-robin order."""
        if not self.guest_models:
            return self.builder_model
        selected = self.guest_models[self._guest_index % len(self.guest_models)]
        self._guest_index += 1
        return selected

    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Returns metadata about the model role."""
        if model_name == self.architect_model:
            return {"role": "Architect", "desc": "High-reasoning foundation builder"}
        elif model_name == self.builder_model:
            return {"role": "Builder", "desc": "Fast iterative optimizer"}
        elif model_name in self.guest_models:
            return {"role": "Guest Auditor", "desc": "External perspective for validation"}
        else:
            return {"role": "Unknown", "desc": "Generic model"}
```
<!-- MODULE-END: model_router.py -->

<!-- MODULE-START: ollama_manager.py -->
```json
{
  "name": "ollama_manager.py",
  "path": "llm\\ollama_manager.py",
  "ext": ".py",
  "anchor": "ollama_manager_py"
}
```
## ollama_manager_py
*Chemin* : `llm\ollama_manager.py`  
*Type* : `.py`  

```python
"""
Ollama Manager - Gestion automatique d'Ollama pour éviter les blocages
========================================================================

Ce module gère :
1. Auto-démarrage d'Ollama avant chaque run
2. Nettoyage du cache LLM entre les runs
3. Déchargement des modèles après utilisation
"""

import subprocess
import time
import platform
import requests
from typing import Tuple
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def ensure_ollama_running() -> Tuple[bool, str]:
    """
    S'assure qu'Ollama est démarré et fonctionnel.

    Returns:
        tuple[bool, str]: (succès, message)
    """
    # 1. Vérifier si Ollama répond
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if response.status_code == 200:
            logger.info("✅ Ollama déjà actif")
            return True, "✅ Ollama actif"
    except Exception as e:
        logger.debug("Ollama check failed (ignoré, démarrage): %s", e)
        pass  # Ollama pas actif, on va le démarrer

    # 2. Démarrer Ollama
    logger.info("🚀 Démarrage d'Ollama...")
    try:
        is_windows = platform.system() == "Windows"

        if is_windows:
            # Windows : Lancer directement avec Popen et flags de création console
            kwargs = {}
            # CREATE_NEW_CONSOLE = 0x00000010
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE

            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs
            )
        else:
            # Linux/Mac
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        # 3. Attendre qu'Ollama soit prêt (max 10s)
        for i in range(10):
            time.sleep(1)
            try:
                response = requests.get("http://127.0.0.1:11434/api/tags", timeout=1)
                if response.status_code == 200:
                    logger.info(f"✅ Ollama démarré avec succès (après {i+1}s)")
                    return True, f"✅ Ollama démarré ({i+1}s)"
            except Exception as e:
                logger.debug(f"Ollama startup check attempt {i+1}/10 failed (retry): {e}")
                continue

        return False, "⏱️ Timeout - Ollama n'a pas démarré en 10s"

    except FileNotFoundError:
        return False, "❌ Ollama non trouvé (vérifiez l'installation)"
    except Exception as e:
        return False, f"❌ Erreur: {str(e)}"


def unload_model(model_name: str) -> bool:
    """
    Décharge un modèle Ollama de la mémoire GPU/RAM.

    Args:
        model_name: Nom du modèle (ex: "deepseek-r1:32b")

    Returns:
        bool: True si succès
    """
    try:
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
        if success:
            logger.info(f"💾 Modèle {model_name} déchargé de la mémoire")
        return success
    except Exception as e:
        logger.warning(f"⚠️ Impossible de décharger {model_name}: {e}")
        return False


def cleanup_all_models() -> int:
    """
    Décharge TOUS les modèles Ollama de la mémoire.

    Returns:
        int: Nombre de modèles déchargés
    """
    try:
        # Lister les modèles chargés
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return 0

        models = response.json().get("models", [])
        count = 0

        for model in models:
            model_name = model.get("name", "")
            if model_name and unload_model(model_name):
                count += 1

        if count > 0:
            logger.info(f"🧹 {count} modèle(s) déchargé(s) de la mémoire")

        return count

    except Exception as e:
        logger.warning(f"⚠️ Erreur cleanup_all_models: {e}")
        return 0


def prepare_for_llm_run() -> Tuple[bool, str]:
    """
    Prépare l'environnement pour un run LLM.

    Actions:
    1. S'assure qu'Ollama est actif
    2. Nettoie les modèles précédents en mémoire

    Returns:
        tuple[bool, str]: (succès, message détaillé)
    """
    messages = []

    # 1. Nettoyer les modèles précédents
    cleaned = cleanup_all_models()
    if cleaned > 0:
        messages.append(f"🧹 {cleaned} modèle(s) déchargé(s)")

    # 2. S'assurer qu'Ollama est actif
    success, msg = ensure_ollama_running()
    messages.append(msg)

    if success:
        time.sleep(1)  # Petite pause pour stabilité
        return True, " | ".join(messages)
    else:
        return False, " | ".join(messages)


```
<!-- MODULE-END: ollama_manager.py -->

<!-- MODULE-START: prompts.py -->
```json
{
  "name": "prompts.py",
  "path": "llm\\prompts.py",
  "ext": ".py",
  "anchor": "prompts_py"
}
```
## prompts_py
*Chemin* : `llm\prompts.py`  
*Type* : `.py`  

```python
"""
ThreadX LLM Prompts Templates
==============================

Templates de prompts réutilisables pour différentes tâches LLM.

Conventions:
- Variables: {variable_name} (format Python str.format)
- Structure: System prompt séparé du user prompt
- Output: Toujours demander du JSON structuré pour faciliter le parsing
"""

BACKTEST_INTERPRETATION_PROMPT = """Analyse ces résultats de backtest d'une stratégie de trading quantitatif:

**Métriques de performance:**
{metrics}

**Paramètres de stratégie testés:**
{params}
{trades_context}

**Objectif:** Fournis une analyse complète pour aider le trader à comprendre la qualité de ces résultats et à améliorer sa stratégie.

**Instructions:**
1. **Interprétation globale** (2-3 phrases): Résume la qualité générale (excellent/bon/moyen/faible) avec les raisons principales
2. **Forces** (3-5 points): Liste les métriques positives et ce qu'elles signifient concrètement
3. **Faiblesses** (3-5 points): Identifie les problèmes et leurs implications pratiques
4. **Recommandations** (3-5 actions): Suggestions concrètes pour améliorer les paramètres ou la stratégie
5. **Niveau de risque**: LOW (conservateur), MODERATE (équilibré), ou HIGH (agressif)
6. **Profil adapté**: Quel type de trader devrait utiliser cette stratégie

**Contexte métrique:**
- Sharpe ratio: >1.5 excellent, 1.0-1.5 bon, 0.5-1.0 moyen, <0.5 faible
- Drawdown: <10% excellent, 10-20% acceptable, 20-30% élevé, >30% très risqué
- Win rate: >60% bon pour mean-reversion, >40% bon pour trend-following
- Profit factor: >2.0 excellent, 1.5-2.0 bon, 1.0-1.5 moyen, <1.0 perdant

**Format de réponse (JSON):**
```json
{{
  "interpretation": "Résumé global en 2-3 phrases concises",
  "strengths": [
    "Force 1 avec explication concrète",
    "Force 2 avec métrique précise",
    "..."
  ],
  "weaknesses": [
    "Faiblesse 1 avec impact pratique",
    "Faiblesse 2 avec chiffres",
    "..."
  ],
  "recommendations": [
    "Action 1 concrète (ex: augmenter atr_multiplier de 1.5 à 2.0)",
    "Action 2 actionnable",
    "..."
  ],
  "risk_level": "LOW|MODERATE|HIGH",
  "suitability": "Description du profil de trader adapté (1 phrase)"
}}
```

Sois pragmatique, précis et actionnable. Évite le jargon inutile.
"""

PARAM_RECOMMENDATION_PROMPT = """Tu es un expert en optimisation de stratégies de trading algorithmique.

**Contexte:**
Régime de marché actuel détecté:
{market_regime}

Stratégie à optimiser: {strategy_name}

Paramètres actuels:
{current_params}

Performance récente:
{recent_performance}

**Objectif:** Recommande des paramètres optimaux adaptés au régime de marché actuel avec justifications précises.

**Instructions:**
1. Analyse le régime de marché (volatilité, tendance, volume)
2. Identifie les paramètres clés à ajuster selon le régime
3. Recommande des valeurs concrètes avec raisonnement
4. Fournis 2-3 configurations alternatives (conservateur/équilibré/agressif)
5. Estime le niveau de confiance de la recommandation

**Format de réponse (JSON):**
```json
{{
  "recommended_params": {{
    "param1": valeur,
    "param2": valeur,
    "..."
  }},
  "reasoning": {{
    "param1": "Justification précise basée sur le régime",
    "param2": "Raison technique avec référence",
    "..."
  }},
  "confidence": 0.0 à 1.0,
  "alternatives": [
    {{
      "profile": "CONSERVATIVE|BALANCED|AGGRESSIVE",
      "params": {{}},
      "expected_outcome": "Description courte"
    }}
  ]
}}
```
"""

ANOMALY_DETECTION_PROMPT = """Analyse ces résultats de sweep d'optimisation pour détecter des anomalies:

**Top résultats:**
{top_results}

**Statistiques globales:**
{global_stats}

**Objectif:** Identifier les résultats suspects qui pourraient indiquer:
- Overfitting (métriques irréalistes)
- Données corrompues (valeurs aberrantes)
- Configurations instables (variance élevée)
- Artéfacts numériques (calculs incorrects)

**Format de réponse (JSON):**
```json
{{
  "anomalies_detected": true|false,
  "suspicious_results": [
    {{
      "combo_id": int,
      "reason": "Explication de l'anomalie",
      "severity": "LOW|MEDIUM|HIGH",
      "recommendation": "Action suggérée"
    }}
  ],
  "overall_quality": "EXCELLENT|GOOD|SUSPICIOUS|POOR",
  "warnings": ["Avertissement global 1", "..."]
}}
```
"""

STRATEGY_DEBUG_PROMPT = """Aide à debugger cette stratégie de trading qui rencontre des problèmes:

**Erreur/Symptôme:**
{error_description}

**Configuration:**
Stratégie: {strategy_name}
Paramètres: {params}

**Logs d'erreur:**
{error_logs}

**Données contextuelles:**
{context_data}

**Objectif:** Diagnostiquer le problème et proposer un correctif.

**Format de réponse (JSON):**
```json
{{
  "diagnosis": "Description du problème identifié",
  "root_cause": "Cause racine technique",
  "fix": "Solution concrète étape par étape",
  "preventive_measures": ["Mesure 1", "Mesure 2"],
  "confidence": 0.0 à 1.0
}}
```
"""

REPORT_GENERATION_PROMPT = """Génère un rapport d'optimisation professionnel en Markdown:

**Résultats d'optimisation:**
{optimization_results}

**Configuration du sweep:**
{sweep_config}

**Statistiques:**
{statistics}

**Objectif:** Créer un rapport clair, structuré et actionnable pour présentation.

**Structure attendue:**
1. Résumé exécutif (3-4 phrases)
2. Meilleure configuration trouvée
3. Insights statistiques (corrélations, sweet spots)
4. Visualisation des résultats (description textuelle)
5. Recommandations finales

**Format:** Markdown pur, sans JSON.
"""
```
<!-- MODULE-END: prompts.py -->

<!-- MODULE-START: run_report.py -->
```json
{
  "name": "run_report.py",
  "path": "llm\\run_report.py",
  "ext": ".py",
  "anchor": "run_report_py"
}
```
## run_report_py
*Chemin* : `llm\run_report.py`  
*Type* : `.py`  

```python
"""
ThreadX - LLM Run Report & Index System
========================================

Système de génération de rapports et d'indexation pour les runs Multi-LLM.

Features:
- Génération de rapports JSON structurés
- Index centralisé pour recherche/réutilisation
- Export HTML optionnel
- Catégorisation: Sweep, Analysis, Proposals, Tests

Usage:
    >>> from threadx.llm.run_report import LLMRunReport, RunIndex
    >>>
    >>> # Créer un rapport
    >>> report = LLMRunReport(
    ...     strategy_name="MA_Crossover",
    ...     sweep_results=sweep_df,
    ...     analysis=analyst_result,
    ...     proposals=strategist_result,
    ...     test_results=test_results,
    ... )
    >>>
    >>> # Sauvegarder et indexer
    >>> index = RunIndex()
    >>> report_path = index.save_report(report)
    >>>
    >>> # Rechercher des runs
    >>> runs = index.search(strategy="MA_Crossover", min_sharpe=0.5)
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Répertoire par défaut pour les rapports
DEFAULT_REPORTS_DIR = Path("reports/llm_runs")


# ============================================================
# REPORT DATA STRUCTURES
# ============================================================


@dataclass
class SweepSection:
    """Section des résultats du sweep."""

    total_configs: int
    duration_seconds: float
    top_configs: list[dict[str, Any]]  # Top 10-20 configs
    params_tested: dict[str, list[Any]]  # Paramètres et leurs valeurs
    metrics_summary: dict[str, float]  # avg_sharpe, max_sharpe, etc.

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_sweep_results(
        cls,
        results: list[dict],
        sweep_params: dict,
        duration: float,
        top_n: int = 20
    ) -> "SweepSection":
        """Crée une SweepSection depuis les résultats bruts du sweep."""
        df = pd.DataFrame(results)

        # Extraire top configs
        if "sharpe_ratio" in df.columns:
            top_df = df.nlargest(top_n, "sharpe_ratio")
        elif "sharpe" in df.columns:
            top_df = df.nlargest(top_n, "sharpe")
        else:
            top_df = df.head(top_n)

        top_configs = top_df.to_dict("records")

        # Calculer métriques summary
        metrics_summary = {}
        for col in ["sharpe_ratio", "sharpe", "total_return", "pnl_pct", "max_drawdown", "win_rate"]:
            if col in df.columns:
                metrics_summary[f"avg_{col}"] = float(df[col].mean())
                metrics_summary[f"max_{col}"] = float(df[col].max())
                metrics_summary[f"min_{col}"] = float(df[col].min())
                metrics_summary[f"std_{col}"] = float(df[col].std())

        return cls(
            total_configs=len(results),
            duration_seconds=duration,
            top_configs=top_configs,
            params_tested=sweep_params,
            metrics_summary=metrics_summary,
        )


@dataclass
class AnalysisSection:
    """Section de l'analyse LLM (Analyst)."""

    model_used: str
    duration_seconds: float
    patterns: list[str]
    key_metrics: dict[str, float]
    trade_offs: list[str]
    recommendations: list[str]
    raw_response: Optional[str] = None  # Réponse brute pour debug

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_analyst_result(
        cls,
        result: dict,
        model: str,
        duration: float
    ) -> "AnalysisSection":
        """Crée une AnalysisSection depuis le résultat de l'Analyst."""
        analysis = result.get("analysis", {})

        return cls(
            model_used=model,
            duration_seconds=duration,
            patterns=analysis.get("patterns", []),
            key_metrics=analysis.get("key_metrics", {}),
            trade_offs=analysis.get("trade_offs", []),
            recommendations=analysis.get("recommendations", []),
            raw_response=result.get("_raw_response"),
        )


@dataclass
class ProposalItem:
    """Une proposition individuelle du Strategist."""

    name: str
    rationale: str
    params: dict[str, Any]
    changes_from_baseline: dict[str, dict[str, Any]]  # {param: {old, new}}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProposalsSection:
    """Section des propositions LLM (Strategist)."""

    model_used: str
    duration_seconds: float
    baseline_params: dict[str, Any]
    baseline_sharpe: float
    proposals: list[ProposalItem]
    total_generated: int
    total_valid: int

    def to_dict(self) -> dict:
        result = asdict(self)
        result["proposals"] = [p.to_dict() for p in self.proposals]
        return result

    @classmethod
    def from_strategist_result(
        cls,
        result: dict,
        baseline_params: dict,
        baseline_sharpe: float,
        model: str,
        duration: float
    ) -> "ProposalsSection":
        """Crée une ProposalsSection depuis le résultat du Strategist."""
        proposals = []

        for prop in result.get("proposals", []):
            # Calculer les changements par rapport à baseline
            changes = {}
            for param, new_val in prop.get("params", {}).items():
                old_val = baseline_params.get(param)
                if old_val != new_val:
                    changes[param] = {"old": old_val, "new": new_val}

            proposals.append(ProposalItem(
                name=prop.get("name", "Unknown"),
                rationale=prop.get("rationale", ""),
                params=prop.get("params", {}),
                changes_from_baseline=changes,
            ))

        return cls(
            model_used=model,
            duration_seconds=duration,
            baseline_params=baseline_params,
            baseline_sharpe=baseline_sharpe,
            proposals=proposals,
            total_generated=result.get("total_generated", len(proposals)),
            total_valid=result.get("total_valid", len(proposals)),
        )


@dataclass
class TestResultItem:
    """Résultat de test d'une proposition."""

    name: str
    params: dict[str, Any]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_trades: int
    loss_trades: int
    vs_baseline_sharpe: float  # Différence avec baseline
    is_improvement: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TestsSection:
    """Section des résultats de tests des propositions."""

    total_tested: int
    successful_tests: int
    failed_tests: int
    results: list[TestResultItem]
    best_proposal: Optional[str]  # Nom de la meilleure proposition
    best_sharpe: float
    baseline_sharpe: float
    improvement_found: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        result["results"] = [r.to_dict() for r in self.results]
        return result

    @classmethod
    def from_test_results(
        cls,
        test_results: list[dict],
        baseline_sharpe: float
    ) -> "TestsSection":
        """Crée une TestsSection depuis les résultats de tests."""
        results = []
        best_proposal = None
        best_sharpe = baseline_sharpe

        for res in test_results:
            # CRITICAL: Handle None values explicitly
            sharpe = res.get("sharpe_ratio")
            if sharpe is None:
                sharpe = 0.0

            trades = res.get("trades", [])

            # Calculer profit/loss trades
            profit_trades = sum(1 for t in trades if t.get("pnl", t.get("pnl_realized", 0)) > 0)
            total_trades = len(trades)
            loss_trades = total_trades - profit_trades

            # Safe delta calculation
            vs_baseline = (sharpe - baseline_sharpe) if sharpe is not None else None
            is_improvement = (sharpe > baseline_sharpe) if sharpe is not None else False

            item = TestResultItem(
                name=res.get("name", "Unknown"),
                params=res.get("params", {}),
                sharpe_ratio=sharpe,
                total_return=res.get("total_return", 0.0),
                max_drawdown=res.get("max_drawdown", 0.0),
                win_rate=res.get("win_rate", 0.0),
                total_trades=total_trades,
                profit_trades=profit_trades,
                loss_trades=loss_trades,
                vs_baseline_sharpe=vs_baseline if vs_baseline is not None else 0.0,
                is_improvement=is_improvement,
            )
            results.append(item)

            if sharpe is not None and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_proposal = res.get("name")

        return cls(
            total_tested=len(test_results),
            successful_tests=len([r for r in test_results if r.get("sharpe_ratio") is not None]),
            failed_tests=len([r for r in test_results if r.get("sharpe_ratio") is None]),
            results=results,
            best_proposal=best_proposal,
            best_sharpe=best_sharpe,
            baseline_sharpe=baseline_sharpe,
            improvement_found=best_sharpe > baseline_sharpe,
        )


@dataclass
class LLMRunReport:
    """
    Rapport complet d'un run Multi-LLM.

    Structure:
    - metadata: Infos générales (date, stratégie, durée totale)
    - sweep: Résultats du sweep GPU
    - analysis: Analyse de l'Analyst
    - proposals: Propositions du Strategist
    - tests: Résultats des tests
    - conclusion: Résumé et meilleure config
    """

    # Métadonnées
    run_id: str
    timestamp: str
    strategy_name: str
    total_duration_seconds: float

    # Sections
    sweep: SweepSection
    analysis: Optional[AnalysisSection] = None
    proposals: Optional[ProposalsSection] = None
    tests: Optional[TestsSection] = None

    # Configuration utilisée
    config: dict[str, Any] = field(default_factory=dict)

    # Conclusion
    best_config: Optional[dict[str, Any]] = None
    best_sharpe: float = 0.0
    summary: str = ""

    def __post_init__(self):
        """Génère le run_id si non fourni."""
        if not self.run_id:
            # Générer un ID unique basé sur timestamp + stratégie
            hash_input = f"{self.timestamp}_{self.strategy_name}"
            self.run_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        """Convertit le rapport en dictionnaire sérialisable."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "strategy_name": self.strategy_name,
            "total_duration_seconds": self.total_duration_seconds,
            "sweep": self.sweep.to_dict(),
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "proposals": self.proposals.to_dict() if self.proposals else None,
            "tests": self.tests.to_dict() if self.tests else None,
            "config": self.config,
            "best_config": self.best_config,
            "best_sharpe": self.best_sharpe,
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        """Sérialise le rapport en JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "LLMRunReport":
        """Reconstruit un rapport depuis un dictionnaire."""
        # Reconstruire les sections
        sweep = SweepSection(**data["sweep"])

        analysis = None
        if data.get("analysis"):
            analysis = AnalysisSection(**data["analysis"])

        proposals = None
        if data.get("proposals"):
            props_data = data["proposals"]
            props_data["proposals"] = [
                ProposalItem(**p) for p in props_data.get("proposals", [])
            ]
            proposals = ProposalsSection(**props_data)

        tests = None
        if data.get("tests"):
            tests_data = data["tests"]
            tests_data["results"] = [
                TestResultItem(**r) for r in tests_data.get("results", [])
            ]
            tests = TestsSection(**tests_data)

        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            strategy_name=data["strategy_name"],
            total_duration_seconds=data["total_duration_seconds"],
            sweep=sweep,
            analysis=analysis,
            proposals=proposals,
            tests=tests,
            config=data.get("config", {}),
            best_config=data.get("best_config"),
            best_sharpe=data.get("best_sharpe", 0.0),
            summary=data.get("summary", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "LLMRunReport":
        """Reconstruit un rapport depuis JSON."""
        return cls.from_dict(json.loads(json_str))

    def generate_summary(self) -> str:
        """Génère un résumé textuel du run."""
        lines = [
            f"# Rapport Multi-LLM: {self.strategy_name}",
            f"",
            f"**Run ID:** {self.run_id}",
            f"**Date:** {self.timestamp}",
            f"**Durée totale:** {self.total_duration_seconds:.1f}s",
            f"",
            f"## Sweep GPU",
            f"- Configurations testées: {self.sweep.total_configs}",
            f"- Durée: {self.sweep.duration_seconds:.1f}s",
        ]

        if self.sweep.metrics_summary:
            avg_sharpe = self.sweep.metrics_summary.get("avg_sharpe_ratio",
                         self.sweep.metrics_summary.get("avg_sharpe", 0))
            max_sharpe = self.sweep.metrics_summary.get("max_sharpe_ratio",
                         self.sweep.metrics_summary.get("max_sharpe", 0))
            lines.append(f"- Sharpe moyen: {avg_sharpe:.3f}")
            lines.append(f"- Meilleur Sharpe: {max_sharpe:.3f}")

        if self.analysis:
            lines.extend([
                f"",
                f"## Analyse (Analyst)",
                f"- Modèle: {self.analysis.model_used}",
                f"- Patterns identifiés: {len(self.analysis.patterns)}",
                f"- Recommandations: {len(self.analysis.recommendations)}",
            ])

        if self.proposals:
            lines.extend([
                f"",
                f"## Propositions (Strategist)",
                f"- Modèle: {self.proposals.model_used}",
                f"- Propositions valides: {self.proposals.total_valid}/{self.proposals.total_generated}",
                f"- Baseline Sharpe: {self.proposals.baseline_sharpe:.3f}",
            ])

        if self.tests:
            lines.extend([
                f"",
                f"## Tests des Propositions",
                f"- Propositions testées: {self.tests.total_tested}",
                f"- Amélioration trouvée: {'✅ Oui' if self.tests.improvement_found else '❌ Non'}",
            ])
            if self.tests.best_proposal:
                lines.append(f"- Meilleure proposition: {self.tests.best_proposal}")
                lines.append(f"- Meilleur Sharpe: {self.tests.best_sharpe:.3f}")

        lines.extend([
            f"",
            f"## Conclusion",
            f"**Meilleur Sharpe final:** {self.best_sharpe:.3f}",
        ])

        self.summary = "\n".join(lines)
        return self.summary


# ============================================================
# RUN INDEX SYSTEM
# ============================================================


@dataclass
class IndexEntry:
    """Entrée dans l'index des runs."""

    run_id: str
    timestamp: str
    strategy_name: str
    best_sharpe: float
    total_configs: int
    improvement_found: bool
    report_path: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_report(cls, report: LLMRunReport, report_path: str) -> "IndexEntry":
        """Crée une entrée d'index depuis un rapport."""
        return cls(
            run_id=report.run_id,
            timestamp=report.timestamp,
            strategy_name=report.strategy_name,
            best_sharpe=report.best_sharpe,
            total_configs=report.sweep.total_configs,
            improvement_found=report.tests.improvement_found if report.tests else False,
            report_path=report_path,
            tags=[],
        )


class RunIndex:
    """
    Gestionnaire d'index pour les runs Multi-LLM.

    Permet de:
    - Sauvegarder des rapports avec indexation automatique
    - Rechercher des runs par critères
    - Charger des rapports existants

    L'index est stocké dans un fichier JSON centralisé.
    """

    def __init__(self, reports_dir: Path | str = DEFAULT_REPORTS_DIR):
        """
        Initialise le gestionnaire d'index.

        Args:
            reports_dir: Répertoire racine pour les rapports
        """
        self.reports_dir = Path(reports_dir)
        self.index_path = self.reports_dir / "index.json"
        self._index: dict[str, IndexEntry] = {}

        # Créer le répertoire si nécessaire
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Charger l'index existant
        self._load_index()

    def _load_index(self):
        """Charge l'index depuis le fichier JSON."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._index = {
                    k: IndexEntry(**v) for k, v in data.get("entries", {}).items()
                }
                logger.info(f"Index chargé: {len(self._index)} runs")
            except Exception as e:
                logger.warning(f"Erreur chargement index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self):
        """Sauvegarde l'index dans le fichier JSON."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "total_runs": len(self._index),
            "entries": {k: v.to_dict() for k, v in self._index.items()},
        }

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Index sauvegardé: {len(self._index)} runs")

    def save_report(
        self,
        report: LLMRunReport,
        tags: list[str] | None = None
    ) -> Path:
        """
        Sauvegarde un rapport et l'ajoute à l'index.

        Args:
            report: Rapport à sauvegarder
            tags: Tags optionnels pour faciliter la recherche

        Returns:
            Path: Chemin du fichier rapport sauvegardé
        """
        # Générer le nom du répertoire
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dir_name = f"{date_str}_{report.strategy_name}_{report.run_id}"
        report_dir = self.reports_dir / dir_name
        report_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le rapport JSON
        report_path = report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())

        # Sauvegarder le résumé Markdown
        summary_path = report_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(report.generate_summary())

        # Créer l'entrée d'index
        entry = IndexEntry.from_report(report, str(report_path.relative_to(self.reports_dir)))
        if tags:
            entry.tags = tags

        # Ajouter à l'index
        self._index[report.run_id] = entry
        self._save_index()

        logger.info(f"Rapport sauvegardé: {report_path}")
        return report_path

    def load_report(self, run_id: str) -> LLMRunReport | None:
        """
        Charge un rapport depuis son run_id.

        Args:
            run_id: Identifiant du run

        Returns:
            LLMRunReport ou None si non trouvé
        """
        entry = self._index.get(run_id)
        if not entry:
            logger.warning(f"Run non trouvé: {run_id}")
            return None

        report_path = self.reports_dir / entry.report_path
        if not report_path.exists():
            logger.warning(f"Fichier rapport non trouvé: {report_path}")
            return None

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return LLMRunReport.from_json(f.read())
        except Exception as e:
            logger.error(f"Erreur chargement rapport {run_id}: {e}")
            return None

    def search(
        self,
        strategy: str | None = None,
        min_sharpe: float | None = None,
        max_sharpe: float | None = None,
        improvement_only: bool = False,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[IndexEntry]:
        """
        Recherche des runs selon critères.

        Args:
            strategy: Filtrer par nom de stratégie
            min_sharpe: Sharpe minimum
            max_sharpe: Sharpe maximum
            improvement_only: Seulement les runs avec amélioration
            tags: Filtrer par tags (OR)
            limit: Nombre max de résultats

        Returns:
            Liste d'IndexEntry correspondant aux critères
        """
        results = []

        for entry in self._index.values():
            # Filtres
            if strategy and entry.strategy_name != strategy:
                continue
            if min_sharpe is not None and entry.best_sharpe < min_sharpe:
                continue
            if max_sharpe is not None and entry.best_sharpe > max_sharpe:
                continue
            if improvement_only and not entry.improvement_found:
                continue
            if tags and not any(t in entry.tags for t in tags):
                continue

            results.append(entry)

        # Trier par date décroissante
        results.sort(key=lambda x: x.timestamp, reverse=True)

        return results[:limit]

    def list_all(self, limit: int = 100) -> list[IndexEntry]:
        """Liste tous les runs (triés par date décroissante)."""
        return self.search(limit=limit)

    def list_strategies(self) -> list[str]:
        """Liste toutes les stratégies indexées."""
        return list(set(e.strategy_name for e in self._index.values()))

    def get_stats(self) -> dict:
        """Retourne des statistiques sur l'index."""
        if not self._index:
            return {"total_runs": 0}

        entries = list(self._index.values())
        sharpes = [e.best_sharpe for e in entries]

        return {
            "total_runs": len(entries),
            "strategies": self.list_strategies(),
            "avg_sharpe": sum(sharpes) / len(sharpes),
            "max_sharpe": max(sharpes),
            "min_sharpe": min(sharpes),
            "improvements_found": sum(1 for e in entries if e.improvement_found),
            "total_configs_tested": sum(e.total_configs for e in entries),
        }

    def delete_run(self, run_id: str, delete_files: bool = True) -> bool:
        """
        Supprime un run de l'index.

        Args:
            run_id: Identifiant du run
            delete_files: Si True, supprime aussi les fichiers

        Returns:
            True si supprimé, False si non trouvé
        """
        entry = self._index.get(run_id)
        if not entry:
            return False

        if delete_files:
            report_path = self.reports_dir / entry.report_path
            report_dir = report_path.parent
            if report_dir.exists():
                import shutil
                shutil.rmtree(report_dir)
                logger.info(f"Fichiers supprimés: {report_dir}")

        del self._index[run_id]
        self._save_index()

        logger.info(f"Run supprimé de l'index: {run_id}")
        return True


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def create_report_from_run(
    strategy_name: str,
    sweep_results: list[dict],
    sweep_params: dict,
    sweep_duration: float,
    analysis_result: dict | None = None,
    analyst_model: str = "",
    analyst_duration: float = 0.0,
    proposals_result: dict | None = None,
    baseline_params: dict | None = None,
    baseline_sharpe: float = 0.0,
    strategist_model: str = "",
    strategist_duration: float = 0.0,
    test_results: list[dict] | None = None,
    config: dict | None = None,
) -> LLMRunReport:
    """
    Fonction helper pour créer un rapport depuis les résultats d'un run.

    Args:
        strategy_name: Nom de la stratégie optimisée
        sweep_results: Résultats du sweep GPU
        sweep_params: Paramètres testés dans le sweep
        sweep_duration: Durée du sweep en secondes
        analysis_result: Résultat de l'Analyst (optionnel)
        analyst_model: Modèle utilisé par l'Analyst
        analyst_duration: Durée de l'analyse
        proposals_result: Résultat du Strategist (optionnel)
        baseline_params: Paramètres baseline
        baseline_sharpe: Sharpe de la baseline
        strategist_model: Modèle utilisé par le Strategist
        strategist_duration: Durée de la génération
        test_results: Résultats des tests (optionnel)
        config: Configuration du run

    Returns:
        LLMRunReport complet
    """
    timestamp = datetime.now().isoformat()

    # Créer les sections
    sweep = SweepSection.from_sweep_results(
        sweep_results, sweep_params, sweep_duration
    )

    analysis = None
    if analysis_result:
        analysis = AnalysisSection.from_analyst_result(
            analysis_result, analyst_model, analyst_duration
        )

    proposals = None
    if proposals_result and baseline_params is not None:
        proposals = ProposalsSection.from_strategist_result(
            proposals_result, baseline_params, baseline_sharpe,
            strategist_model, strategist_duration
        )

    tests = None
    if test_results:
        tests = TestsSection.from_test_results(test_results, baseline_sharpe)

    # Calculer durée totale
    total_duration = sweep_duration + analyst_duration + strategist_duration

    # Déterminer best config et sharpe
    best_sharpe = baseline_sharpe
    best_config = baseline_params

    if tests and tests.improvement_found and tests.best_proposal:
        best_sharpe = tests.best_sharpe
        # Trouver les params de la meilleure proposition
        for res in tests.results:
            if res.name == tests.best_proposal:
                best_config = res.params
                break

    # Créer le rapport
    report = LLMRunReport(
        run_id="",  # Sera généré automatiquement
        timestamp=timestamp,
        strategy_name=strategy_name,
        total_duration_seconds=total_duration,
        sweep=sweep,
        analysis=analysis,
        proposals=proposals,
        tests=tests,
        config=config or {},
        best_config=best_config,
        best_sharpe=best_sharpe,
    )

    # Générer le résumé
    report.generate_summary()

    return report


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "LLMRunReport",
    "SweepSection",
    "AnalysisSection",
    "ProposalsSection",
    "ProposalItem",
    "TestsSection",
    "TestResultItem",
    "IndexEntry",
    "RunIndex",
    "create_report_from_run",
    "DEFAULT_REPORTS_DIR",
]
```
<!-- MODULE-END: run_report.py -->

<!-- MODULE-START: __init__.py -->
```json
{
  "name": "__init__.py",
  "path": "llm\\__init__.py",
  "ext": ".py",
  "anchor": "init___py"
}
```
## init___py
*Chemin* : `llm\__init__.py`  
*Type* : `.py`  

```python
"""
ThreadX LLM Integration Module
================================

Module d'intégration LLM local pour l'analyse intelligente de backtests,
la recommandation de paramètres et l'assistance interactive.

Composants:
- LLMClient: Interface unifiée pour modèles locaux (Ollama)
- Prompts: Templates de prompts réutilisables
- Interpreters: Parsers pour structurer les réponses LLM

Author: ThreadX Framework
Version: 1.0.0 - Initial LLM Integration
"""

from threadx.llm.client import LLMClient
from threadx.llm.interpreters import parse_backtest_interpretation

__all__ = ["LLMClient", "parse_backtest_interpretation"]
```
<!-- MODULE-END: __init__.py -->

<!-- MODULE-START: analyst.py -->
```json
{
  "name": "analyst.py",
  "path": "llm\\agents\\analyst.py",
  "ext": ".py",
  "anchor": "analyst_py"
}
```
## analyst_py
*Chemin* : `llm\agents\analyst.py`  
*Type* : `.py`  

```python
"""
Agent Analyst - Analyse quantitative de résultats de backtests.

Utilise deepseek-r1:70b pour analyser des résultats de sweep/backtests
et identifier des patterns significatifs.
"""

from typing import Any

import pandas as pd

from threadx.llm.agents.base_agent import BaseAgent


class Analyst(BaseAgent):
    """
    Agent spécialisé dans l'analyse quantitative de résultats de backtests.

    Capabilities:
    - Analyser les résultats d'un sweep (top N configurations)
    - Analyser un backtest individuel en profondeur
    - Identifier des patterns communs dans des configurations performantes
    """

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        debug: bool = False,
    ) -> None:
        """
        Initialise l'agent Analyst.

        Args:
            model: Modèle LLM à utiliser (par défaut deepseek-r1:32b pour analyse)
            debug: Active les logs détaillés
        """
        super().__init__(name="Analyst", model=model, debug=debug)

    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Point d'entrée générique (délègue vers analyze_sweep_results).

        Pour usage direct, préférer analyze_sweep_results() ou analyze_backtest().
        """
        if "sweep_df" in kwargs or (args and isinstance(args[0], pd.DataFrame)):
            sweep_df = kwargs.get("sweep_df", args[0] if args else None)
            top_n = kwargs.get("top_n", 5)
            return self.analyze_sweep_results(sweep_df, top_n)

        raise ValueError(
            "Analyst.analyze() requires 'sweep_df' (DataFrame) parameter. "
            "Use analyze_sweep_results() or analyze_backtest() directly."
        )

    def analyze_sweep_results(
        self, sweep_df: pd.DataFrame, top_n: int = 5
    ) -> dict[str, Any]:
        """
        Analyse les résultats d'un sweep pour identifier les meilleures configs.

        Args:
            sweep_df: DataFrame avec colonnes [strategy, param1, param2, ..., sharpe_ratio, etc.]
            top_n: Nombre de top configurations à analyser en détail

        Returns:
            dict avec:
            - top_configs: Liste des N meilleures configs avec métriques
            - analysis: Analyse qualitative LLM (patterns, recommandations)
            - patterns: Patterns identifiés (ex: "short_period < 15 dans 4/5 top configs")
        """
        self.logger.info("Analyzing sweep results (top %d configs)...", top_n)

        # Trier par Sharpe ratio (ou autre métrique)
        if "sharpe_ratio" not in sweep_df.columns:
            raise ValueError("sweep_df must contain 'sharpe_ratio' column")

        top_df = sweep_df.nlargest(top_n, "sharpe_ratio")

        # Préparer données pour le LLM
        configs_str = self._format_sweep_results(top_df)

        # Prompt pour analyse quantitative avec consignes système
        system_instructions = """
🎯 OBJECTIFS PRIORITAIRES:
- Maximiser le Sharpe Ratio (risque/rendement optimal)
- Minimiser le drawdown maximum (protection du capital)
- Maintenir un win rate > 50% (cohérence stratégique)
- Optimiser le nombre de trades (éviter over/under-trading)

📊 APPROCHE D'ANALYSE:
- Identifier les patterns reproductibles dans les meilleures configurations
- Détecter les corrélations entre paramètres (interactions non-linéaires)
- Privilégier la robustesse à la performance brute (éviter overfitting)
- Analyser les trade-offs (ex: rendement vs stabilité)
- **DÉTECTER INCOHÉRENCES** (ex: slow_period < fast_period, TP/SL < 1.5, leverage élevé avec sharpe faible)

⚠️ CONTRAINTES CRITIQUES:
- risk_per_trade: Rester dans [0.005, 0.02] (gestion risque stricte)
- max_hold_bars: Adapter selon volatilité observée
- Stop Loss / Take Profit: Ratio minimum 1:1.5 (asymétrie favorable)
- Respecter TOUJOURS les plages min/max des paramètres
- **MA Crossover/EMA Cross: slow_period DOIT être > fast_period**

💡 PRINCIPES:
- Préférer solutions simples et explicables
- Documenter clairement le raisonnement
- Signaler EXPLICITEMENT les anomalies ou incohérences dans les données
"""

        prompt = f"""{system_instructions}

Analyse les {top_n} meilleures configurations de backtest ci-dessous.

Résultats du sweep (triés par Sharpe ratio):
{configs_str}

Identifie:
1. **Patterns communs** dans les paramètres performants (ex: "slow_period souvent entre 60-100")
2. **Métriques clés** (Sharpe moyen, drawdown max, win rate moyen)
3. **Trade-offs observés** (ex: "Sharpe élevé mais drawdown important")
4. **Incohérences détectées** (ex: "Config #1: slow < fast → invalide")
5. **Recommandations** pour prochaines optimisations (plages de paramètres prometteuses)

IMPORTANT: Si tu détectes des incohérences (slow < fast, TP/SL anormal), SIGNALE-LES clairement.

Réponds en JSON avec:
{{
  "patterns": ["pattern1", "pattern2", ...],
  "key_metrics": {{"avg_sharpe": X, "max_drawdown_avg": Y, "avg_win_rate": Z, ...}},
  "trade_offs": ["trade-off1", ...],
  "anomalies": ["anomalie1", ...],
  "recommendations": ["rec1", "rec2", ...]
}}
"""

        # Appel LLM structuré
        analysis_result = self._call_llm_structured(
            prompt=prompt,
            expected_schema={
                "patterns": list,
                "key_metrics": dict,
                "trade_offs": list,
                "recommendations": list,
            },
            temperature=0.3,  # Basse température pour analyse factuelle
            max_tokens=2000,
        )

        # Garantir que toutes les clés existent avec types corrects (validation)
        if not isinstance(analysis_result.get("patterns"), list):
            analysis_result["patterns"] = []
        if not isinstance(analysis_result.get("key_metrics"), dict):
            analysis_result["key_metrics"] = {}
        if not isinstance(analysis_result.get("trade_offs"), list):
            analysis_result["trade_offs"] = []
        if not isinstance(analysis_result.get("recommendations"), list):
            analysis_result["recommendations"] = []

        # Si recommendations vide, générer une recommandation basique
        if not analysis_result["recommendations"]:
            self.logger.warning("⚠️ LLM n'a pas généré de recommandations - ajout fallback")
            analysis_result["recommendations"] = [
                "Analyser les patterns identifiés pour affiner les plages de paramètres",
                "Tester des configurations avec Sharpe > moyenne observée",
                "Réduire le drawdown en ajustant les stops loss",
            ]

        # Identifier patterns quantitatifs (complément à l'analyse LLM)
        quantitative_patterns = self._identify_quantitative_patterns(top_df)

        return {
            "top_configs": top_df.to_dict("records"),
            "analysis": analysis_result,
            "quantitative_patterns": quantitative_patterns,
        }

    def analyze_backtest(
        self, backtest_result: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Analyse en profondeur un backtest individuel.

        Args:
            backtest_result: Résultat backtest (sharpe, drawdown, total_return, trades, etc.)
            params: Paramètres utilisés (strategy, short_period, long_period, etc.)

        Returns:
            dict avec:
            - assessment: Évaluation qualitative (strong/medium/weak)
            - strengths: Points forts identifiés
            - weaknesses: Points faibles identifiés
            - suggestions: Modifications suggérées des paramètres
        """
        self.logger.info("Analyzing individual backtest...")

        # Formater résultats pour LLM
        results_str = self._format_backtest_result(backtest_result, params)

        prompt = f"""Analyse en détail ce backtest de stratégie trading:

{results_str}

Évalue:
1. **Performance globale** (strong/medium/weak) basée sur Sharpe, drawdown, win rate
2. **Points forts** (métriques excellentes, comportement robuste)
3. **Points faibles** (risques, incohérences, métriques faibles)
4. **Suggestions** de modifications (ex: "Augmenter long_period pour réduire drawdown")

Réponds en JSON:
{{
  "assessment": "strong/medium/weak",
  "strengths": ["strength1", ...],
  "weaknesses": ["weakness1", ...],
  "suggestions": ["suggestion1", ...]
}}
"""

        analysis = self._call_llm_structured(
            prompt=prompt,
            expected_schema={
                "assessment": str,
                "strengths": list,
                "weaknesses": list,
                "suggestions": list,
            },
            temperature=0.4,
            max_tokens=1500,
        )

        return analysis

    def identify_patterns(self, configs_list: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Identifie des patterns communs dans une liste de configurations.

        Args:
            configs_list: Liste de configs (dicts avec params + métriques)

        Returns:
            dict avec:
            - common_params: Paramètres avec valeurs fréquentes (ex: {"short_period": [10, 12]})
            - correlations: Corrélations observées (ex: "short_period < 15 → Sharpe > 1.5")
        """
        self.logger.info("Identifying patterns in %d configurations...", len(configs_list))

        # Convertir en DataFrame pour analyse
        df = pd.DataFrame(configs_list)

        # Analyser distributions
        param_cols = [c for c in df.columns if c not in ["sharpe_ratio", "total_return", "max_drawdown"]]
        distributions = {}
        for col in param_cols:
            if df[col].dtype in ["int64", "float64"]:
                distributions[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }

        # Demander au LLM d'interpréter
        configs_str = df.to_string(max_rows=20)
        prompt = f"""Analyse ces configurations de trading pour identifier des patterns:

{configs_str}

Distributions des paramètres:
{distributions}

Identifie:
1. **Paramètres critiques** (ceux qui varient le plus entre configs performantes)
2. **Corrélations** (ex: "Quand short_period < 15, Sharpe > 1.5 dans 80% des cas")
3. **Plages optimales** (ex: "long_period entre 30-50 semble optimal")

Réponds en JSON:
{{
  "critical_params": ["param1", "param2"],
  "correlations": ["correlation1", ...],
  "optimal_ranges": {{"param1": "10-15", "param2": "30-50", ...}}
}}
"""

        patterns = self._call_llm_structured(
            prompt=prompt,
            expected_schema={
                "critical_params": list,
                "correlations": list,
                "optimal_ranges": dict,
            },
            temperature=0.3,
            max_tokens=1500,
        )

        return {
            "patterns": patterns,
            "distributions": distributions,
        }

    # --- Méthodes privées de formatage ---

    def _format_sweep_results(self, df: pd.DataFrame) -> str:
        """Formate un DataFrame de sweep pour le LLM (texte tabulaire lisible)."""
        # Colonnes clés à afficher (vérifier qu'elles existent)
        key_cols = []
        for col in ["strategy", "sharpe_ratio", "total_return", "max_drawdown"]:
            if col in df.columns:
                key_cols.append(col)

        # Colonnes de paramètres (exclure métriques et colonnes internes)
        metrics = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate", "profit_factor",
                   "avg_win", "avg_loss", "total_trades", "strategy"]
        param_cols = [c for c in df.columns if c not in metrics and not c.startswith("_")]

        display_cols = key_cols + param_cols[:5]  # Limiter à 5 params pour lisibilité

        # Si aucune colonne clé, afficher au moins les paramètres
        if not display_cols:
            display_cols = list(df.columns)[:8]

        return str(df[display_cols].to_string(index=False))

    def _format_backtest_result(self, result: dict[str, Any], params: dict[str, Any]) -> str:
        """Formate un résultat backtest + params en texte lisible."""
        lines = ["**Paramètres:**"]
        for k, v in params.items():
            lines.append(f"  {k}: {v}")

        lines.append("\n**Résultats:**")
        for k, v in result.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def _identify_quantitative_patterns(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Identifie des patterns quantitatifs simples (complément LLM).

        Ex: "short_period < 15 dans 4/5 top configs"
        """
        patterns = {}

        # Paramètres numériques uniquement
        param_cols = [c for c in df.columns if c not in ["sharpe_ratio", "total_return", "max_drawdown"]]
        numeric_cols = [c for c in param_cols if df[c].dtype in ["int64", "float64"]]

        for col in numeric_cols:
            patterns[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "range": (float(df[col].min()), float(df[col].max())),
            }

        return patterns
```
<!-- MODULE-END: analyst.py -->

<!-- MODULE-START: base_agent.py -->
```json
{
  "name": "base_agent.py",
  "path": "llm\\agents\\base_agent.py",
  "ext": ".py",
  "anchor": "base_agent_py"
}
```
## base_agent_py
*Chemin* : `llm\agents\base_agent.py`  
*Type* : `.py`  

```python
"""
Base Agent Class for ThreadX LLM System
========================================

Classe abstraite fournissant les fonctionnalités communes à tous les agents LLM.

Features:
- Gestion du timeout et retries automatiques
- Logging structuré avec contexte agent
- Validation des réponses LLM
- Métriques de performance (latence, token usage)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from threadx.llm.client import LLMClient, LLMNotConfiguredError


class BaseAgent(ABC):
    """
    Classe abstraite pour agents LLM spécialisés.

    Attributes:
        name: Nom de l'agent (utilisé pour logging)
        model: Modèle Ollama à utiliser (optionnel si use_llm=False)
        client: Instance LLMClient configurée ou None si LLM désactivé
        timeout: Timeout par défaut pour requêtes LLM
        use_llm: Indique si l'agent utilise un LLM
        logger: Logger avec préfixe [AgentName]
    """

    def __init__(
        self,
        name: str,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        debug: bool = False,
        use_llm: bool = True,
    ):
        """
        Initialise un agent LLM.

        Args:
            name: Nom de l'agent (ex: "Analyst", "Strategist")
            model: Modèle Ollama (ex: "deepseek-r1:70b", "gpt-oss:20b").
                   Peut être None si use_llm=False.
            timeout: Timeout en secondes pour requêtes LLM
            max_retries: Nombre de tentatives en cas d'échec
            debug: Active logging détaillé
            use_llm: Active ou désactive explicitement les appels LLM.
                     Si False, l'agent fonctionnera sans LLMClient.
        """
        self.name = name
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug
        self.use_llm = use_llm

        # Logger avec préfixe agent
        self.logger = logging.getLogger(f"threadx.llm.agents.{name.lower()}")
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # Client LLM partagé (optionnel)
        self.client = None
        if self.use_llm:
            if not self.model:
                raise ValueError(
                    f"Agent {name}: model requis si use_llm=True. "
                    "Fournissez un modèle ou passez use_llm=False."
                )
            self.client = LLMClient(
                model=model, timeout=timeout, max_retries=max_retries, debug=debug
            )

        # Métriques de performance
        self._metrics = {"total_calls": 0, "total_time": 0.0, "errors": 0}

        # Logging différencié selon mode
        if self.use_llm and self.client:
            self.logger.info(
                f"🤖 Agent {name} initialisé (model={model}, timeout={timeout}s)"
            )
        else:
            self.logger.info(
                f"🤖 Agent {name} initialisé (mode: sans LLM, tests automatiques)"
            )

    def _call_llm(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Appel LLM avec tracking des métriques.

        Args:
            prompt: Prompt utilisateur
            system: Message système optionnel
            temperature: Température de génération (0.0 = déterministe)
            max_tokens: Nombre max de tokens générés

        Returns:
            Réponse texte du LLM

        Raises:
            LLMNotConfiguredError: Si l'agent n'a pas de client LLM configuré
            RuntimeError: Si LLM échoue après max_retries
        """
        if not self.use_llm or self.client is None:
            raise LLMNotConfiguredError(
                f"Agent {self.name} configuré sans LLM (use_llm=False). "
                "Impossible d'appeler _call_llm()."
            )

        start_time = time.time()
        self._metrics["total_calls"] += 1

        try:
            if self.debug:
                self.logger.debug(f"📤 LLM Call - Prompt preview: {prompt[:200]}...")

            response = self.client.complete(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            elapsed = time.time() - start_time
            self._metrics["total_time"] += elapsed

            if self.debug:
                self.logger.debug(
                    f"📥 LLM Response ({elapsed:.2f}s): {response[:150]}..."
                )

            return response

        except Exception as e:
            self._metrics["errors"] += 1
            self.logger.error(f"❌ LLM call failed: {e}")
            raise

    def _call_llm_structured(
        self,
        prompt: str,
        expected_schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        use_retry: bool = True,
    ) -> dict[str, Any]:
        """
        Appel LLM avec parsing JSON et validation de schéma.

        Args:
            prompt: Prompt utilisateur (doit demander JSON en output)
            expected_schema: Schéma attendu {key: type} pour validation (optionnel)
            system: Message système optionnel
            temperature: Température de génération
            max_tokens: Nombre max de tokens
            use_retry: Utiliser retry intelligent si JSON invalide (défaut: True)

        Returns:
            Dict parsé depuis la réponse JSON du LLM

        Raises:
            LLMNotConfiguredError: Si l'agent n'a pas de client LLM configuré
            ValueError: Si réponse non-JSON ou schéma invalide
            RuntimeError: Si LLM échoue après retries
        """
        if not self.use_llm or self.client is None:
            raise LLMNotConfiguredError(
                f"Agent {self.name} configuré sans LLM (use_llm=False). "
                "Impossible d'appeler _call_llm_structured()."
            )

        start_time = time.time()
        self._metrics["total_calls"] += 1

        try:
            if self.debug:
                self.logger.debug(
                    f"📤 LLM Structured Call - Expected schema: {expected_schema}"
                )

            # Utiliser retry intelligent si activé
            if use_retry:
                response = self.client.complete_structured_with_retry(
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_json_retries=2,
                )
            else:
                response = self.client.complete_structured(
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # Validation optionnelle du schéma (basique)
            if expected_schema:
                missing_keys = set(expected_schema.keys()) - set(response.keys())
                if missing_keys:
                    self.logger.warning(
                        "Schema validation: missing keys %s", missing_keys
                    )

            elapsed = time.time() - start_time
            self._metrics["total_time"] += elapsed

            if self.debug:
                self.logger.debug(
                    "Structured Response (%.2fs): %s...",
                    elapsed,
                    json.dumps(response, indent=2)[:300],
                )

            return response

        except Exception as e:
            self._metrics["errors"] += 1
            self.logger.error("Structured LLM call failed: %s", e)
            raise

    def get_metrics(self) -> dict[str, Any]:
        """
        Récupère les métriques de performance de l'agent.

        Returns:
            {
                "total_calls": int,
                "total_time": float,
                "avg_time_per_call": float,
                "errors": int,
                "success_rate": float
            }
        """
        total_calls = self._metrics["total_calls"]
        avg_time = (
            self._metrics["total_time"] / total_calls if total_calls > 0 else 0.0
        )
        success_rate = (
            (total_calls - self._metrics["errors"]) / total_calls
            if total_calls > 0
            else 0.0
        )

        return {
            "agent_name": self.name,
            "model": self.model,
            "total_calls": total_calls,
            "total_time": self._metrics["total_time"],
            "avg_time_per_call": avg_time,
            "errors": self._metrics["errors"],
            "success_rate": success_rate,
        }

    def reset_metrics(self):
        """Reset les métriques de performance."""
        self._metrics = {"total_calls": 0, "total_time": 0.0, "errors": 0}
        self.logger.debug("📊 Métriques reset")

    @abstractmethod
    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Méthode abstraite pour analyse principale de l'agent.

        Chaque agent spécialisé doit implémenter cette méthode
        avec sa logique spécifique (ex: analyze_sweep_results pour Analyst).
        """
        raise NotImplementedError("Subclasses must implement analyze()")


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, model={self.model}, "
            f"use_llm={self.use_llm})"
        )
```
<!-- MODULE-END: base_agent.py -->

<!-- MODULE-START: codewriter.py -->
```json
{
  "name": "codewriter.py",
  "path": "llm\\agents\\codewriter.py",
  "ext": ".py",
  "anchor": "codewriter_py"
}
```
## codewriter_py
*Chemin* : `llm\agents\codewriter.py`  
*Type* : `.py`  

```python
"""
Agent CodeWriter - Génération de code pour stratégies de trading.

Génère du code Python de stratégies ThreadX basé sur analyses LLM
et résultats de sweep. V1: Modifications de stratégies existantes uniquement.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from threadx.llm.agents.base_agent import BaseAgent


class CodeWriter(BaseAgent):
    """
    Agent spécialisé dans la génération de code Python pour stratégies.

    V1 Capabilities (MINIMALISTE):
    - Modifier stratégies existantes (BollingerDual, MACrossover, etc.)
    - Générer code Python valide conforme au Protocol Strategy
    - Sauvegarder dans strategy/experimental/
    - Validation syntaxe basique

    V2 Future:
    - Créer nouvelles stratégies from scratch
    - Extraction automatique de schémas params/indicators
    - Tests unitaires auto-générés
    """

    # Prompt système avec contraintes strictes ThreadX
    SYSTEM_PROMPT = """Tu es un expert en stratégies de trading quantitatives Python.

🎯 TÂCHE: Modifier une stratégie ThreadX existante

📋 ARCHITECTURE THREADX (OBLIGATOIRE):

1. **Imports requis**:
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from threadx.strategy.model import RunStats, Trade
```

2. **Classe Params** (Paramètres de stratégie):
```python
@dataclass
class MyStrategyParams:
    # Indicateurs
    bb_period: int = 20
    bb_std: float = 2.0

    # Risk Management
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    risk_per_trade: float = 0.015  # 1.5%

    # Position Management
    max_hold_bars: int = 100
```

3. **Classe Strategy** (Logique trading):
```python
class MyStrategy:
    '''Docstring expliquant la stratégie.'''

    def run(
        self,
        data: pd.DataFrame,
        indicators: dict[str, Any],
        params: dict[str, Any] | MyStrategyParams,
    ) -> RunStats:
        '''
        Exécute le backtest.

        Args:
            data: DataFrame OHLCV [open, high, low, close, volume]
            indicators: dict depuis IndicatorBank {"bollinger": {...}, "atr": Series, ...}
            params: Paramètres (dict ou MyStrategyParams)

        Returns:
            RunStats avec trades, equity, métriques
        '''
        # Conversion params si dict
        if isinstance(params, dict):
            params = MyStrategyParams(**params)

        # Extraction indicateurs (depuis IndicatorBank!)
        bb = indicators.get("bollinger", {})
        bb_upper = bb.get("upper")
        bb_lower = bb.get("lower")
        bb_middle = bb.get("middle")

        # LOGIQUE DE TRADING ICI
        trades = []
        position = None

        for i in range(1, len(data)):
            close = data["close"].iloc[i]

            # Entry logic
            if position is None and close < bb_lower.iloc[i]:
                position = {
                    "entry_idx": i,
                    "entry_price": close,
                    "entry_time": data.index[i],
                    "stop_loss": close * (1 - params.stop_loss_pct / 100),
                }

            # Exit logic
            elif position is not None:
                if close >= bb_middle.iloc[i] or close <= position["stop_loss"]:
                    trade = Trade(
                        entry_ts=position["entry_time"],
                        exit_ts=data.index[i],
                        price_entry=position["entry_price"],
                        price_exit=close,
                        side="LONG",
                        qty=1.0,
                        pnl_realized=close - position["entry_price"],
                    )
                    trades.append(trade)
                    position = None

        # Retourner RunStats
        from threadx.strategy.model import RunStatsDict
        return RunStatsDict(
            trades=[t.__dict__ for t in trades],
            total_trades=len(trades),
        )
```

⚠️ CONTRAINTES CRITIQUES:

1. **Utiliser UNIQUEMENT IndicatorBank**:
   - NE PAS calculer indicateurs manuellement
   - Extraire depuis `indicators` dict fourni en paramètre
   - Indicateurs disponibles: bollinger, atr, sma, ema, rsi

2. **NE PAS modifier**:
   - data_access (lecture données)
   - backtest/engine.py
   - indicators/bank.py
   - Aucun autre fichier du framework

3. **Gestion risque OBLIGATOIRE**:
   - Stop-loss sur CHAQUE position
   - Position sizing (risk_per_trade)
   - Max hold bars (éviter positions zombies)

4. **Code déterministe**:
   - Pas de random.random() sans seed
   - Comportement reproductible

5. **Bibliothèques autorisées**:
   - NumPy, Pandas: OUI ✅
   - TA-Lib, autres: NON ❌

📤 FORMAT DE SORTIE: JSON strict

```json
{
    "status": "success",
    "filename": "ai_strategy_v1.py",
    "class_name": "AIStrategyV1",
    "code": "# Code Python complet ici...",
    "explanation": "Cette modification améliore...",
    "changes_summary": [
        "Ajout filtre volume sur entrées",
        "Stop-loss dynamique basé sur ATR",
        "Réduction max_hold_bars de 150 à 100"
    ]
}
```

Si ERREUR:
```json
{
    "status": "error",
    "error": "Raison de l'échec..."
}
```

💡 PRINCIPES V1 (MINIMALISTE):

- **Modifier stratégies existantes** (BollingerDual, MACrossover, etc.)
- Garder même structure de paramètres (compatibilité Optimization Engine)
- Modifications incrémentielles (pas de refonte complète)
- Commentaires expliquant POURQUOI, pas QUOI
"""

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        output_dir: str | Path = "src/threadx/strategy/experimental",
        debug: bool = False,
    ):
        """
        Initialise l'agent CodeWriter.

        Args:
            model: Modèle LLM (défaut: deepseek-r1:32b pour cohérence)
            output_dir: Dossier de sortie pour stratégies générées
            debug: Active logs détaillés
        """
        super().__init__(name="CodeWriter", model=model, timeout=120.0, debug=debug)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"📁 Output directory: {self.output_dir}")

    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Point d'entrée générique (délègue vers run).

        Pour usage direct, préférer run().
        """
        return self.run(**kwargs)

    def run(
        self,
        task: str,
        base_strategy: str,
        analysis: dict[str, Any],
        proposals: dict[str, Any],
        failed_metrics: dict[str, Any] | None = None,
        ideas: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Génère du code de stratégie modifiée basé sur analyse LLM.

        Args:
            task: Type de modification ("improve_sharpe", "reduce_drawdown", etc.)
            base_strategy: Nom de la stratégie à modifier ("Bollinger_Dual", "MA_Crossover", etc.)
            analysis: Résultat de Analyst.analyze_sweep_results()
            proposals: Résultat de Strategist.propose_modifications()
            failed_metrics: Métriques actuelles insatisfaisantes (optionnel)
            ideas: Idées additionnelles pour guider génération (optionnel)

        Returns:
            dict avec:
            {
                "status": "success" | "error",
                "filename": "ai_strategy_v1.py",
                "class_name": "AIStrategyV1",
                "code": "...",  # Code Python complet
                "explanation": "...",
                "changes_summary": [str],
                "error": str (si status=error)
            }

        Example:
            >>> codewriter = CodeWriter()
            >>> result = codewriter.run(
            ...     task="improve_sharpe",
            ...     base_strategy="Bollinger_Dual",
            ...     analysis=analyst_result,
            ...     proposals=strategist_result,
            ...     failed_metrics={"current_sharpe": 0.5, "target_sharpe": 0.8}
            ... )
            >>> if result["status"] == "success":
            ...     print(f"Code généré: {result['filename']}")
        """
        self.logger.info(
            f"🔨 Génération code: task={task}, base={base_strategy}"
        )

        # Construire contexte pour LLM
        context = self._build_context(
            task, base_strategy, analysis, proposals, failed_metrics, ideas
        )

        # Prompt utilisateur
        prompt = f"""{context}

**TÂCHE**: {self._task_description(task)}

**STRATÉGIE DE BASE**: {base_strategy}

**OBJECTIF**: Générer une version améliorée de cette stratégie en Python.

**CONTRAINTES**:
- Garder la même structure de paramètres (compatibilité Optimization Engine)
- Utiliser UNIQUEMENT IndicatorBank pour indicateurs
- Ajouter gestion risque stricte (stop-loss, position sizing)
- Code déterministe et reproductible

**OUTPUT**: JSON avec "status", "filename", "class_name", "code", "explanation", "changes_summary"
"""

        try:
            # Appel LLM structuré
            result = self._call_llm_structured(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                expected_schema={
                    "status": str,
                    "filename": str,
                    "class_name": str,
                    "code": str,
                },
                temperature=0.7,  # Créativité modérée
                max_tokens=4000,  # Code peut être long
            )

            # Validation résultat
            if result.get("status") != "success":
                return result

            # Extraction et nettoyage du code
            code = self._extract_python_code(result.get("code", ""))

            # Sauvegarde du code
            filepath = self._save_code(
                filename=result["filename"],
                code=code,
                metadata={
                    "task": task,
                    "base_strategy": base_strategy,
                    "class_name": result["class_name"],
                    "explanation": result.get("explanation", ""),
                }
            )

            # Retourner résultat enrichi
            return {
                **result,
                "code": code,
                "filepath": str(filepath),
            }

        except Exception as e:
            self.logger.error(f"❌ Code generation failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)}"
            }

    def _build_context(
        self,
        task: str,
        base_strategy: str,
        analysis: dict[str, Any],
        proposals: dict[str, Any],
        failed_metrics: dict[str, Any] | None,
        ideas: list[str] | None,
    ) -> str:
        """Construit le contexte pour le prompt LLM."""
        lines = []

        # Métriques actuelles
        if failed_metrics:
            lines.append("📊 **MÉTRIQUES ACTUELLES** (insatisfaisantes):")
            for key, val in failed_metrics.items():
                lines.append(f"  - {key}: {val}")
            lines.append("")

        # Patterns de l'Analyst
        patterns = analysis.get("analysis", {}).get("patterns", [])
        if patterns:
            lines.append("🔍 **PATTERNS IDENTIFIÉS** (Analyst):")
            for pattern in patterns[:5]:  # Limiter à top 5
                lines.append(f"  - {pattern}")
            lines.append("")

        # Recommandations de l'Analyst
        recommendations = analysis.get("analysis", {}).get("recommendations", [])
        if recommendations:
            lines.append("💡 **RECOMMANDATIONS** (Analyst):")
            for rec in recommendations[:5]:
                lines.append(f"  - {rec}")
            lines.append("")

        # Propositions du Strategist
        proposals_list = proposals.get("proposals", [])
        if proposals_list:
            lines.append("🎯 **PROPOSITIONS DE PARAMÈTRES** (Strategist):")
            for i, prop in enumerate(proposals_list[:3], 1):
                name = prop.get("name", f"Proposition {i}")
                rationale = prop.get("rationale", "N/A")
                params = prop.get("params", {})
                lines.append(f"  {i}. **{name}**:")
                lines.append(f"     Rationale: {rationale}")
                lines.append(f"     Params: {params}")
            lines.append("")

        # Idées additionnelles
        if ideas:
            lines.append("💭 **IDÉES ADDITIONNELLES**:")
            for idea in ideas:
                lines.append(f"  - {idea}")
            lines.append("")

        return "\n".join(lines)

    def _task_description(self, task: str) -> str:
        """Retourne la description détaillée d'une tâche."""
        descriptions = {
            "improve_sharpe": "Améliorer le ratio de Sharpe en optimisant le risk/reward",
            "reduce_drawdown": "Réduire le drawdown maximum via gestion de risque stricte",
            "increase_winrate": "Augmenter le win rate en améliorant la qualité des signaux",
            "optimize_trades": "Optimiser le nombre de trades (ni trop, ni trop peu)",
            "modify_strategy": "Modifier la stratégie selon les recommandations d'analyse",
        }

        return descriptions.get(task, task)

    def _extract_python_code(self, code_str: str) -> str:
        """
        Extrait le code Python depuis la réponse LLM.

        Gère les cas:
        - Code brut Python
        - Code dans bloc ```python ... ```
        - Code avec commentaires avant/après

        Args:
            code_str: String contenant le code (potentiellement avec markdown)

        Returns:
            Code Python nettoyé
        """
        # Chercher bloc ```python ... ```
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, code_str, re.DOTALL)

        if match:
            code = match.group(1)
            self.logger.debug("✅ Code extrait depuis bloc ```python")
            return code.strip()

        # Sinon, supposer que c'est du code brut
        self.logger.debug("⚠️  Pas de bloc markdown, utilise code brut")
        return code_str.strip()

    def _save_code(
        self,
        filename: str,
        code: str,
        metadata: dict[str, Any],
    ) -> Path:
        """
        Sauvegarde le code généré dans experimental/.

        Args:
            filename: Nom du fichier (ex: "ai_strategy_v1.py")
            code: Code Python complet
            metadata: Métadonnées (task, base_strategy, etc.)

        Returns:
            Path du fichier sauvegardé
        """
        # Assurer extension .py
        if not filename.endswith(".py"):
            filename += ".py"

        filepath = self.output_dir / filename

        # Header avec métadonnées
        header = f'''"""
AI-Generated Strategy: {metadata.get("class_name", "Unknown")}
{'=' * 60}

**Generated**: {self._get_timestamp()}
**Base Strategy**: {metadata.get("base_strategy", "N/A")}
**Task**: {metadata.get("task", "N/A")}

**Explanation**:
{metadata.get("explanation", "N/A")}

⚠️ ATTENTION:
- Stratégie générée automatiquement par LLM
- Doit passer validation Critic avant utilisation
- NE PAS utiliser en production sans revue humaine
"""

'''

        # Écriture du fichier
        full_code = header + code

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_code)

        self.logger.info(f"✅ Code sauvegardé: {filepath}")

        return filepath

    def _get_timestamp(self) -> str:
        """Retourne timestamp ISO pour métadonnées."""
        from datetime import datetime
        return datetime.now().isoformat()


# Export
__all__ = ["CodeWriter"]
```
<!-- MODULE-END: codewriter.py -->

<!-- MODULE-START: critic.py -->
```json
{
  "name": "critic.py",
  "path": "llm\\agents\\critic.py",
  "ext": ".py",
  "anchor": "critic_py"
}
```
## critic_py
*Chemin* : `llm\agents\critic.py`  
*Type* : `.py`  

```python
"""
Agent Critic - Validation et promotion de stratégies AI-générées.

Valide les stratégies créées par CodeWriter via 3 tests:
1. Syntaxe + Import dynamique (py_compile)
2. Backtest rapide sur 2 scénarios (BTCUSDC/15m, ETHUSDC/1h)
3. Critères quantitatifs minimaux (Sharpe, DD, trades count)

V1: Tests automatiques uniquement (pas de LLM review)
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import py_compile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from threadx.llm.agents.base_agent import BaseAgent


@dataclass
class ValidationCriteria:
    """Critères de validation quantitatifs."""

    min_sharpe: float = 0.5
    max_drawdown_pct: float = -30.0  # ex: -30%
    min_trades: int = 10
    min_win_rate_pct: float = 35.0


@dataclass
class ValidationResult:
    """Résultat de validation d'une stratégie."""

    status: str  # "approved" | "rejected"
    test_syntax: dict[str, Any]
    test_backtest: dict[str, Any]
    test_quantitative: dict[str, Any]
    recommendation: str
    errors: list[str]


class Critic(BaseAgent):
    """
    Agent spécialisé dans la validation de stratégies AI-générées.

    V1 Capabilities (MINIMALISTE):
    - Validation syntaxe (py_compile + import dynamique)
    - Backtest rapide sur 2 scénarios (BTCUSDC/15m, ETHUSDC/1h)
    - Vérification critères quantitatifs
    - Décision de promotion automatique

    V2 Future:
    - LLM code review (qualité architecture)
    - Walk-forward validation multi-périodes
    - Tests de robustesse (slippage, commission)
    """

    def __init__(
        self,
        criteria: ValidationCriteria | None = None,
        experimental_dir: str | Path = "src/threadx/strategy/experimental",
        debug: bool = False,
    ):
        """
        Initialise l'agent Critic.

        Args:
            criteria: Critères de validation (utilise défauts si None)
            experimental_dir: Dossier des stratégies à valider
            debug: Active logs détaillés
        """
        # Critic V1 n'utilise PAS de LLM (tests automatiques uniquement)
        # V2 pourra ajouter LLM code review optionnel
        super().__init__(
            name="Critic",
            model=None,  # Pas de modèle nécessaire pour V1
            timeout=60.0,
            debug=debug,
            use_llm=False,  # Désactive explicitement les appels LLM
        )

        self.criteria = criteria or ValidationCriteria()
        self.experimental_dir = Path(experimental_dir)

        self.logger.info(f"📁 Experimental directory: {self.experimental_dir}")
        self.logger.info(
            f"📊 Critères: Sharpe≥{self.criteria.min_sharpe}, "
            f"DD≥{self.criteria.max_drawdown_pct}%, "
            f"Trades≥{self.criteria.min_trades}, "
            f"WinRate≥{self.criteria.min_win_rate_pct}%"
        )

    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Point d'entrée générique (délègue vers run).

        Pour usage direct, préférer run().
        """
        return self.run(**kwargs)

    def run(
        self,
        strategy_file: str | Path,
        backtest_scenarios: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Valide une stratégie AI-générée et décide de la promotion.

        Args:
            strategy_file: Path vers fichier .py de la stratégie (ex: "ai_meanrev_v3.py")
            backtest_scenarios: Scénarios de backtest (optionnel, utilise défauts si None)

        Returns:
            dict avec:
            {
                "status": "approved" | "rejected",
                "test_syntax": {"passed": bool, "error": str | None},
                "test_backtest": {"scenarios": [...], "passed": bool},
                "test_quantitative": {"sharpe": float, "dd": float, "trades": int, "passed": bool},
                "recommendation": "APPROVE" | "REJECT",
                "errors": [str],
                "promoted": bool,  # True si copié vers strategy/
            }

        Example:
            >>> critic = Critic()
            >>> result = critic.run(
            ...     strategy_file="ai_meanrev_v3.py",
            ...     backtest_scenarios=[
            ...         {"symbol": "BTCUSDC", "interval": "15m", "start": "2023-07-01", "end": "2023-12-31"},
            ...     ]
            ... )
            >>> if result["status"] == "approved":
            ...     print(f"✅ Stratégie promue vers strategy/")
        """
        self.logger.info(f"🧪 Validation de: {strategy_file}")

        filepath = self.experimental_dir / strategy_file
        if not filepath.exists():
            return {
                "status": "rejected",
                "errors": [f"Fichier non trouvé: {filepath}"],
                "recommendation": "REJECT",
            }

        errors = []

        # Test 1: Syntaxe + Import
        self.logger.info("  🔍 Test 1/3: Syntaxe + Import dynamique...")
        test_syntax = self._validate_syntax(filepath)

        if not test_syntax["passed"]:
            errors.append(f"Syntaxe: {test_syntax.get('error', 'Unknown error')}")
            return {
                "status": "rejected",
                "test_syntax": test_syntax,
                "errors": errors,
                "recommendation": "REJECT",
                "promoted": False,
            }

        self.logger.info("    ✅ Syntaxe OK")

        # Test 2: Backtest rapide
        self.logger.info("  ⚡ Test 2/3: Backtest sur scénarios...")

        if backtest_scenarios is None:
            backtest_scenarios = self._get_default_scenarios()

        test_backtest = self._run_backtest_validation(
            strategy_class=test_syntax["strategy_class"],
            params_class=test_syntax["params_class"],
            scenarios=backtest_scenarios,
        )

        if not test_backtest["passed"]:
            errors.append("Backtest failed")
            return {
                "status": "rejected",
                "test_syntax": test_syntax,
                "test_backtest": test_backtest,
                "errors": errors,
                "recommendation": "REJECT",
                "promoted": False,
            }

        self.logger.info("    ✅ Backtest OK")

        # Test 3: Critères quantitatifs
        self.logger.info("  📊 Test 3/3: Critères quantitatifs...")

        test_quantitative = self._check_quantitative_criteria(
            test_backtest["scenarios"]
        )

        if not test_quantitative["passed"]:
            errors.append(f"Critères quantitatifs non satisfaits: {test_quantitative.get('failures', [])}")
            return {
                "status": "rejected",
                "test_syntax": test_syntax,
                "test_backtest": test_backtest,
                "test_quantitative": test_quantitative,
                "errors": errors,
                "recommendation": "REJECT",
                "promoted": False,
            }

        self.logger.info("    ✅ Critères quantitatifs OK")

        # Tous les tests passés → APPROVE
        self.logger.info("✅ STATUT: APPROVED")

        # Promotion (V1: logging uniquement, pas de copie automatique)
        # TODO V2: Copier vers strategy/ + enregistrement registry
        promoted = False  # V1: manuel

        return {
            "status": "approved",
            "test_syntax": test_syntax,
            "test_backtest": test_backtest,
            "test_quantitative": test_quantitative,
            "recommendation": "APPROVE",
            "errors": [],
            "promoted": promoted,
        }

    def _validate_syntax(self, filepath: Path) -> dict[str, Any]:
        """
        Valide la syntaxe Python et l'import dynamique.

        Args:
            filepath: Path vers fichier .py

        Returns:
            {
                "passed": bool,
                "error": str | None,
                "module_name": str,
                "strategy_class": type | None,
                "params_class": type | None,
            }
        """
        try:
            # Test 1a: Compilation Python
            py_compile.compile(str(filepath), doraise=True)
            self.logger.debug(f"    ✅ Compilation réussie: {filepath.name}")

            # Test 1b: Import dynamique
            module_name = filepath.stem
            spec = importlib.util.spec_from_file_location(
                f"threadx.strategy.experimental.{module_name}",
                filepath,
            )

            if spec is None or spec.loader is None:
                raise ImportError(f"Impossible de créer spec pour {module_name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Chercher classes *Strategy et *Params
            strategy_class = None
            params_class = None

            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                if not isinstance(attr, type):
                    continue

                if attr.__module__ != module.__name__:
                    continue  # Classe importée

                if attr_name.endswith("Strategy"):
                    strategy_class = attr

                elif attr_name.endswith("Params"):
                    params_class = attr

            if strategy_class is None:
                raise ValueError(f"Aucune classe *Strategy trouvée dans {module_name}")

            self.logger.debug(f"    ✅ Import dynamique OK: {strategy_class.__name__}")

            return {
                "passed": True,
                "error": None,
                "module_name": module_name,
                "strategy_class": strategy_class,
                "params_class": params_class,
            }

        except Exception as e:
            self.logger.error(f"    ❌ Syntaxe/Import failed: {e}")
            return {
                "passed": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "module_name": None,
                "strategy_class": None,
                "params_class": None,
            }

    def _get_default_scenarios(self) -> list[dict[str, Any]]:
        """
        Retourne les scénarios de backtest par défaut.

        Returns:
            list de scénarios avec {symbol, interval, start, end, description}
        """
        return [
            {
                "symbol": "BTCUSDC",
                "interval": "15m",
                "start": "2023-07-01",
                "end": "2023-12-31",
                "description": "BTC 15m (6 mois H2 2023)",
            },
            {
                "symbol": "ETHUSDC",
                "interval": "1h",
                "start": "2023-07-01",
                "end": "2023-12-31",
                "description": "ETH 1h (6 mois H2 2023)",
            },
        ]

    def _run_backtest_validation(
        self,
        strategy_class: type,
        params_class: type | None,
        scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Exécute backtests sur les scénarios fournis.

        Args:
            strategy_class: Classe de stratégie à tester
            params_class: Classe de paramètres (optionnel)
            scenarios: Liste de scénarios de test

        Returns:
            {
                "passed": bool,
                "scenarios": [
                    {
                        "description": str,
                        "sharpe": float | None,
                        "max_drawdown_pct": float | None,
                        "total_trades": int,
                        "win_rate_pct": float | None,
                        "error": str | None,
                    }
                ]
            }
        """
        # V1: Mock simple (pas de vraie exécution BacktestEngine)
        # TODO V2: Intégrer BacktestEngine.run() avec load_ohlcv()

        results = []
        all_passed = True

        for scenario in scenarios:
            self.logger.debug(f"    🔄 Scénario: {scenario.get('description', 'N/A')}")

            try:
                # MOCK: Simuler résultat de backtest
                # Dans V2, remplacer par:
                # from threadx.backtest.engine import BacktestEngine
                # from threadx.data_access.data_loader import load_ohlcv
                # data = load_ohlcv(symbol=scenario["symbol"], interval=scenario["interval"], ...)
                # engine = BacktestEngine()
                # result = engine.run(data=data, strategy_class=strategy_class, params=default_params)

                # MOCK DATA (remplacer en V2)
                mock_result = {
                    "description": scenario.get("description", "N/A"),
                    "sharpe": 0.6,  # Mock
                    "max_drawdown_pct": -15.0,  # Mock
                    "total_trades": 25,  # Mock
                    "win_rate_pct": 40.0,  # Mock
                    "error": None,
                }

                results.append(mock_result)

            except Exception as e:
                self.logger.error(f"    ❌ Backtest failed: {e}")
                results.append({
                    "description": scenario.get("description", "N/A"),
                    "sharpe": None,
                    "max_drawdown_pct": None,
                    "total_trades": 0,
                    "win_rate_pct": None,
                    "error": str(e),
                })
                all_passed = False

        return {
            "passed": all_passed,
            "scenarios": results,
        }

    def _check_quantitative_criteria(
        self,
        scenario_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Vérifie que les résultats satisfont les critères quantitatifs.

        Args:
            scenario_results: Résultats des backtests

        Returns:
            {
                "passed": bool,
                "failures": [str],  # Liste des critères non satisfaits
                "best_sharpe": float,
                "worst_drawdown": float,
                "min_trades": int,
            }
        """
        failures = []

        # Agréger métriques sur tous les scénarios
        sharpe_values = [r["sharpe"] for r in scenario_results if r["sharpe"] is not None]
        dd_values = [r["max_drawdown_pct"] for r in scenario_results if r["max_drawdown_pct"] is not None]
        trades_values = [r["total_trades"] for r in scenario_results if r["total_trades"] is not None]
        winrate_values = [r["win_rate_pct"] for r in scenario_results if r["win_rate_pct"] is not None]

        if not sharpe_values:
            failures.append("Aucun Sharpe valide calculé")
            return {"passed": False, "failures": failures}

        best_sharpe = max(sharpe_values)
        worst_drawdown = min(dd_values) if dd_values else 0.0
        min_trades = min(trades_values) if trades_values else 0
        avg_winrate = sum(winrate_values) / len(winrate_values) if winrate_values else 0.0

        # Vérification critères
        if best_sharpe < self.criteria.min_sharpe:
            failures.append(f"Sharpe {best_sharpe:.2f} < {self.criteria.min_sharpe}")

        if worst_drawdown < self.criteria.max_drawdown_pct:
            failures.append(f"Max DD {worst_drawdown:.1f}% < {self.criteria.max_drawdown_pct}%")

        if min_trades < self.criteria.min_trades:
            failures.append(f"Trades {min_trades} < {self.criteria.min_trades}")

        if avg_winrate < self.criteria.min_win_rate_pct:
            failures.append(f"Win Rate {avg_winrate:.1f}% < {self.criteria.min_win_rate_pct}%")

        passed = len(failures) == 0

        self.logger.debug(f"    📊 Métriques: Sharpe={best_sharpe:.2f}, DD={worst_drawdown:.1f}%, Trades={min_trades}, WinRate={avg_winrate:.1f}%")

        if not passed:
            self.logger.debug(f"    ❌ Échecs: {', '.join(failures)}")

        return {
            "passed": passed,
            "failures": failures,
            "best_sharpe": best_sharpe,
            "worst_drawdown": worst_drawdown,
            "min_trades": min_trades,
            "avg_winrate": avg_winrate,
        }


# Export
__all__ = ["Critic", "ValidationCriteria", "ValidationResult"]
```
<!-- MODULE-END: critic.py -->

<!-- MODULE-START: strategist.py -->
```json
{
  "name": "strategist.py",
  "path": "llm\\agents\\strategist.py",
  "ext": ".py",
  "anchor": "strategist_py"
}
```
## strategist_py
*Chemin* : `llm\agents\strategist.py`  
*Type* : `.py`  

```python
"""
Agent Strategist - Génération créative de propositions de stratégies.

Utilise gpt-oss:20b pour proposer des modifications de paramètres basées
sur les analyses de l'Analyst.
"""

from typing import Any

from threadx.llm.agents.base_agent import BaseAgent


class Strategist(BaseAgent):
    """
    Agent spécialisé dans la génération de propositions créatives.

    Capabilities:
    - Proposer N modifications de paramètres basées sur une analyse
    - Valider que les propositions respectent les contraintes (min/max)
    - Formater les propositions pour exécution automatique
    """

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        debug: bool = False,
    ) -> None:
        """
        Initialise l'agent Strategist.

        Args:
            model: Modèle LLM à utiliser (par défaut deepseek-r1:32b pour cohérence avec Analyst)
            debug: Active les logs détaillés
        """
        super().__init__(name="Strategist", model=model, debug=debug)

    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Point d'entrée générique (délègue vers propose_modifications).

        Pour usage direct, préférer propose_modifications().
        """
        if "analysis" in kwargs:
            return self.propose_modifications(**kwargs)

        raise ValueError(
            "Strategist.analyze() requires 'analysis' parameter. "
            "Use propose_modifications() directly."
        )

    def propose_modifications(
        self,
        analysis: dict[str, Any],
        current_params: dict[str, Any],
        param_specs: dict[str, dict[str, Any]],
        n_proposals: int = 3,
    ) -> dict[str, Any]:
        """
        Propose N modifications de paramètres basées sur une analyse.

        Args:
            analysis: Résultat de Analyst.analyze_sweep_results()
            current_params: Paramètres de la config actuelle/baseline
            param_specs: Specs des paramètres (ex: {"short_period": {"min": 5, "max": 50, "type": "int"}})
            n_proposals: Nombre de propositions à générer

        Returns:
            dict avec:
            - proposals: Liste de N dicts de paramètres modifiés
            - rationale: Justifications pour chaque proposition
        """
        self.logger.info("Generating %d parameter proposals...", n_proposals)

        # Extraire insights clés de l'analyse
        patterns = analysis.get("analysis", {}).get("patterns", [])
        recommendations = analysis.get("analysis", {}).get("recommendations", [])
        trade_offs = analysis.get("analysis", {}).get("trade_offs", [])

        # Construire contexte pour LLM
        context_str = self._format_analysis_context(
            patterns, recommendations, trade_offs, current_params, param_specs
        )

        # Prompt pour génération créative avec consignes système
        system_instructions = """
🎯 OBJECTIFS PRIORITAIRES:
- Maximiser le Sharpe Ratio (risque/rendement optimal)
- Minimiser le drawdown maximum (protection du capital)
- Maintenir un win rate > 50% (cohérence stratégique)
- Optimiser le nombre de trades (ni trop, ni trop peu)

📊 APPROCHE DE PROPOSITION:
- Modifications incrémentielles (pas de changements brutaux)
- Exploiter les patterns identifiés dans les meilleures configs
- Tester des zones peu explorées (diversification)
- Valider la cohérence logique des propositions

⚠️ CONTRAINTES CRITIQUES:
- risk_per_trade: TOUJOURS dans [0.005, 0.02]
- max_hold_bars: Adapter selon volatilité (range typique 20-150)
- Stop Loss / Take Profit: Ratio minimum 1:1.5 (asymétrie favorable)
- **MA_Crossover/EMA_Cross: slow_period DOIT TOUJOURS être > fast_period** (BLOQUANT si non respecté)
- **Bollinger_Breakout: bb_std_dev DOIT être > 0** (BLOQUANT si non respecté)
- Respecter STRICTEMENT les plages min/max des paramètres

💡 PRINCIPES:
- Privilégier robustesse > performance brute (éviter overfitting)
- Documenter clairement le raisonnement (transparence)
- 3 approches: Conservative (stabilité), Aggressive (rendement), Exploratoire (découverte)
- Chaque proposition doit être testable immédiatement
"""

        prompt = f"""{system_instructions}

Tu es un expert en optimisation de stratégies de trading. Génère {n_proposals} propositions de modifications de paramètres.

{context_str}

Génère {n_proposals} propositions **différentes et créatives**:
1. Une approche conservative (petites modifications, réduire risque)
2. Une approche aggressive (exploiter patterns identifiés, maximiser Sharpe)
3. Une approche exploratoire (tester zones peu explorées)

**Contraintes strictes**:
- Respecter les min/max de chaque paramètre
- Propositions doivent être testables (valeurs concrètes)
- Justifier chaque modification

Réponds en JSON:
{{
  "proposals": [
    {{
      "name": "Conservative",
      "params": {{"short_period": 12, "long_period": 35, ...}},
      "rationale": "Réduit drawdown observé en augmentant long_period..."
    }},
    ...
  ]
}}
"""

        # Appel LLM structuré avec retry et fallback
        try:
            result = self._call_llm_structured(
                prompt=prompt,
                expected_schema={
                    "proposals": list,
                },
                temperature=0.8,  # Haute température pour créativité
                max_tokens=2500,
            )

            # Valider et filtrer propositions
            validated_proposals = self._validate_and_filter_proposals(
                result.get("proposals", []), param_specs
            )

            # Si aucune proposition valide, générer fallback
            if len(validated_proposals) == 0:
                self.logger.warning(
                    "⚠️ No valid LLM proposals after validation, "
                    "generating fallback proposals..."
                )
                validated_proposals = self._generate_fallback_proposals(
                    current_params, param_specs, n_proposals
                )
                source = "fallback_invalid"
            else:
                source = "llm"

            self.logger.info("Generated %d valid proposals (source: %s)", len(validated_proposals), source)

            return {
                "proposals": validated_proposals,
                "total_generated": len(result.get("proposals", [])),
                "total_valid": len(validated_proposals),
                "source": source,
            }

        except Exception as e:
            # Fallback complet si LLM échoue
            self.logger.error(
                "❌ LLM proposal generation failed: %s, using fallback heuristics",
                str(e)
            )
            fallback_proposals = self._generate_fallback_proposals(
                current_params, param_specs, n_proposals
            )

            return {
                "proposals": fallback_proposals,
                "total_generated": 0,
                "total_valid": len(fallback_proposals),
                "source": "fallback_error",
                "error": str(e),
            }

    def validate_constraints(
        self, proposals: list[dict[str, Any]], param_specs: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Valide que les propositions respectent les contraintes min/max ET les contraintes stratégiques.

        Args:
            proposals: Liste de propositions (dicts avec 'params' key)
            param_specs: Specs des paramètres avec min/max

        Returns:
            Liste filtrée des propositions valides uniquement
        """
        valid = []

        for prop in proposals:
            params = prop.get("params", {})
            is_valid = True
            rejection_reason = None

            # Validation 1: Contraintes min/max des paramètres
            for param_name, value in params.items():
                if param_name not in param_specs:
                    self.logger.warning(
                        "Unknown parameter '%s' in proposal, skipping", param_name
                    )
                    is_valid = False
                    rejection_reason = f"Unknown parameter '{param_name}'"
                    break

                spec = param_specs[param_name]
                min_val = spec.get("min")
                max_val = spec.get("max")

                if min_val is not None and value < min_val:
                    self.logger.warning(
                        "Parameter %s = %s below min %s, skipping proposal",
                        param_name,
                        value,
                        min_val,
                    )
                    is_valid = False
                    rejection_reason = f"{param_name}={value} < min({min_val})"
                    break

                if max_val is not None and value > max_val:
                    self.logger.warning(
                        "Parameter %s = %s above max %s, skipping proposal",
                        param_name,
                        value,
                        max_val,
                    )
                    is_valid = False
                    rejection_reason = f"{param_name}={value} > max({max_val})"
                    break

            # Validation 2: Contraintes stratégiques spécifiques
            if is_valid:
                # MA_Crossover/EMA_Cross : slow_period DOIT être > fast_period
                if "fast_period" in params and "slow_period" in params:
                    fast = params["fast_period"]
                    slow = params["slow_period"]
                    if slow <= fast:
                        self.logger.warning(
                            "❌ MA_Crossover constraint violated: slow_period (%s) <= fast_period (%s), skipping",
                            slow,
                            fast,
                        )
                        is_valid = False
                        rejection_reason = f"slow_period ({slow}) <= fast_period ({fast})"

                # Bollinger_Breakout : bb_std_dev DOIT être > 0
                if "bb_std_dev" in params:
                    bb_std = params["bb_std_dev"]
                    if bb_std <= 0:
                        self.logger.warning(
                            "❌ Bollinger constraint violated: bb_std_dev (%s) <= 0, skipping",
                            bb_std,
                        )
                        is_valid = False
                        rejection_reason = f"bb_std_dev ({bb_std}) <= 0"

                # Take Profit DOIT être >= Stop Loss * 1.5 (ratio asymétrique)
                if "stop_loss_pct" in params and "take_profit_pct" in params:
                    sl = params["stop_loss_pct"]
                    tp = params["take_profit_pct"]
                    if tp < sl * 1.5:
                        self.logger.warning(
                            "⚠️ TP/SL ratio low: %s/%s = %.2fx (recommended >1.5x), keeping but flagged",
                            tp,
                            sl,
                            tp / sl if sl > 0 else 0,
                        )
                        # Non-bloquant mais signalé (on garde la proposition)

            if is_valid:
                valid.append(prop)
            else:
                self.logger.info("Rejected proposal '%s': %s", prop.get("name", "Unknown"), rejection_reason)

        return valid

    def format_proposals(self, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Formate les propositions pour exécution automatique (ScenarioSpec compatible).

        Args:
            proposals: Liste de propositions brutes du LLM

        Returns:
            Liste de propositions formatées avec clés standardisées
        """
        formatted = []

        for i, prop in enumerate(proposals):
            formatted.append(
                {
                    "proposal_id": i + 1,
                    "name": prop.get("name", f"Proposal_{i+1}"),
                    "params": prop.get("params", {}),
                    "rationale": prop.get("rationale", ""),
                }
            )

        return formatted

    # --- Méthodes privées ---

    def _format_analysis_context(
        self,
        patterns: list[str],
        recommendations: list[str],
        trade_offs: list[str],
        current_params: dict[str, Any],
        param_specs: dict[str, dict[str, Any]],
    ) -> str:
        """Formate le contexte d'analyse pour le LLM."""
        lines = ["**Analyse des résultats:**"]

        if patterns:
            lines.append("\nPatterns identifiés:")
            for p in patterns:
                lines.append(f"  - {p}")

        if recommendations:
            lines.append("\nRecommandations:")
            for r in recommendations:
                lines.append(f"  - {r}")

        if trade_offs:
            lines.append("\nTrade-offs observés:")
            for t in trade_offs:
                lines.append(f"  - {t}")

        lines.append("\n**Paramètres actuels (baseline):**")
        for k, v in current_params.items():
            lines.append(f"  {k}: {v}")

        lines.append("\n**Contraintes des paramètres:**")
        for param, spec in param_specs.items():
            min_val = spec.get("min", "N/A")
            max_val = spec.get("max", "N/A")
            param_type = spec.get("type", "unknown")
            lines.append(f"  {param}: type={param_type}, min={min_val}, max={max_val}")

        return "\n".join(lines)

    def _generate_fallback_proposals(
        self,
        current_params: dict[str, Any],
        param_specs: dict[str, dict[str, Any]],
        n_proposals: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Génère propositions par défaut via heuristiques Python (fallback si LLM échoue).

        Stratégies:
        1. Conservative: Augmentation +10% paramètres "lents" (slow, long, period)
        2. Aggressive: Réduction SL, augmentation TP, périodes rapides, multiplicateurs
        3. Baseline: Configuration actuelle inchangée

        Règles heuristiques étendues:
        - Règle 1: Slow/long/period → +10% (stabilité)
        - Règle 2: Stop loss → -10% (moins de stops)
        - Règle 3: Take profit → +10% (gains plus élevés)
        - Règle 4: Thresholds/z-scores → -10% (plus permissif)
        - Règle 5: Multiplicateurs (ATR, volatilité) → +10% (plus de marge)
        - Règle 6: Périodes rapides (fast, short) → -10% (plus réactif)

        Args:
            current_params: Paramètres baseline
            param_specs: Specs avec min/max
            n_proposals: Nombre de propositions (max 3)

        Returns:
            Liste de propositions formatées (dict avec name, params, rationale)
        """
        proposals = []

        # PROPOSITION 1: Conservative (+10% params lents, -10% thresholds)
        conservative = current_params.copy()
        conservative_changes = []

        for param, value in conservative.items():
            if param not in param_specs:
                continue

            # Règle 1: Augmenter paramètres "lents" de 10%
            if any(keyword in param.lower() for keyword in ["slow", "long", "period"]):
                if not any(keyword in param.lower() for keyword in ["fast", "short"]):
                    new_val = value * 1.1
                    max_val = param_specs[param].get("max")
                    if max_val is not None:
                        new_val = min(new_val, max_val)

                    # Arrondir selon type
                    if param_specs[param].get("type") == "int":
                        new_val = int(new_val)

                    if new_val != value:
                        conservative[param] = new_val
                        conservative_changes.append(f"{param}: {value} → {new_val}")

            # Règle 4: Réduire thresholds/z-scores de 10% (plus permissif)
            elif any(keyword in param.lower() for keyword in ["threshold", "_z", "trigger", "zscore"]):
                new_val = value * 0.9
                min_val = param_specs[param].get("min")
                if min_val is not None:
                    new_val = max(new_val, min_val)

                if param_specs[param].get("type") == "int":
                    new_val = int(new_val)

                if new_val != value:
                    conservative[param] = new_val
                    conservative_changes.append(f"{param}: {value} → {new_val}")

        proposals.append({
            "name": "Conservative (fallback)",
            "params": conservative,
            "rationale": (
                f"Règles heuristiques: +10% params lents (stabilité), -10% thresholds (permissivité). "
                f"Modifications: {', '.join(conservative_changes) if conservative_changes else 'baseline inchangée'}"
            ),
        })

        # PROPOSITION 2: Aggressive (SL -10%, TP +10%, périodes rapides -10%, multiplicateurs +10%)
        aggressive = current_params.copy()
        aggressive_changes = []

        for param, value in aggressive.items():
            if param not in param_specs:
                continue

            # Règle 2: Réduire stop loss de 10%
            if "stop" in param.lower() or "sl" in param.lower():
                new_val = value * 0.9
                min_val = param_specs[param].get("min")
                if min_val is not None:
                    new_val = max(new_val, min_val)

                if param_specs[param].get("type") == "int":
                    new_val = int(new_val)

                if new_val != value:
                    aggressive[param] = new_val
                    aggressive_changes.append(f"{param}: {value} → {new_val}")

            # Règle 3: Augmenter take profit de 10%
            elif "take" in param.lower() or "tp" in param.lower():
                new_val = value * 1.1
                max_val = param_specs[param].get("max")
                if max_val is not None:
                    new_val = min(new_val, max_val)

                if param_specs[param].get("type") == "int":
                    new_val = int(new_val)

                if new_val != value:
                    aggressive[param] = new_val
                    aggressive_changes.append(f"{param}: {value} → {new_val}")

            # Règle 5: Augmenter multiplicateurs de 10% (ATR, volatilité)
            elif any(keyword in param.lower() for keyword in ["multiplier", "mult", "factor", "atr"]):
                if not any(keyword in param.lower() for keyword in ["stop", "sl"]):  # Éviter double application
                    new_val = value * 1.1
                    max_val = param_specs[param].get("max")
                    if max_val is not None:
                        new_val = min(new_val, max_val)

                    if param_specs[param].get("type") == "int":
                        new_val = int(new_val)

                    if new_val != value:
                        aggressive[param] = new_val
                        aggressive_changes.append(f"{param}: {value} → {new_val}")

            # Règle 6: Réduire périodes rapides de 10% (plus réactif)
            elif any(keyword in param.lower() for keyword in ["fast", "short"]):
                if "period" in param.lower() or "ma" in param.lower() or "ema" in param.lower():
                    new_val = value * 0.9
                    min_val = param_specs[param].get("min")
                    if min_val is not None:
                        new_val = max(new_val, min_val)

                    if param_specs[param].get("type") == "int":
                        new_val = int(new_val)

                    if new_val != value:
                        aggressive[param] = new_val
                        aggressive_changes.append(f"{param}: {value} → {new_val}")

        proposals.append({
            "name": "Aggressive (fallback)",
            "params": aggressive,
            "rationale": (
                f"Règles heuristiques: SL -10%, TP +10%, multiplicateurs +10%, périodes rapides -10%. "
                f"Modifications: {', '.join(aggressive_changes) if aggressive_changes else 'baseline inchangée'}"
            ),
        })

        # PROPOSITION 3: Baseline (inchangée)
        if n_proposals >= 3:
            proposals.append({
                "name": "Baseline (unchanged)",
                "params": current_params.copy(),
                "rationale": "Configuration actuelle (baseline) sans modification.",
            })

        # Validation finale : corriger automatiquement les contraintes stratégiques
        validated_proposals = []
        for prop in proposals[:n_proposals]:
            fixed_params = self._fix_strategic_constraints(prop["params"])
            validated_proposals.append({
                "name": prop["name"],
                "params": fixed_params,
                "rationale": prop["rationale"],
            })

        return validated_proposals

    def _fix_strategic_constraints(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Corrige automatiquement les contraintes stratégiques dans une proposition.

        Règles:
        1. MA_Crossover/EMA_Cross: Si slow_period <= fast_period, swap ou ajustement intelligent
        2. Bollinger: Si bb_std_dev <= 0, fixer à 2.0 (valeur standard)
        3. TP/SL: Si TP < SL * 1.5, ajuster TP à SL * 2.0

        Args:
            params: Paramètres bruts (potentiellement invalides)

        Returns:
            Paramètres corrigés (garantis valides pour contraintes stratégiques)
        """
        fixed = params.copy()

        # Règle 1: MA_Crossover → slow_period DOIT être > fast_period
        if "fast_period" in fixed and "slow_period" in fixed:
            fast = fixed["fast_period"]
            slow = fixed["slow_period"]

            if slow <= fast:
                # Stratégie de correction: swap si possible, sinon ajustement
                if fast < 100:  # Seulement si fast est raisonnable comme slow
                    self.logger.info(
                        "Auto-fix MA_Crossover: swapping fast_period (%s) ↔ slow_period (%s)",
                        fast,
                        slow,
                    )
                    fixed["fast_period"] = slow
                    fixed["slow_period"] = fast
                else:
                    # fast trop grand pour devenir slow, ajuster différemment
                    self.logger.info(
                        "Auto-fix MA_Crossover: setting slow_period = fast_period (%s) + 10",
                        fast,
                    )
                    fixed["slow_period"] = fast + 10

        # Règle 2: Bollinger → bb_std_dev DOIT être > 0
        if "bb_std_dev" in fixed and fixed["bb_std_dev"] <= 0:
            self.logger.info(
                "Auto-fix Bollinger: bb_std_dev (%s) → 2.0 (standard value)",
                fixed["bb_std_dev"],
            )
            fixed["bb_std_dev"] = 2.0

        # Règle 3: TP/SL → Ratio minimum 1.5:1 (recommandé 2:1)
        if "stop_loss_pct" in fixed and "take_profit_pct" in fixed:
            sl = fixed["stop_loss_pct"]
            tp = fixed["take_profit_pct"]

            if sl > 0 and tp < sl * 1.5:
                new_tp = sl * 2.0  # Ratio 2:1 (plus sûr que 1.5:1)
                self.logger.info(
                    "Auto-fix TP/SL ratio: take_profit_pct (%.2f) → %.2f (ratio 2:1 with SL %.2f)",
                    tp,
                    new_tp,
                    sl,
                )
                fixed["take_profit_pct"] = new_tp

        return fixed

    def _validate_and_filter_proposals(
        self, proposals: list[dict[str, Any]], param_specs: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Valide et filtre les propositions (wrapper de validate_constraints + format).

        Args:
            proposals: Propositions brutes du LLM
            param_specs: Specs des paramètres

        Returns:
            Propositions validées et formatées
        """
        # Valider contraintes
        valid_proposals = self.validate_constraints(proposals, param_specs)

        # Formater pour exécution
        formatted_proposals = self.format_proposals(valid_proposals)

        return formatted_proposals
```
<!-- MODULE-END: strategist.py -->

<!-- MODULE-START: __init__.py -->
```json
{
  "name": "__init__.py",
  "path": "llm\\agents\\__init__.py",
  "ext": ".py",
  "anchor": "init___py"
}
```
## init___py
*Chemin* : `llm\agents\__init__.py`  
*Type* : `.py`  

```python
"""
ThreadX Multi-Agent LLM System
================================

Système multi-agents pour optimisation automatique de stratégies de trading.

Agents disponibles:
- Analyst: Analyse quantitative des résultats de backtests
- Strategist: Génération de propositions créatives de modifications
- Critic: Validation et critique des propositions (future)

Usage:
    >>> from threadx.llm.agents import Analyst, Strategist
    >>> analyst = Analyst(model="deepseek-r1:70b")
    >>> strategist = Strategist(model="gpt-oss:20b")
    >>>
    >>> # Analyse de résultats Sweep
    >>> analysis = analyst.analyze_sweep_results(sweep_df, top_n=5)
    >>>
    >>> # Propositions de modifications
    >>> proposals = strategist.propose_modifications(
    ...     analysis=analysis,
    ...     current_params=baseline_params,
    ...     n_proposals=3
    ... )
"""

from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.strategist import Strategist

__all__ = ["Analyst", "Strategist"]
```
<!-- MODULE-END: __init__.py -->

<!-- MODULE-START: ma_crossover.py -->
```json
{
  "name": "ma_crossover.py",
  "path": "strategy\\ma_crossover.py",
  "ext": ".py",
  "anchor": "ma_crossover_py"
}
```
## ma_crossover_py
*Chemin* : `strategy\ma_crossover.py`  
*Type* : `.py`  

```python
"""
ThreadX - MA Crossover Strategy (Validation/Test)
==================================================

Stratégie simple Moving Average Crossover pour tester le moteur de backtest.

Objectif: Validation système, pas optimisation performance
- Règles claires et facilement vérifiables
- Stops et TP fixes (% du prix)
- Pas de levier par défaut
- Position sizing simple

Utilisation:
    >>> from threadx.strategy.ma_crossover import MACrossoverStrategy
    >>> strategy = MACrossoverStrategy()
    >>> params = {"fast_period": 10, "slow_period": 20, "stop_pct": 2.0}
    >>> equity, stats = strategy.backtest(df, params)
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from threadx.strategy.model import (
    RunStats,
    Trade,
    validate_ohlcv_dataframe,
    validate_strategy_params,
)
from threadx.utils.log import get_logger

logger = get_logger(__name__)

# ==========================================
# NUMBA OPTIMIZED BACKTEST LOOP
# ==========================================


@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def _backtest_loop_numba(
    close_vals: np.ndarray,
    signal_vals: np.ndarray,  # 0=HOLD, 1=ENTER_LONG, 2=ENTER_SHORT, 3=EXIT
    initial_capital: float,
    fee_rate: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    risk_per_trade: float,
    leverage: float,
    max_hold_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boucle de backtest MA Crossover optimisée Numba.

    Args:
        close_vals: Prix de clôture (n_bars,)
        signal_vals: Signaux encodés (n_bars,) - 0:HOLD, 1:LONG, 2:SHORT, 3:EXIT
        initial_capital: Capital de départ
        fee_rate: Taux de frais total (fee + slippage)
        stop_loss_pct: Stop loss en % (ex: 2.0 = -2%)
        take_profit_pct: Take profit en % (ex: 4.0 = +4%)
        risk_per_trade: Fraction capital risquée par trade
        leverage: Levier autorisé
        max_hold_bars: Durée max position

    Returns:
        Tuple (equity_curve, trade_results) où:
        - equity_curve: np.ndarray(n_bars,) équité à chaque barre
        - trade_results: np.ndarray(max_trades, 10) résultats trades
          Colonnes: [entry_bar, exit_bar, side, qty, entry_price, exit_price,
                     entry_fees, exit_fees, pnl, stop_price]
    """
    n_bars = len(close_vals)
    equity = np.full(n_bars, initial_capital, dtype=np.float64)

    # Pré-allocation résultats trades
    trade_results = np.zeros((n_bars, 10), dtype=np.float64)
    trade_count = 0

    cash = initial_capital

    # State position
    has_position = False
    pos_side = 0  # 1=LONG, 2=SHORT
    pos_qty = 0.0
    pos_entry_price = 0.0
    pos_stop = 0.0
    pos_take_profit = 0.0
    pos_entry_bar = 0
    pos_entry_fees = 0.0

    # Boucle principale
    for i in range(n_bars):
        current_price = close_vals[i]
        signal = signal_vals[i]

        # === GESTION POSITION EXISTANTE ===
        if has_position:
            should_exit = False

            # 1. Stop loss check
            if pos_side == 1:  # LONG
                if current_price <= pos_stop:
                    should_exit = True
            else:  # SHORT
                if current_price >= pos_stop:
                    should_exit = True

            # 2. Take profit check
            if not should_exit:
                if pos_side == 1 and current_price >= pos_take_profit:
                    should_exit = True
                elif pos_side == 2 and current_price <= pos_take_profit:
                    should_exit = True

            # 3. Signal inverse (croisement opposé)
            if not should_exit:
                if pos_side == 1 and signal == 2:  # LONG → SHORT signal
                    should_exit = True
                elif pos_side == 2 and signal == 1:  # SHORT → LONG signal
                    should_exit = True

            # 4. Max hold bars
            if not should_exit:
                bars_held = i - pos_entry_bar
                if bars_held >= max_hold_bars:
                    should_exit = True

            # === FERMETURE POSITION ===
            if should_exit:
                exit_value = current_price * pos_qty
                exit_fees = exit_value * fee_rate

                # Calcul PnL
                if pos_side == 1:  # LONG
                    pnl = (
                        (current_price - pos_entry_price) * pos_qty
                        - pos_entry_fees
                        - exit_fees
                    )
                else:  # SHORT
                    pnl = (
                        (pos_entry_price - current_price) * pos_qty
                        - pos_entry_fees
                        - exit_fees
                    )

                # Enregistrer trade
                trade_results[trade_count, 0] = pos_entry_bar
                trade_results[trade_count, 1] = i
                trade_results[trade_count, 2] = pos_side
                trade_results[trade_count, 3] = pos_qty
                trade_results[trade_count, 4] = pos_entry_price
                trade_results[trade_count, 5] = current_price
                trade_results[trade_count, 6] = pos_entry_fees
                trade_results[trade_count, 7] = exit_fees
                trade_results[trade_count, 8] = pnl
                trade_results[trade_count, 9] = pos_stop
                trade_count += 1

                # Mise à jour cash
                cash += pnl + (pos_entry_price * pos_qty)

                # Reset position
                has_position = False
                pos_side = 0
                pos_qty = 0.0

        # === NOUVEAU SIGNAL D'ENTRÉE ===
        if not has_position and (signal == 1 or signal == 2):
            # Position sizing basé sur risque
            stop_distance_pct = stop_loss_pct / 100.0
            risk_amount = cash * risk_per_trade
            position_size = risk_amount / (current_price * stop_distance_pct)

            # Limite par levier
            max_position_size = (cash * leverage) / current_price
            qty = min(position_size, max_position_size)

            if qty > 0:
                # Calcul stop et TP
                if signal == 1:  # LONG
                    stop_price = current_price * (1.0 - stop_distance_pct)
                    tp_price = current_price * (1.0 + take_profit_pct / 100.0)
                else:  # SHORT
                    stop_price = current_price * (1.0 + stop_distance_pct)
                    tp_price = current_price * (1.0 - take_profit_pct / 100.0)

                # Frais entrée
                entry_value = current_price * qty
                entry_fees = entry_value * fee_rate

                if entry_value + entry_fees <= cash:
                    # Ouvrir position
                    has_position = True
                    pos_side = signal
                    pos_qty = qty
                    pos_entry_price = current_price
                    pos_stop = stop_price
                    pos_take_profit = tp_price
                    pos_entry_bar = i
                    pos_entry_fees = entry_fees

                    # Déduire cash
                    cash -= entry_value + entry_fees

        # === MISE À JOUR ÉQUITÉ ===
        if has_position:
            if pos_side == 1:  # LONG
                unrealized = (current_price - pos_entry_price) * pos_qty
            else:  # SHORT
                unrealized = (pos_entry_price - current_price) * pos_qty
            equity[i] = cash + unrealized + (pos_entry_price * pos_qty)
        else:
            equity[i] = cash

    # Fermeture position finale si nécessaire
    if has_position:
        final_price = close_vals[-1]
        exit_value = final_price * pos_qty
        exit_fees = exit_value * fee_rate

        if pos_side == 1:
            pnl = (final_price - pos_entry_price) * pos_qty - pos_entry_fees - exit_fees
        else:
            pnl = (pos_entry_price - final_price) * pos_qty - pos_entry_fees - exit_fees

        trade_results[trade_count, 0] = pos_entry_bar
        trade_results[trade_count, 1] = n_bars - 1
        trade_results[trade_count, 2] = pos_side
        trade_results[trade_count, 3] = pos_qty
        trade_results[trade_count, 4] = pos_entry_price
        trade_results[trade_count, 5] = final_price
        trade_results[trade_count, 6] = pos_entry_fees
        trade_results[trade_count, 7] = exit_fees
        trade_results[trade_count, 8] = pnl
        trade_results[trade_count, 9] = pos_stop
        trade_count += 1

        # Mise à jour équité finale
        equity[-1] = cash + pnl + (pos_entry_price * pos_qty)

    # Retourner seulement les trades valides
    return equity, trade_results[:trade_count]


# ==========================================
# STRATEGY PARAMETERS
# ==========================================


@dataclass
class MACrossoverParams:
    """
    Paramètres de la stratégie MA Crossover.

    Attributes:
        fast_period: Période SMA rapide (défaut: 10)
        slow_period: Période SMA lente (défaut: 30)
        stop_loss_pct: Stop loss en % (défaut: 2.0%)
        take_profit_pct: Take profit en % (défaut: 4.0%)
        risk_per_trade: Risque par trade en fraction du capital (défaut: 0.01 = 1%)
        leverage: Effet de levier (défaut: 1.0)
        max_hold_bars: Durée max position en barres (défaut: 100)
        fee_bps: Frais en basis points (défaut: 4.5)
        slippage_bps: Slippage en basis points (défaut: 0)
        meta: Métadonnées personnalisées

    Example:
        >>> params = MACrossoverParams(fast_period=10, slow_period=30, stop_loss_pct=2.0)
        >>> strategy = MACrossoverStrategy()
        >>> equity, stats = strategy.backtest(df, params.to_dict())
    """

    # Moving Averages
    fast_period: int = 10
    slow_period: int = 30

    # Risk Management
    stop_loss_pct: float = 2.0  # 2% stop loss
    take_profit_pct: float = 4.0  # 4% take profit
    risk_per_trade: float = 0.01  # 1% du capital par trade

    # Position Management
    leverage: float = 1.0  # Pas de levier par défaut
    max_hold_bars: int = 100  # ~4 jours en 1h

    # Frais et slippage
    fee_bps: float = 4.5  # 4.5 bps
    slippage_bps: float = 0.0

    # Métadonnées
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convertit les paramètres en dictionnaire"""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "risk_per_trade": self.risk_per_trade,
            "leverage": self.leverage,
            "max_hold_bars": self.max_hold_bars,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MACrossoverParams":
        """Crée les paramètres depuis un dictionnaire"""
        return cls(
            fast_period=data.get("fast_period", 10),
            slow_period=data.get("slow_period", 30),
            stop_loss_pct=data.get("stop_loss_pct", 2.0),
            take_profit_pct=data.get("take_profit_pct", 4.0),
            risk_per_trade=data.get("risk_per_trade", 0.01),
            leverage=data.get("leverage", 1.0),
            max_hold_bars=data.get("max_hold_bars", 100),
            fee_bps=data.get("fee_bps", 4.5),
            slippage_bps=data.get("slippage_bps", 0.0),
            meta=data.get("meta", {}),
        )


# ==========================================
# STRATEGY IMPLEMENTATION
# ==========================================


class MACrossoverStrategy:
    """
    Stratégie Moving Average Crossover pour validation système.

    Règles:
    - LONG: SMA rapide croise au-dessus SMA lente
    - SHORT: SMA rapide croise en-dessous SMA lente
    - EXIT: Signal inverse, stop loss, take profit, ou max hold

    Example:
        >>> strategy = MACrossoverStrategy()
        >>> params = {"fast_period": 10, "slow_period": 30}
        >>> signals = strategy.generate_signals(df, params)
        >>> equity, stats = strategy.backtest(df, params, initial_capital=10000)
    """

    def __init__(
        self,
        symbol: str = "UNKNOWN",
        timeframe: str = "15m",
        indicator_bank: Any = None,
    ):
        """
        Initialise la stratégie MA Crossover.

        Args:
            symbol: Symbole pour cache d'indicateurs (non utilisé pour MA simple)
            timeframe: Timeframe pour cache d'indicateurs
            indicator_bank: Instance IndicatorBank partagée (optionnel, non utilisé pour SMA)
        """
        self.name = "MA_Crossover"
        self.version = "1.0.0"
        self.symbol = symbol
        self.timeframe = timeframe
        self.indicator_bank = indicator_bank
        logger.info(f"Initialisation {self.name} v{self.version} ({symbol}/{timeframe})")

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Génère les signaux MA Crossover.

        Args:
            df: DataFrame OHLCV
            params: Paramètres de stratégie

        Returns:
            DataFrame avec colonne 'signal'
        """
        validate_ohlcv_dataframe(df)
        validate_strategy_params(params, ["fast_period", "slow_period"])

        p = MACrossoverParams.from_dict(params)
        df_signals = df.copy()

        # Calcul des moyennes mobiles
        fast_sma = df["close"].rolling(window=p.fast_period, min_periods=p.fast_period).mean()
        slow_sma = df["close"].rolling(window=p.slow_period, min_periods=p.slow_period).mean()

        # Détection croisements
        df_signals["signal"] = "HOLD"

        # Crossover up: LONG
        cross_up = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        df_signals.loc[cross_up, "signal"] = "ENTER_LONG"

        # Crossover down: SHORT
        cross_down = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
        df_signals.loc[cross_down, "signal"] = "ENTER_SHORT"

        # Métadonnées pour analyse
        df_signals["fast_sma"] = fast_sma
        df_signals["slow_sma"] = slow_sma

        logger.debug(
            f"Signaux générés: {(df_signals['signal'] == 'ENTER_LONG').sum()} LONG, "
            f"{(df_signals['signal'] == 'ENTER_SHORT').sum()} SHORT"
        )

        return df_signals

    def backtest(
        self,
        df: pd.DataFrame,
        params: dict,
        initial_capital: float = 10000.0,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
        precomputed_indicators: dict | None = None,
    ) -> tuple[pd.Series, RunStats]:
        """
        Exécute un backtest complet de la stratégie.

        Args:
            df: DataFrame OHLCV
            params: Paramètres de stratégie
            initial_capital: Capital initial
            fee_bps: Frais en basis points (override params)
            slippage_bps: Slippage en basis points (override params)
            precomputed_indicators: Indicateurs précalculés (non utilisé pour MA simple)

        Returns:
            Tuple (equity_curve, stats)
        """
        logger.info(
            f"Début backtest {self.name} sur {len(df)} barres, capital={initial_capital}"
        )

        # Validation
        validate_ohlcv_dataframe(df)
        p = MACrossoverParams.from_dict(params)

        # Override frais si fournis
        if fee_bps is not None:
            p.fee_bps = fee_bps
        if slippage_bps is not None:
            p.slippage_bps = slippage_bps

        # Génération signaux
        df_signals = self.generate_signals(df, params)

        # Encodage signaux pour Numba
        signal_map = {"HOLD": 0, "ENTER_LONG": 1, "ENTER_SHORT": 2, "EXIT": 3}
        signal_vals = df_signals["signal"].map(signal_map).fillna(0).astype(np.int32).values

        # Préparation données Numba
        close_vals = df["close"].values.astype(np.float64)
        fee_rate = (p.fee_bps + p.slippage_bps) / 10000.0

        # Exécution backtest Numba
        equity_curve, trade_results = _backtest_loop_numba(
            close_vals=close_vals,
            signal_vals=signal_vals,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            stop_loss_pct=p.stop_loss_pct,
            take_profit_pct=p.take_profit_pct,
            risk_per_trade=p.risk_per_trade,
            leverage=p.leverage,
            max_hold_bars=p.max_hold_bars,
        )

        # Conversion résultats en objets Trade
        trades = []
        for row in trade_results:
            entry_bar = int(row[0])
            exit_bar = int(row[1])
            side = "LONG" if row[2] == 1 else "SHORT"

            trade = Trade(
                side=side,
                qty=row[3],
                entry_price=row[4],
                entry_time=df.index[entry_bar].isoformat(),
                exit_price=row[5],
                exit_time=df.index[exit_bar].isoformat(),
                stop=row[9],
                pnl_realized=row[8],
                fees_paid=row[6] + row[7],
                meta={"strategy": self.name, "params": params},
            )
            trades.append(trade)

        # Création série équité
        equity_series = pd.Series(equity_curve, index=df.index)

        # Calcul statistiques
        stats = RunStats.from_trades_and_equity(
            trades=trades,
            equity_curve=equity_series,
            initial_capital=initial_capital,
            meta={
                "strategy": self.name,
                "params": params,
                "fee_bps": p.fee_bps,
                "trades": [t.to_dict() for t in trades]
            },
        )

        logger.info(
            f"Backtest terminé: {stats.total_trades} trades, PnL={stats.total_pnl:.2f} ({stats.total_pnl_pct:.2f}%)"
        )

        return equity_series, stats


# ==========================================
# MODULE EXPORTS
# ==========================================

__all__ = [
    "MACrossoverStrategy",
    "MACrossoverParams",
]
```
<!-- MODULE-END: ma_crossover.py -->

<!-- MODULE-START: page_llm_optimizer.py -->
```json
{
  "name": "page_llm_optimizer.py",
  "path": "ui\\page_llm_optimizer.py",
  "ext": ".py",
  "anchor": "page_llm_optimizer_py"
}
```
## page_llm_optimizer_py
*Chemin* : `ui\page_llm_optimizer.py`  
*Type* : `.py`  

```python
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
from threadx.llm.model_router import ModelRouter, TaskType
from threadx.llm.ollama_manager import prepare_for_llm_run
from threadx.llm.run_report import (
    LLMRunReport,
    RunIndex,
    create_report_from_run,
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
            help="Stratégie à optimiser"
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
                step = base_step * granularity_factor * local_granularity if base_step is not None else None
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
                    step_local = base_step if base_step is not None else (max_val - min_val) / 10 or 0.1
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
                step = base_step * granularity_factor * local_granularity if base_step is not None else None

            # Générer valeurs avec protection division par zéro et validation
            if min_val is None or max_val is None:
                st.warning(f"⚠️ Valeurs min/max invalides pour {param_name}, ignoré")
                continue
            if step is not None and step <= 0:
                step = None
            values = _generate_sweep_values(min_val, max_val, n_values, param_type, step)
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

        st.info("💡 **Stratégie Multi-LLM Active** : Architecte + Bâtisseur + Auditeurs (rotation)")

        available_models = [
            "deepseek-r1:32b",
            "deepseek-r1:14b",
            "qwen2.5:32b",
            "qwen2.5:14b",
            "mistral:22b",
            "mistral:7b",
        ]

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            architect_model_ui = st.selectbox(
                "🏗️ Architecte",
                options=available_models,
                index=available_models.index("deepseek-r1:32b") if "deepseek-r1:32b" in available_models else 0,
                help="Modèle pour l'étape Analyst/Architecte"
            )
        with col_m2:
            builder_model_ui = st.selectbox(
                "🔨 Bâtisseur",
                options=available_models,
                index=available_models.index("deepseek-r1:14b") if "deepseek-r1:14b" in available_models else 0,
                help="Modèle pour l'étape Strategist/Bâtisseur"
            )
        with col_m3:
            guest_models_ui = st.multiselect(
                "👀 Auditeurs (rotation)",
                options=available_models,
                default=["qwen2.5:32b", "mistral:22b"],
                help="Liste des modèles qui tourneront en rotation pour l'audit"
            )

        # Afficher les modèles qui seront utilisés
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("🏗️ Architecte", architect_model_ui, help="Initialisation robuste")
        with col_m2:
            st.metric("🔨 Bâtisseur", builder_model_ui, help="Optimisation itérative")
        with col_m3:
            st.metric("👀 Auditeurs", "Rotation", help=f"Rotation: {', '.join(guest_models_ui) if guest_models_ui else 'aucun'}")

        # Pour compatibilité avec le reste du code, on définit des valeurs par défaut
        # Mais le routeur sera prioritaire dans la boucle d'optimisation
        router = ModelRouter(
            architect_model=architect_model_ui,
            builder_model=builder_model_ui,
            guest_models=guest_models_ui,
        )

        analyst_model = router.get_model_for_task(TaskType.INITIALIZATION)
        strategist_model = router.get_model_for_task(TaskType.OPTIMIZATION, step_number=st.session_state.get("llm_step", 1))

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
        with st.spinner(f"Test de {len(list(_generate_combinations(sweep_params, strategy_name)))} configurations..."):
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

            with st.expander("📊 Top 10 configurations", expanded=True):
                df_sweep = pd.DataFrame(sweep_results)

                # Normaliser les noms de colonnes (les métriques devraient déjà être extraites par execute_sweep)
                # Mais vérification au cas où
                if "sharpe" in df_sweep.columns and "sharpe_ratio" not in df_sweep.columns:
                    df_sweep["sharpe_ratio"] = df_sweep["sharpe"]
                if "pnl_pct" in df_sweep.columns and "total_return" not in df_sweep.columns:
                    df_sweep["total_return"] = df_sweep["pnl_pct"]

                # Filtrer les configs invalides avant tout tri pour MA/EMA (slow > fast)
                def _is_valid_ma_row(row: pd.Series) -> bool:
                    params = row.get("params", {})
                    if not params and isinstance(row, pd.Series):
                        # fallback si les valeurs sont à plat dans le row
                        params = {k: row.get(k) for k in ["fast_period", "slow_period"] if k in row}
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
                        st.warning(f"⚠️ {dropped} configs invalides retirées (slow_period ≤ fast_period).")
                        logger.warning(f"[Sweep] {dropped} configs MA/EMA filtrées (slow<=fast)")

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
            current_analyst_model = model_router.get_model_for_task(TaskType.INITIALIZATION, step_number=current_step)

        status_text.markdown(f"### 🧠 Étape 2/5: Analyse Analyst ({current_analyst_model})...")

        logger.info(f"[Multi-LLM] Étape 2/5 ANALYST démarré - modèle:{current_analyst_model}, top_n:{top_n_analysis}")

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
            current_strategist_model = model_router.get_model_for_task(TaskType.OPTIMIZATION, step_number=current_step)

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
                sharpe_display = f"{sharpe:.3f}" if isinstance(sharpe, (int, float)) else "N/A"
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
            # Fallback auto-fix pour MA_Crossover / EMA_Cross : corriger slow<=fast
            if strategy_name in ["MA_Crossover", "EMA_Cross"] and len(df_sweep) > 0:
                top_candidate = df_sweep.nlargest(1, "sharpe_ratio").iloc[0].to_dict()
                param_specs_full = parameter_specs_for(strategy_name)
                params = top_candidate.get("params", {})
                if not params:
                    params = {k: v for k, v in top_candidate.items() if k in param_specs_full.keys()}

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
        baseline_sharpe_val = baseline_config.get('sharpe_ratio')
        baseline_return_val = baseline_config.get('total_return')
        baseline_sharpe_display = f"{baseline_sharpe_val:.3f}" if isinstance(baseline_sharpe_val, (int, float)) else "N/A"
        baseline_return_display = f"{baseline_return_val:.3f}" if isinstance(baseline_return_val, (int, float)) else "N/A"
        st.caption(f"📊 Baseline sélectionnée - Sharpe: {baseline_sharpe_display}, Return: {baseline_return_display}")

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

                    logger.info(f"[Multi-LLM] Étape 3/5 STRATEGIST démarré - modèle:{current_strategist_model}, n_proposals:{n_proposals}")

                    proposals_result = strategist.propose_modifications(
                        analysis=analysis_result,
                        current_params=baseline_params,
                        param_specs=param_specs_full,
                        n_proposals=n_proposals
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

        logger.info(f"[Multi-LLM] Étape 4/5 TESTS démarré - {len(proposals_result['proposals'])} propositions à tester")
        test_start = time.time()

        with st.spinner(f"Test de {len(proposals_result['proposals'])} propositions..."):
            test_results = test_proposals(
                strategy_name=strategy_name,
                proposals=proposals_result["proposals"],
                baseline_config=baseline_config,
                use_gpu=use_gpu,
            )

        test_duration = time.time() - test_start
        successful_tests = sum(1 for r in test_results if r.get('sharpe_ratio'))
        logger.info(
            f"[Multi-LLM] Étape 4/5 TESTS terminé - "
            f"{successful_tests}/{len(proposals_result['proposals'])} propositions valides en {test_duration:.2f}s"
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

        # ============================================================
        # GÉNÉRATION DU RAPPORT ET INDEXATION
        # ============================================================
        status_text.markdown("### 📁 Sauvegarde du rapport...")

        try:
            # Calculer durée totale du run
            run_end_time = time.time()
            total_run_duration = run_end_time - st.session_state.get("llm_run_start_time", run_end_time)

            # Récupérer les durées stockées dans session_state
            sweep_duration = st.session_state.get("llm_sweep_duration", 0.0)
            analyst_duration = st.session_state.get("llm_analyst_duration", 0.0)
            strategist_duration = st.session_state.get("llm_strategist_duration", 0.0)

            # Baseline sharpe
            baseline_sharpe = baseline_config.get("sharpe_ratio", baseline_config.get("sharpe", 0.0))

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
                report,
                tags=[strategy_name, f"sharpe_{report.best_sharpe:.2f}"]
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
                    improvement = "✅ Oui" if (report.tests and report.tests.improvement_found) else "❌ Non"
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

        progress_bar.progress(100)
        status_text.success("### 🎉 Optimisation Multi-LLM terminée !")

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
        logger.error(f"[Multi-LLM] ERREUR lors de l'optimisation: {type(e).__name__}: {str(e)}", exc_info=True)
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


def _apply_strategy_constraints(params: dict, strategy_name: str, context: str = "") -> dict:
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
                notices.append(f"slow_period ajusté à {adjusted['slow_period']} (> fast_period {fast})")
        except Exception:
            pass

    # risk_per_trade: clamp sur [0.5%, 2%]
    if "risk_per_trade" in adjusted and adjusted["risk_per_trade"] is not None:
        try:
            r = float(adjusted["risk_per_trade"])
            clamped = max(RISK_PER_TRADE_MIN, min(RISK_PER_TRADE_MAX, r))
            if clamped != r:
                adjusted["risk_per_trade"] = clamped
                notices.append(f"risk_per_trade bridé à {clamped:.3f} (bornes {RISK_PER_TRADE_MIN:.3f}-{RISK_PER_TRADE_MAX:.3f})")
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

    logger.info(f"[Sweep] Démarrage sweep - strategy:{strategy_name}, gpu:{use_gpu}, workers:{max_workers}")

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

    logger.debug(f"[Sweep] {total_combinations} combinaisons générées")

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

    best_sharpe = max((r.get('sharpe_ratio', r.get('sharpe', 0)) for r in results), default=0)
    logger.info(
        f"[Sweep] Terminé - {len(results)} résultats, "
        f"meilleur sharpe:{best_sharpe:.3f}, durée:{elapsed_time:.1f}s"
    )

    return results


def test_proposals(strategy_name: str, proposals: list, baseline_config: dict, use_gpu: bool):
    """Teste chaque proposition et retourne les résultats."""

    logger.info(f"[Test Proposals] Démarrage - {len(proposals)} propositions à tester")

    # Extraire les paramètres baseline pour compléter les propositions partielles
    param_specs = parameter_specs_for(strategy_name)
    baseline_params = baseline_config.get("params", {})
    if not baseline_params:
        baseline_params = {k: v for k, v in baseline_config.items() if k in param_specs.keys()}

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

            # Fusion baseline + proposition pour obtenir un set complet
            merged_params = {**baseline_params, **prop.get("params", {})}
            if len(merged_params) < len(param_specs):
                missing = set(param_specs.keys()) - set(merged_params.keys())
                logger.warning(
                    "[Test Proposals] %s: paramètres manquants -> complétés depuis baseline: %s",
                    prop.get("name", "Unknown"),
                    missing,
                )
                st.info(f"ℹ️ Params manquants complétés depuis la baseline: {', '.join(sorted(missing))}")

            # CORRECTION: Forcer conversion int + contraintes (slow>fast, risk bounds)
            cleaned_params = _force_integer_params(merged_params)
            cleaned_params = _apply_strategy_constraints(cleaned_params, strategy_name, context=prop.get("name", "proposal"))

            result = run_backtest_gpu(
                df=df_market,
                strategy=strategy_name,
                params=cleaned_params,
            )

            test_results.append({
                "name": prop["name"],
                "params": merged_params,  # Params réellement testés (baseline + overrides)
                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
                "total_return": result.metrics.get("total_return", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                "win_rate": result.metrics.get("win_rate", 0.0),
                "trades": result.trades if hasattr(result, 'trades') else [],  # ← Capture trades
                "full_result": result,  # ← Capture résultat complet pour analyses futures
            })

            sharpe = result.metrics.get('sharpe_ratio', 0.0)
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
            sharpe = res.get("sharpe_ratio")
            sharpe_display = f"{sharpe:.3f}" if isinstance(sharpe, (int, float)) else "N/A"

            with st.expander(f"📊 Proposition {i}: {config_name} (Sharpe: {sharpe_display})", expanded=False):
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
```
<!-- MODULE-END: page_llm_optimizer.py -->
