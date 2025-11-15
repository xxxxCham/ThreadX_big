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
from threadx.ui.system_monitor import get_global_monitor

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


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("# ThreadX v2.0")
        st.markdown("*Trading Quantitatif Haute Performance*")
        st.markdown("---")

        # Barre de progression du workflow
        st.markdown("### 📍 Progression")
        steps_total = 3
        page_to_step = {"config": 1, "backtest": 2, "monitor": 3}
        current_page = st.session_state.get("page", "config")
        current_step = page_to_step.get(current_page, 1)

        # Afficher la barre de progression
        st.progress(current_step / steps_total)

        # Afficher les étapes avec statut
        st.caption(
            f"Étape 1/3 : Configuration données "
            f"{'✅' if current_step > 1 else '⏳' if current_step == 1 else '⭕'}"
        )
        st.caption(
            f"Étape 2/3 : Optimisation "
            f"{'✅' if current_step > 2 else '⏳' if current_step == 2 else '⭕'}"
        )
        st.caption(
            f"Étape 3/3 : Monitoring système "
            f"{'⏳' if current_step == 3 else '⭕'}"
        )

        # Bouton "Suivant" selon l'étape actuelle
        if current_page == "config":
            if st.button("➡️ Passer à l'Optimisation", type="primary", use_container_width=True):
                st.session_state.page = "backtest"
                st.rerun()
        elif current_page == "backtest":
            if st.button("➡️ Passer au Monitoring", type="primary", use_container_width=True):
                st.session_state.page = "monitor"
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
        st.markdown("---")
        if st.button("🔄 Rafraîchir Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache vidé!")
        st.markdown("---")
        st.caption("**ThreadX v2.0** | © 2025")


def main() -> None:
    init_session()
    render_sidebar()
    page_key = st.session_state.get("page", "config")
    renderer = PAGE_RENDERERS.get(page_key, config_page_main)
    renderer()


if __name__ == "__main__":
    main()
