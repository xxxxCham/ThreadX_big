"""
Page Streamlit: Historique des Rapports Multi-LLM
=================================================

Interface pour visualiser, rechercher et comparer les runs Multi-LLM précédents.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ajouter src au path si nécessaire
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from threadx.llm.run_report import RunIndex, LLMRunReport, IndexEntry


def render_page():
    """Affiche la page d'historique des rapports."""

    st.title("📚 Historique des Rapports Multi-LLM")
    st.markdown("""
    **Consultez et comparez** les résultats de vos optimisations Multi-LLM précédentes.
    """)

    # Charger l'index
    try:
        index = RunIndex()
    except Exception as e:
        st.error(f"Erreur chargement index: {e}")
        return

    # Statistiques globales
    stats = index.get_stats()

    if stats["total_runs"] == 0:
        st.info("📭 Aucun rapport trouvé. Lancez une optimisation Multi-LLM pour créer votre premier rapport.")
        return

    # Afficher les stats
    st.markdown("### 📊 Statistiques Globales")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔢 Total Runs", stats["total_runs"])
    with col2:
        st.metric("📈 Sharpe Max", f"{stats['max_sharpe']:.3f}")
    with col3:
        st.metric("✅ Améliorations", stats["improvements_found"])
    with col4:
        st.metric("🧪 Configs Testées", f"{stats['total_configs_tested']:,}")

    st.divider()

    # Filtres de recherche
    st.markdown("### 🔍 Filtres de Recherche")
    col1, col2, col3 = st.columns(3)

    with col1:
        strategies = ["Toutes"] + index.list_strategies()
        selected_strategy = st.selectbox(
            "Stratégie",
            options=strategies,
            index=0,
        )

    with col2:
        min_sharpe = st.number_input(
            "Sharpe minimum",
            value=-10.0,
            step=0.1,
            format="%.2f",
        )

    with col3:
        improvement_only = st.checkbox(
            "Améliorations uniquement",
            value=False,
        )

    # Rechercher
    search_params = {
        "limit": 50,
        "min_sharpe": min_sharpe if min_sharpe > -10 else None,
        "improvement_only": improvement_only,
    }
    if selected_strategy != "Toutes":
        search_params["strategy"] = selected_strategy

    results = index.search(**search_params)

    st.markdown(f"### 📋 Résultats ({len(results)} runs)")

    if not results:
        st.info("Aucun run correspondant aux critères.")
        return

    # Afficher la liste des runs
    for entry in results:
        with st.expander(
            f"**{entry.run_id}** | {entry.strategy_name} | Sharpe: {entry.best_sharpe:.3f} | {entry.timestamp[:10]}",
            expanded=False,
        ):
            render_run_summary(entry, index)


def render_run_summary(entry: IndexEntry, index: RunIndex):
    """Affiche le résumé d'un run."""

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Stratégie", entry.strategy_name)
    with col2:
        st.metric("📊 Best Sharpe", f"{entry.best_sharpe:.3f}")
    with col3:
        improvement = "✅ Oui" if entry.improvement_found else "❌ Non"
        st.metric("📈 Amélioration", improvement)

    st.caption(f"📅 Date: {entry.timestamp}")
    st.caption(f"🧪 Configs testées: {entry.total_configs}")

    if entry.tags:
        st.caption(f"🏷️ Tags: {', '.join(entry.tags)}")

    # Boutons d'action
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📖 Voir détails", key=f"view_{entry.run_id}"):
            st.session_state["view_report_id"] = entry.run_id

    with col2:
        # Charger le rapport pour téléchargement
        report = index.load_report(entry.run_id)
        if report:
            st.download_button(
                label="📥 JSON",
                data=report.to_json(),
                file_name=f"report_{entry.run_id}.json",
                mime="application/json",
                key=f"dl_{entry.run_id}",
            )

    with col3:
        if st.button("🗑️ Supprimer", key=f"del_{entry.run_id}"):
            if index.delete_run(entry.run_id, delete_files=True):
                st.success("Run supprimé")
                st.rerun()

    # Si on demande les détails
    if st.session_state.get("view_report_id") == entry.run_id:
        render_full_report(entry.run_id, index)


def render_full_report(run_id: str, index: RunIndex):
    """Affiche le rapport complet d'un run."""

    st.divider()
    st.markdown("## 📋 Rapport Détaillé")

    report = index.load_report(run_id)
    if not report:
        st.error("Impossible de charger le rapport")
        return

    # Résumé
    st.markdown(report.summary)

    st.divider()

    # Onglets pour les différentes sections
    tabs = st.tabs(["🔄 Sweep", "🧠 Analyse", "🎨 Propositions", "🧪 Tests"])

    # === TAB SWEEP ===
    with tabs[0]:
        st.markdown("### Résultats du Sweep GPU")
        st.caption(f"⏱️ Durée: {report.sweep.duration_seconds:.1f}s | 🧪 Configs: {report.sweep.total_configs}")

        if report.sweep.metrics_summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_sharpe = report.sweep.metrics_summary.get("avg_sharpe_ratio",
                             report.sweep.metrics_summary.get("avg_sharpe", 0))
                st.metric("📊 Sharpe Moyen", f"{avg_sharpe:.3f}")
            with col2:
                max_sharpe = report.sweep.metrics_summary.get("max_sharpe_ratio",
                             report.sweep.metrics_summary.get("max_sharpe", 0))
                st.metric("📈 Sharpe Max", f"{max_sharpe:.3f}")
            with col3:
                std_sharpe = report.sweep.metrics_summary.get("std_sharpe_ratio",
                             report.sweep.metrics_summary.get("std_sharpe", 0))
                st.metric("📉 Sharpe Std", f"{std_sharpe:.3f}")

        st.markdown("**Top Configurations:**")
        if report.sweep.top_configs:
            df_top = pd.DataFrame(report.sweep.top_configs[:10])
            st.dataframe(df_top, use_container_width=True, hide_index=True)

        st.markdown("**Paramètres testés:**")
        st.json(report.sweep.params_tested)

    # === TAB ANALYSE ===
    with tabs[1]:
        if report.analysis:
            st.markdown("### Analyse de l'Analyst")
            st.caption(f"🤖 Modèle: {report.analysis.model_used} | ⏱️ Durée: {report.analysis.duration_seconds:.1f}s")

            if report.analysis.patterns:
                st.markdown("**🔍 Patterns identifiés:**")
                for pattern in report.analysis.patterns:
                    st.markdown(f"- {pattern}")

            if report.analysis.key_metrics:
                st.markdown("**📊 Métriques clés:**")
                col1, col2 = st.columns(2)
                metrics_list = list(report.analysis.key_metrics.items())
                for i, (k, v) in enumerate(metrics_list):
                    with col1 if i % 2 == 0 else col2:
                        st.metric(k, f"{v:.3f}" if isinstance(v, float) else str(v))

            if report.analysis.trade_offs:
                st.markdown("**⚖️ Trade-offs:**")
                for tradeoff in report.analysis.trade_offs:
                    st.markdown(f"- {tradeoff}")

            if report.analysis.recommendations:
                st.markdown("**💡 Recommandations:**")
                for rec in report.analysis.recommendations:
                    st.markdown(f"- {rec}")
        else:
            st.info("Pas d'analyse disponible pour ce run.")

    # === TAB PROPOSITIONS ===
    with tabs[2]:
        if report.proposals:
            st.markdown("### Propositions du Strategist")
            st.caption(
                f"🤖 Modèle: {report.proposals.model_used} | "
                f"⏱️ Durée: {report.proposals.duration_seconds:.1f}s | "
                f"✅ Valides: {report.proposals.total_valid}/{report.proposals.total_generated}"
            )

            st.markdown(f"**Baseline Sharpe:** {report.proposals.baseline_sharpe:.3f}")

            for i, prop in enumerate(report.proposals.proposals):
                with st.expander(f"📝 {prop.name}", expanded=i == 0):
                    st.markdown(f"**Rationale:** {prop.rationale}")

                    if prop.changes_from_baseline:
                        st.markdown("**Modifications:**")
                        for param, change in prop.changes_from_baseline.items():
                            st.markdown(f"- `{param}`: {change['old']} → {change['new']}")

                    st.markdown("**Paramètres complets:**")
                    st.json(prop.params)
        else:
            st.info("Pas de propositions disponibles pour ce run.")

    # === TAB TESTS ===
    with tabs[3]:
        if report.tests:
            st.markdown("### Résultats des Tests")
            st.caption(f"🧪 Testées: {report.tests.total_tested} | ✅ Succès: {report.tests.successful_tests}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Baseline Sharpe", f"{report.tests.baseline_sharpe:.3f}")
            with col2:
                st.metric("📈 Best Sharpe", f"{report.tests.best_sharpe:.3f}")
            with col3:
                improvement = "✅ Oui" if report.tests.improvement_found else "❌ Non"
                st.metric("🏆 Amélioration", improvement)

            if report.tests.best_proposal:
                st.success(f"🏆 **Meilleure proposition:** {report.tests.best_proposal}")

            st.markdown("**Détails par proposition:**")
            for result in report.tests.results:
                icon = "✅" if result.is_improvement else "❌"
                delta = f"{result.vs_baseline_sharpe:+.3f}"

                with st.expander(f"{icon} {result.name} | Sharpe: {result.sharpe_ratio:.3f} ({delta})"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Sharpe", f"{result.sharpe_ratio:.3f}")
                    with col2:
                        st.metric("💰 Return", f"{result.total_return:.2f}%")
                    with col3:
                        st.metric("📉 Max DD", f"{result.max_drawdown:.2f}%")
                    with col4:
                        st.metric("🎯 Win Rate", f"{result.win_rate:.1f}%")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📈 Trades", result.total_trades)
                    with col2:
                        st.metric("✅ Profits", result.profit_trades)
                    with col3:
                        st.metric("❌ Pertes", result.loss_trades)
        else:
            st.info("Pas de tests disponibles pour ce run.")

    # Bouton pour fermer les détails
    if st.button("❌ Fermer les détails", key=f"close_{run_id}"):
        st.session_state["view_report_id"] = None
        st.rerun()


if __name__ == "__main__":
    render_page()
