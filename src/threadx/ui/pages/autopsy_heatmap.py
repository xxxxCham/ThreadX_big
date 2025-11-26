"""
Autopsy Heatmap - Dashboard Patterns Échecs Stratégies
======================================================

Visualisation real-time patterns échecs Autopsy.
Aggrégation causes principales, metrics victimes, recommandations.

Features:
    - Heatmap causes (fréquence + severity)
    - Timeline échecs (derniers 30 jours)
    - Top correctifs recommandés
    - Kill rules actives
    - Feedback Strategist preview

Usage:
    streamlit run src/threadx/ui/pages/autopsy_heatmap.py
    Ou intégré dans menu principal Streamlit app

Author: ThreadX Framework
Version: 1.0 - Auto-Learning System
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="Autopsy - Heatmap Échecs",
    page_icon="🔬",
    layout="wide",
)


def load_autopsy_reports(reports_dir: Path = Path("./autopsy_reports")) -> list[dict]:
    """
    Charge tous rapports Autopsy.

    Args:
        reports_dir: Répertoire rapports

    Returns:
        Liste rapports (triés par timestamp desc)
    """
    if not reports_dir.exists():
        return []

    reports = []
    for report_file in reports_dir.glob("*.json"):
        try:
            with open(report_file) as f:
                report = json.load(f)
                reports.append(report)
        except Exception as e:
            st.warning(f"Failed to load {report_file.name}: {e}")

    # Trier par timestamp desc
    reports.sort(
        key=lambda r: r.get("timestamp", "1970-01-01T00:00:00"),
        reverse=True,
    )

    return reports


def get_failure_patterns_summary(reports: list[dict]) -> dict[str, dict]:
    """
    Agrège patterns échecs par cause.

    Args:
        reports: Liste rapports Autopsy

    Returns:
        Dict {cause: {count, last_seen, avg_sharpe_victims, avg_improvement_score}}
    """
    patterns = {}

    for report in reports:
        cause = report.get("cause_principale", "unknown")
        timestamp = report.get("timestamp", "")
        improvement_score = report.get("score_amelioration_attendue", 0)

        # Extraction Sharpe ratio victimes (depuis code_snapshot ou metadata)
        sharpe_victim = 0.0
        code_snapshot = report.get("code_snapshot", "")
        if "sharpe_ratio" in code_snapshot.lower():
            # Tentative extraction (pattern basique)
            import re

            match = re.search(r"sharpe[_\s]*ratio[:\s]*([-\d.]+)", code_snapshot.lower())
            if match:
                try:
                    sharpe_victim = float(match.group(1))
                except ValueError:
                    pass

        if cause not in patterns:
            patterns[cause] = {
                "count": 0,
                "last_seen": timestamp,
                "sharpe_victims": [],
                "improvement_scores": [],
            }

        patterns[cause]["count"] += 1
        patterns[cause]["last_seen"] = max(patterns[cause]["last_seen"], timestamp)
        patterns[cause]["sharpe_victims"].append(sharpe_victim)
        patterns[cause]["improvement_scores"].append(improvement_score)

    # Calculer moyennes
    for cause, data in patterns.items():
        data["avg_sharpe_victims"] = (
            sum(data["sharpe_victims"]) / len(data["sharpe_victims"])
            if data["sharpe_victims"]
            else 0.0
        )
        data["avg_improvement_score"] = (
            sum(data["improvement_scores"]) / len(data["improvement_scores"])
            if data["improvement_scores"]
            else 0.0
        )

        # Cleanup listes (garder seulement aggregés)
        del data["sharpe_victims"]
        del data["improvement_scores"]

    # Trier par fréquence
    patterns = dict(sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True))

    return patterns


def load_kill_rules(rules_path: Path = Path("./kill_rules.json")) -> list[dict]:
    """
    Charge kill rules actives.

    Args:
        rules_path: Chemin fichier kill rules

    Returns:
        Liste rules actives
    """
    if not rules_path.exists():
        return []

    try:
        with open(rules_path) as f:
            rules = json.load(f)

        # Filtrer actives
        return [r for r in rules if r.get("active", True)]

    except Exception as e:
        st.warning(f"Failed to load kill rules: {e}")
        return []


def render_header():
    """Affiche header page."""
    st.title("🔬 Autopsy - Heatmap Échecs Stratégies")
    st.markdown(
        """
    **Dashboard real-time patterns échecs stratégies.**
    Analyse post-mortem automatique → Kill Rules → Auto-correction.
    """
    )


def render_stats(reports: list[dict], patterns: dict[str, dict]):
    """Affiche stats globales."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Échecs", len(reports))

    with col2:
        st.metric("Patterns Uniques", len(patterns))

    with col3:
        if patterns:
            top_cause = max(patterns.items(), key=lambda x: x[1]["count"])
            st.metric("Top Cause", top_cause[0], f"{top_cause[1]['count']} occ.")
        else:
            st.metric("Top Cause", "N/A")

    with col4:
        if reports:
            latest = reports[0]
            latest_time = datetime.fromisoformat(latest.get("timestamp", "1970-01-01T00:00:00"))
            delay = datetime.now() - latest_time
            delay_str = f"il y a {delay.days}j" if delay.days > 0 else "aujourd'hui"
            st.metric("Dernier Échec", delay_str)
        else:
            st.metric("Dernier Échec", "N/A")


def render_heatmap(patterns: dict[str, dict]):
    """Affiche heatmap patterns."""
    st.subheader("📊 Heatmap Causes Principales")

    if not patterns:
        st.info("Aucun échec enregistré (aucun pattern à afficher)")
        return

    # Créer DataFrame pour heatmap
    df = pd.DataFrame.from_dict(patterns, orient="index")
    df = df.reset_index()
    df.columns = ["Cause", "Count", "Last Seen", "Avg Sharpe Victims", "Avg Improvement Score"]

    # Formatter dates
    df["Last Seen"] = pd.to_datetime(df["Last Seen"]).dt.strftime("%Y-%m-%d %H:%M")

    # Formater floats
    df["Avg Sharpe Victims"] = df["Avg Sharpe Victims"].round(2)
    df["Avg Improvement Score"] = df["Avg Improvement Score"].round(1)

    # Trier par count desc
    df = df.sort_values("Count", ascending=False)

    # Afficher table avec couleurs
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Count": st.column_config.NumberColumn(
                "Nb Occurrences",
                help="Nombre fois ce pattern détecté",
                format="%d",
            ),
            "Avg Improvement Score": st.column_config.ProgressColumn(
                "Score Amélioration Moyen",
                help="Score amélioration attendue (0-10)",
                min_value=0,
                max_value=10,
                format="%.1f/10",
            ),
            "Avg Sharpe Victims": st.column_config.NumberColumn(
                "Sharpe Moyen Victimes",
                help="Sharpe ratio moyen stratégies victimes",
                format="%.2f",
            ),
        },
    )


def render_timeline(reports: list[dict]):
    """Affiche timeline échecs (30 derniers jours)."""
    st.subheader("📅 Timeline Échecs (30 derniers jours)")

    if not reports:
        st.info("Aucun échec enregistré")
        return

    # Filtrer 30 derniers jours
    cutoff = datetime.now() - timedelta(days=30)
    recent_reports = [
        r
        for r in reports
        if datetime.fromisoformat(r.get("timestamp", "1970-01-01T00:00:00")) >= cutoff
    ]

    if not recent_reports:
        st.info("Aucun échec dans les 30 derniers jours")
        return

    # Créer DataFrame timeline
    timeline_data = []
    for r in recent_reports:
        timeline_data.append(
            {
                "Date": datetime.fromisoformat(r.get("timestamp", "")).strftime("%Y-%m-%d"),
                "Stratégie": r.get("strategy_name", "unknown"),
                "Cause": r.get("cause_principale", "unknown"),
                "Score": r.get("score_amelioration_attendue", 0),
            }
        )

    df_timeline = pd.DataFrame(timeline_data)

    # Agréger par date + cause
    df_agg = (
        df_timeline.groupby(["Date", "Cause"])
        .size()
        .reset_index(name="Count")
    )

    # Bar chart
    import altair as alt

    chart = (
        alt.Chart(df_agg)
        .mark_bar()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Count:Q", title="Nombre Échecs"),
            color=alt.Color("Cause:N", title="Cause Principale"),
            tooltip=["Date", "Cause", "Count"],
        )
        .properties(height=400)
    )

    st.altair_chart(chart, use_container_width=True)


def render_top_correctifs(reports: list[dict]):
    """Affiche top correctifs recommandés."""
    st.subheader("🛠️ Top Correctifs Recommandés")

    if not reports:
        st.info("Aucun correctif disponible")
        return

    # Extraire tous correctifs
    all_correctifs = []
    for r in reports:
        correctifs = r.get("correctifs_concrets", [])
        all_correctifs.extend(correctifs)

    if not all_correctifs:
        st.info("Aucun correctif dans rapports")
        return

    # Compter fréquence
    from collections import Counter

    correctif_counts = Counter(all_correctifs)
    top_correctifs = correctif_counts.most_common(10)

    # Afficher liste
    for i, (correctif, count) in enumerate(top_correctifs, start=1):
        st.markdown(f"{i}. **{correctif}** ({count} fois recommandé)")


def render_kill_rules(rules: list[dict]):
    """Affiche kill rules actives."""
    st.subheader("⚔️ Kill Rules Actives")

    if not rules:
        st.info("Aucune kill rule active (toutes stratégies acceptées)")
        return

    st.markdown(
        f"**{len(rules)} règles actives** (stratégies violant ces règles = rejet automatique)"
    )

    # Afficher table
    df_rules = pd.DataFrame(rules)

    # Sélectionner colonnes pertinentes
    display_cols = ["rule", "added_at", "source", "improvement_score"]
    df_rules = df_rules[[c for c in display_cols if c in df_rules.columns]]

    # Formatter dates
    if "added_at" in df_rules.columns:
        df_rules["added_at"] = pd.to_datetime(df_rules["added_at"]).dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(
        df_rules,
        use_container_width=True,
        column_config={
            "rule": st.column_config.TextColumn("Règle", width="large"),
            "added_at": st.column_config.TextColumn("Ajoutée le"),
            "source": st.column_config.TextColumn("Source"),
            "improvement_score": st.column_config.NumberColumn(
                "Score Amélioration",
                format="%.1f/10",
            ),
        },
    )


def render_strategist_feedback(reports: list[dict], rules: list[dict]):
    """Affiche preview feedback Strategist."""
    st.subheader("💬 Preview Feedback Strategist")

    # Simuler feedback (top 5 échecs)
    if not reports:
        st.info("Aucun feedback disponible (pas d'échecs)")
        return

    patterns = get_failure_patterns_summary(reports)
    top_patterns = list(patterns.items())[:5]

    feedback = f"**Tu as déjà échoué {len(reports)} fois.**\n\n"
    feedback += "**Top 5 Causes:**\n\n"

    for i, (cause, data) in enumerate(top_patterns, start=1):
        last_seen = datetime.fromisoformat(data["last_seen"])
        delay = datetime.now() - last_seen
        delay_str = f"il y a {delay.days}j" if delay.days > 0 else "aujourd'hui"

        feedback += (
            f"{i}. **{cause}** – {data['count']} occurrences (dernière: {delay_str})\n"
        )

    feedback += f"\n**Kill Rules Actives:** {len(rules)}\n\n"
    if rules:
        feedback += "**Top 3 règles:**\n"
        for i, rule in enumerate(rules[:3], start=1):
            feedback += f"{i}. {rule['rule']}\n"

    feedback += "\n**→ Tu DOIS éviter ces patterns à tout prix.**"

    st.markdown(feedback)


def main():
    """Main function."""
    render_header()

    # Charger données
    reports = load_autopsy_reports()
    patterns = get_failure_patterns_summary(reports)
    rules = load_kill_rules()

    # Stats globales
    render_stats(reports, patterns)

    st.divider()

    # Heatmap
    render_heatmap(patterns)

    st.divider()

    # Timeline
    render_timeline(reports)

    st.divider()

    # 2 colonnes: correctifs + kill rules
    col1, col2 = st.columns(2)

    with col1:
        render_top_correctifs(reports)

    with col2:
        render_kill_rules(rules)

    st.divider()

    # Preview feedback
    render_strategist_feedback(reports, rules)

    # Refresh button
    if st.button("🔄 Rafraîchir données"):
        st.rerun()


if __name__ == "__main__":
    main()
