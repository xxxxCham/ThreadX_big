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

# ================================================================================
# MÉTRIQUES TIER S 2025 - STANDARDS PROFESSIONNELS THREADX
# ================================================================================
# Configuration seuils validation Critic (à intégrer dans critic.py si besoin)
CRITIC_THRESHOLD_2025 = {
    "sharpe_ratio": 1.80,           # Roi des métriques
    "sortino_ratio": 2.80,          # Plus honnête en crypto
    "calmar_ratio": 1.50,           # Gain annuel / MaxDD
    "profit_factor": 2.00,          # Gross profit / Gross loss
    "max_drawdown_pct": -18.0,      # Seuil douleur acceptable
    "recovery_factor": 6.0,         # NetProfit / MaxDD
    "win_rate_trend": 0.58,         # Pour trend-following
    "win_rate_meanrev": 0.68,       # Pour mean-reversion
    "expectancy_pct": 0.8,          # Espérance par trade
    "sqn": 2.8,                     # Van Tharp System Quality Number
    "outlier_adjusted_sharpe": 1.4, # Sharpe sans 3 meilleurs trades
    "multi_token_sharpe_mean": 1.6, # Robustesse multi-assets
    "r_multiple": 2.0,              # Avg Win / Avg Loss
    "max_time_in_market": 0.45,     # 45% max (plus de cash = mieux)
    "max_flat_period_days": 35,     # Seuil mort stratégie en range
    "max_consecutive_loss_pct": -12.0, # Plus grosse série noire
}

# Tier A/B/C disponibles pour extensions futures
# Pain-Adjusted Return (PAR) = Annual Return × (1 + Recovery Factor)
# Serenity Ratio = Sharpe × (1 - % temps en marché)
# Stability Score = Moyenne mobile 30j du Sharpe
# Edge Ratio = Expectancy / Avg Trade Duration (heures)

# ================================================================================
# PROMPTS MULTI-AGENTS - ORCHESTRATION AUTONOME
# ================================================================================

ANALYST_SYSTEM_PROMPT = """Tu es un analyste quantitatif expert en trading algorithmique.

Ton rôle:
- Analyser rigoureusement les résultats de backtests
- Identifier patterns statistiques significatifs
- Détecter anomalies et risques d'overfitting
- Fournir diagnostic factuel et objectif

Principes:
- Rigueur scientifique (pas de spéculation)
- Métriques quantifiées (avec chiffres précis)
- Détection proactive des biais
- Température basse = Analyse factuelle

Format de sortie: JSON structuré uniquement."""

ANALYST_BACKTEST_PROMPT = """Analyse ces résultats de backtest en profondeur:

**Métriques de performance:**
{metrics_json}

**Statistiques trades:**
{trades_stats}

**Objectif:** Fournis une analyse quantitative rigoureuse.

**Format de réponse (JSON STRICT):**
```json
{{
  "score_global": 0-10 (note qualité globale),
  "forces": [
    "Sharpe ratio 1.2 = rendement ajusté risque solide",
    "Win rate 65% = stratégie efficace court-terme",
    "..."
  ],
  "faiblesses": [
    "Drawdown 18% = risque élevé en période stress",
    "Nombre trades 45 = échantillon statistiquement faible",
    "..."
  ],
  "hypotheses": [
    "Période courte (fast=8) génère faux signaux en volatilité",
    "Stop loss 1.0% trop serré pour timeframe 15m",
    "..."
  ],
  "anomalies": [
    "Config #23 : Sharpe 3.5 suspect (possible overfitting)",
    "Aucun trade perdant consécutif >2 (données biaisées?)",
    "..."
  ],
  "metrics_quality": {{
    "sharpe_ratio": "EXCELLENT|GOOD|AVERAGE|POOR",
    "drawdown": "EXCELLENT|GOOD|AVERAGE|POOR",
    "sample_size": "SUFFICIENT|MARGINAL|INSUFFICIENT"
  }},
  "overfitting_risk": "LOW|MEDIUM|HIGH",
  "confidence": 0.0-1.0
}}
```

**Critères qualité (Standards 2025 - Tier S):**

**TIER S (Métriques décisives):**
- Sharpe ≥1.80 EXCELLENT, 1.5-1.8 GOOD, 1.0-1.5 AVERAGE, <1.0 POOR (seuil pro 2025: 1.80)
- Sortino ≥2.80 EXCELLENT, 2.0-2.8 GOOD, 1.5-2.0 AVERAGE, <1.5 POOR (plus honnête que Sharpe)
- Calmar ≥1.50 EXCELLENT, 1.0-1.5 GOOD, 0.5-1.0 AVERAGE, <0.5 POOR (gain/MaxDD)
- Profit Factor ≥2.00 EXCELLENT, 1.8-2.0 GOOD, 1.3-1.8 AVERAGE, <1.3 POOR
- Max Drawdown ≤-12% EXCELLENT, -12 à -18% GOOD, -18 à -25% AVERAGE, >-25% POOR
- Recovery Factor ≥6.0 EXCELLENT, 4.0-6.0 GOOD, 2.0-4.0 AVERAGE, <2.0 POOR (NetProfit/MaxDD)
- Win Rate ≥68% EXCELLENT (mean-reversion), ≥58% GOOD (trend), <50% POOR
- Expectancy ≥0.8% EXCELLENT, 0.5-0.8% GOOD, 0.2-0.5% AVERAGE, <0.2% POOR (par trade)
- SQN ≥2.8 EXCELLENT, 2.0-2.8 GOOD, 1.5-2.0 AVERAGE, <1.5 POOR (Van Tharp Quality Number)
- Outlier-Adjusted Sharpe ≥1.4 EXCELLENT (Sharpe sans 3 meilleurs trades → détecte dépendance luck)

**TIER A (Validation):**
- R-Multiple (Avg Win/Loss) ≥2.0 GOOD
- % temps en marché ≤45% GOOD (plus de cash = moins stress)
- Plus long flat period ≤35 jours GOOD (>60j = stratégie morte en range)

**Sample size:**
- >100 trades SUFFICIENT, 50-100 MARGINAL, <50 INSUFFICIENT

**Overfitting detection:**
- Win rate >80% suspect (curve fitting)
- Sharpe >3.5 irréaliste (sauf HFT)
- Outlier-Adjusted Sharpe chute >30% vs Sharpe normal = dépendance aux coups de chance
- Plus grosse perte consécutive >-12% = danger

Sois factuel, quantifié, sans concession sur la rigueur. Standards 2025 = seuils professionnels élevés."""

STRATEGIST_SYSTEM_PROMPT = """Tu es un stratège créatif en optimisation de stratégies trading.

Ton rôle:
- Proposer modifications paramètres innovantes
- Générer solutions créatives basées sur analyse
- Respecter contraintes techniques (min/max params)
- Maximiser ratio amélioration/risque

Principes:
- Créativité contrôlée (innovations testables)
- Justifications techniques solides
- Diversité propositions (conservateur ↔ agressif)
- Température élevée = Exploration solutions

Format de sortie: JSON structuré uniquement."""

STRATEGIST_PROPOSAL_PROMPT = """Génère des propositions d'amélioration pour cette stratégie:

**Analyse actuelle (Analyst):**
{analysis_json}

**Paramètres actuels:**
{current_params}

**Contraintes registry:**
{param_constraints}

**Historique récent (5 dernières itérations):**
{memory_recent}

**Objectif:** Propose {n_proposals} configurations qui améliorent le Sharpe ratio.

**Format de réponse (JSON STRICT):**
```json
{{
  "propositions": [
    {{
      "id": 1,
      "type": "ajustement_params|nouvelle_approche|combinaison",
      "modifications": {{
        "fast_period": 14,
        "slow_period": 28,
        "stop_loss_pct": 1.5,
        "..."
      }},
      "rationale": "Augmenter fast_period 8→14 réduit faux signaux (vu dans analyse: 'trop de trades perdants courts'). Impact estimé: -30% trades mais +20% win rate.",
      "impact_estime": "Sharpe +0.3, Drawdown -5%, Win rate +15%",
      "risk_level": "LOW|MEDIUM|HIGH",
      "confidence": 0.0-1.0,
      "expected_sharpe": 1.8
    }},
    {{
      "id": 2,
      "..."
    }},
    {{
      "id": 3,
      "..."
    }}
  ],
  "strategy_reasoning": "Vue d'ensemble: pourquoi ces 3 propositions forment un ensemble cohérent",
  "exploration_vs_exploitation": "EXPLORE (nouvelles zones params) | EXPLOIT (raffiner zone actuelle)"
}}
```

**Consignes:**
1. **Éviter reproposer configs déjà testées** (voir memory_recent)
2. **Respecter contraintes** (ex: fast_period < slow_period)
3. **Diversifier**: 1 conservateur (changements mineurs), 1 équilibré, 1 agressif (changements majeurs)
4. **Justifier quantitativement**: référence analyse Analyst
5. **Estimer impact**: projections réalistes (pas +100% Sharpe)

Sois créatif, pragmatique, et ancré dans l'analyse fournie."""

CRITIC_SYSTEM_PROMPT = """Tu es un validateur rigoureux de propositions trading.

Ton rôle:
- Filtrer propositions irréalistes ou risquées
- Détecter overfitting potentiel
- Valider cohérence technique (contraintes, plausibilité)
- Scorer confiance dans chaque proposition

Principes:
- Rigueur > Optimisme
- Détecter biais de survivance
- Vérifier plausibilité métriques
- Température basse = Validation stricte

Format de sortie: JSON structuré uniquement."""

CRITIC_VALIDATION_PROMPT = """Valide ces propositions d'optimisation:

**Propositions à valider:**
{proposals_json}

**Analyse contexte (Analyst):**
{analysis_json}

**Objectif:** Filtre propositions irréalistes, risquées ou incohérentes.

**Format de réponse (JSON STRICT):**
```json
{{
  "propositions_validees": [1, 3],
  "propositions_rejetees": [
    {{
      "id": 2,
      "raison": "Impact estimé 'Sharpe +0.8' irréaliste (actuel 1.0 → 1.8 trop optimiste sans changement majeur stratégie)",
      "severity": "HIGH|MEDIUM|LOW"
    }}
  ],
  "scores_confiance": [
    {{"id": 1, "score": 0.85, "justification": "Changements graduels cohérents avec analyse"}},
    {{"id": 3, "score": 0.72, "justification": "Approche innovante mais risque modéré"}}
  ],
  "warnings": [
    "Aucune proposition ne réduit significativement le drawdown (faiblesse majeure détectée)",
    "Overfitting risk si win rate projeté >80%"
  ],
  "global_quality": "EXCELLENT|GOOD|AVERAGE|POOR"
}}
```

**Checklist validation (rejette SI - Standards 2025):**

1. **Métriques Tier S NON atteintes (REJET AUTO):**
   - Sharpe <1.80 (seuil pro 2025)
   - Sortino <2.80 (crypto exige mieux)
   - Calmar <1.50 (gain/douleur insuffisant)
   - Profit Factor <2.00 (frais vont tuer)
   - Max Drawdown >-18% (trop de douleur)
   - Recovery Factor <6.0 (gain/DD ratio faible)
   - SQN <2.8 (qualité système insuffisante)
   - Expectancy <0.8% par trade
   - Outlier-Adjusted Sharpe <1.4 (dépendance luck)

2. **Métriques irréalistes (overfitting suspect):**
   - Sharpe >3.5 (sauf HFT haute fréquence)
   - Win rate >80% (curve fitting probable)
   - Drawdown <-5% avec Sharpe >2.5 (too perfect)
   - Outlier-Adjusted Sharpe chute >30% vs Sharpe (dépendance aux 3 meilleurs trades)
   - Plus grosse perte consécutive >-12% (danger séries noires)

3. **Incohérences techniques:**
   - Violations contraintes (ex: fast >= slow)
   - Paramètres hors range réaliste (ex: period=500 sur 1h)
   - Contradictions internes (augmente risque + réduit drawdown?)
   - Impact estimé >50% amélioration sur 1 seul param (irréaliste)

4. **Risques trading:**
   - Stop loss <0.3% (slippage > protection)
   - % temps en marché >60% (stress constant)
   - Plus long flat period >60 jours (stratégie morte en range)
   - Leverage implicite >3x
   - Paramètres ultra-sensibles (minor change = crash)

5. **Overfitting patterns:**
   - Trop d'ajustements simultanés (>5 params modifiés)
   - Optimisation sur bruit (période <30 jours)
   - Courbe equity trop lisse (probabilité <1%)
   - R-Multiple <1.5 (ratio risque/rendement insuffisant)
   - Robustesse multi-token: si Sharpe moyen sur BTC/ETH/SOL <1.6 = rejet

**Seuils promotion "AI-Evolved-Gold" (ThreadX 2025):**
Si stratégie passe TOUS les seuils Tier S sur:
- ≥3 tokens différents (BTC, ETH, SOL minimum)
- ≥3 timeframes (ex: 15m, 1h, 4h)
→ Tag automatique pour dossier principal

**Ton:** Constructif mais implacable. Standards 2025 = excellence requise. Mieux vaut rejeter proposition douteuse que risquer capital."""

ORCHESTRATOR_CONVERGENCE_PROMPT = """Évalue la convergence de l'optimisation autonome:

**Historique itérations:**
{iterations_history}

**Stats convergence:**
{convergence_stats}

**Objectif:** Détermine si optimisation a convergé ou doit continuer.

**Format de réponse (JSON STRICT):**
```json
{{
  "converged": true|false,
  "reason": "Stagnation 3 cycles|Target Sharpe 2.0 atteint|Exploration exhaustive",
  "confidence": 0.0-1.0,
  "next_action": "STOP|CONTINUE|CHANGE_STRATEGY",
  "recommendations": [
    "Si STOP: Best config trouvée, valider sur out-of-sample",
    "Si CONTINUE: Explorer zone params [14-18, 28-35]",
    "Si CHANGE_STRATEGY: Switching régime (trend → mean-reversion)"
  ]
}}
```

**Critères convergence:**
- Stagnation: N cycles sans amélioration >5%
- Optimal local: Toutes propositions validées testées
- Target atteint: Sharpe >= objectif
- Diminishing returns: Amélioration <2% sur 5 cycles"""

