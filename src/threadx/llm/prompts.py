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
