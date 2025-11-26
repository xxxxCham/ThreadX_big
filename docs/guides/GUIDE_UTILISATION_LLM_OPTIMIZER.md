# 🤖 Guide d'Utilisation - Multi-LLM Optimizer

## 📋 Vue d'Ensemble

Le **Multi-LLM Optimizer** est un système collaboratif d'agents IA pour optimiser automatiquement les stratégies de trading. Il utilise deux agents spécialisés :

- **🧠 Analyst** (deepseek-r1:70b) : Analyse quantitative des résultats de backtests
- **🎨 Strategist** (gpt-oss:20b) : Génération créative de propositions d'optimisation

---

## 🚀 Démarrage Rapide

### 1️⃣ Accès à l'Interface

```bash
# Depuis le dossier ThreadX_big
streamlit run src/threadx/streamlit_app.py
```

**Navigation** : 
1. Page **Configuration** → Charger données
2. Page **Backtest & Optimization** → (optionnel) Tester manuellement
3. Page **🤖 Multi-LLM Optimizer** ← **NOUVELLE PAGE**
4. Page **Monitoring** → Suivre performances système

---

## ⚙️ Configuration par Défaut (Préprogrammée)

### 📊 Stratégie MA_Crossover

La page s'initialise automatiquement avec **MA_Crossover** et les paramètres optimaux :

| Paramètre | Valeur Fixe | Plage | Description |
|-----------|-------------|-------|-------------|
| `max_hold_bars` | **20** | 300-300 | Durée max en position (via override) |
| `risk_per_trade` | **0.005** | 0.02-0.02 | Risque par trade (0.5% du capital) |
| **Analyse IA** | ✅ **Activée** | - | Checkbox cochée par défaut |

> **Note** : Les valeurs 300-300 et 0.02-0.02 dans les sliders sont des plages techniques. Les **vraies valeurs** utilisées sont **20** et **0.005** grâce aux overrides dans `strategy_registry.py` (lignes 869-877).

---

## 🔄 Workflow Complet

### Étape 1 : Configuration Sweep 📋

**Automatique** :
- Stratégie : `MA_Crossover` (pré-sélectionnée)
- Paramètres : Valeurs fixées selon screenshots
- Total configs : Calculé automatiquement

**Personnalisable** :
```python
# Pour autres paramètres (fast_period, slow_period, etc.)
- Ajuster nombre de valeurs (slider 2-6)
- Le système génère des combinaisons uniformes
```

---

### Étape 2 : Configuration LLM 🤖

| Option | Défaut | Alternative |
|--------|--------|-------------|
| **Modèle Analyst** | `deepseek-r1:70b` | `gemma3:27b`, `qwen3-vl:30b` |
| **Modèle Strategist** | `gpt-oss:20b` | `gpt-oss:120b-cloud`, `gemma3:27b` |
| **Nombre propositions** | 3 | 1-5 |
| **Top N configs** | 5 | 3-10 |
| **GPU** | ✅ Activé | - |

---

### Étape 3 : Lancer l'Optimisation 🚀

**Cliquer** : Bouton **"🚀 Lancer l'optimisation Multi-LLM"**

**Déroulement** (2-5 minutes) :

```
[Progress Bar: 0%] 🔄 Exécution du sweep...
                  └─ Teste toutes les combinaisons de paramètres
                  └─ GPU accéléré (RTX 5090 + RTX 2060)
                  └─ Résultats : sharpe_ratio, drawdown, win_rate, etc.

[Progress Bar: 40%] 🧠 Analyse par Analyst...
                   └─ deepseek-r1:70b analyse top 5 configs
                   └─ Identifie patterns communs
                   └─ Calcule métriques agrégées
                   └─ Affichage streaming en chat

[Progress Bar: 70%] 🎨 Propositions par Strategist...
                   └─ gpt-oss:20b génère 3 propositions
                   └─ Approches : Conservative / Aggressive / Exploratoire
                   └─ Valide contraintes (min/max, risk_per_trade)
                   └─ Affichage streaming en chat

[Progress Bar: 90%] ✅ Tests automatiques...
                   └─ Teste chaque proposition sur mêmes données
                   └─ Compare avec baseline
                   └─ Calcule métriques complètes

[Progress Bar: 100%] 📊 Rapport final
                    └─ Graphiques Plotly interactifs
                    └─ Comparaison Sharpe / Return / Drawdown
                    └─ Recommandation meilleure config
```

---

## 📊 Interprétation des Résultats

### 🧠 Analyse Analyst

**Format** : Chat message avec avatar 🧠

**Sections** :
```json
{
  "patterns": [
    "short_period < 15 dans 4/5 top configs",
    "long_period entre 30-40 pour Sharpe > 1.5"
  ],
  "key_metrics": {
    "avg_sharpe": 1.82,
    "max_drawdown_avg": -8.3,
    "avg_win_rate": 0.57
  },
  "trade_offs": [
    "Sharpe élevé mais drawdown important (configs #1, #3)"
  ],
  "recommendations": [
    "Explorer short_period 8-12 (zone peu testée)",
    "Augmenter long_period pour stabilité"
  ]
}
```

**Utilité** :
- Comprendre **pourquoi** certaines configs performent
- Identifier **corrélations** entre paramètres
- Détecter **trade-offs** (rendement vs risque)

---

### 🎨 Propositions Strategist

**Format** : Chat message avec avatar 🎨

**Structure** :
```json
{
  "proposals": [
    {
      "name": "Conservative",
      "params": {
        "fast_period": 12,
        "slow_period": 35,
        "risk_per_trade": 0.005,
        "max_hold_bars": 25
      },
      "rationale": "Réduit drawdown en augmentant slow_period (+5). Maintient risk_per_trade à 0.5% pour stabilité."
    },
    {
      "name": "Aggressive",
      "params": {
        "fast_period": 8,
        "slow_period": 28,
        "risk_per_trade": 0.015,
        "max_hold_bars": 15
      },
      "rationale": "Exploite pattern 'short_period < 10' observé. Augmente risque à 1.5% pour maximiser rendement."
    },
    {
      "name": "Exploratoire",
      "params": {
        "fast_period": 15,
        "slow_period": 45,
        "risk_per_trade": 0.01,
        "max_hold_bars": 30
      },
      "rationale": "Teste zone peu explorée (fast_period > 12). Équilibre risque/rendement."
    }
  ]
}
```

**Affichage** : Expandable par proposition avec métriques de test

---

### 📊 Rapport Final Visuel

**3 Graphiques Plotly** (barres comparatives) :

1. **Sharpe Ratio** :
   - Baseline (config actuelle)
   - 3 propositions LLM
   - Couleur : Bleu (baseline) / Vert (meilleure proposition)

2. **Total Return %** :
   - Même structure
   - Identifie proposition la plus rentable

3. **Max Drawdown %** :
   - Même structure
   - Identifie proposition la plus stable

**Légende** :
- 📊 **Baseline** : Configuration de référence (meilleure du sweep)
- ✅ **Meilleure Proposition** : Surlignée en vert (Sharpe le plus élevé)

---

## 🎯 Consignes Système pour LLM

### 📋 Intégrées Automatiquement

Les agents LLM suivent ces principes (affichés dans l'expandable "Consignes pour les Agents LLM") :

#### 🎯 Objectifs Prioritaires
- **Sharpe Ratio** : Maximiser risque/rendement
- **Drawdown** : Minimiser perte maximale
- **Win Rate** : Maintenir > 50%
- **Nombre Trades** : Éviter over/under-trading

#### 📊 Approche d'Analyse
- Identifier **patterns reproductibles**
- Détecter **corrélations** entre paramètres
- Proposer modifications **incrémentielles** (pas de sauts brutaux)
- Valider **cohérence** avec contraintes de risque

#### ⚠️ Contraintes Critiques
| Contrainte | Plage | Justification |
|------------|-------|---------------|
| `risk_per_trade` | **[0.005, 0.02]** | Gestion risque stricte (0.5%-2% capital) |
| `max_hold_bars` | **[20, 150]** | Adapter selon volatilité |
| **Ratio SL/TP** | **≥ 1:1.5** | Asymétrie favorable (gain > perte) |
| **Min/Max params** | **Respecter TOUJOURS** | Éviter valeurs hors plage technique |

#### 💡 Principes
- **Robustesse > Performance** : Éviter overfitting
- **Documentation claire** : Expliquer chaque modification
- **3 approches** : Conservative (stabilité) / Aggressive (rendement) / Exploratoire (découverte)

---

## 🔧 Personnalisation Avancée

### Modifier les Presets MA_Crossover

**Fichier** : `src/threadx/ui/page_llm_optimizer.py` (lignes 64-67)

```python
# Exemple : Tester plages variables
ma_crossover_presets = {
    "max_hold_bars": {"min": 15, "max": 30, "n_values": 4},  # 4 valeurs entre 15-30
    "risk_per_trade": {"min": 0.005, "max": 0.015, "n_values": 3}  # 3 valeurs
}
```

**Impact** : Génère `4 × 3 = 12` configs au lieu de 1

---

### Ajouter Consignes Personnalisées

**Fichier** : `src/threadx/llm/agents/analyst.py` (lignes 82-104)

```python
# Exemple : Ajouter priorité sur win rate
system_instructions = """
...
🎯 OBJECTIF SUPPLÉMENTAIRE:
- Win rate > 60% (priorité absolue)
...
"""
```

**Fichier** : `src/threadx/llm/agents/strategist.py` (lignes 91-113)

---

### Changer Modèles LLM

**Option 1** : Via interface Streamlit (dropdowns)

**Option 2** : Modifier fichiers agents

```python
# analyst.py ligne 24
def __init__(self, model: str = "gemma3:27b", ...):  # Au lieu de deepseek-r1:70b

# strategist.py ligne 24
def __init__(self, model: str = "gpt-oss:120b-cloud", ...):  # Au lieu de gpt-oss:20b
```

---

## ⚙️ Prérequis Techniques

### 1️⃣ Ollama en Exécution

```bash
# Vérifier si Ollama tourne
ollama list

# Si non démarré
ollama serve
```

### 2️⃣ Modèles Téléchargés

```bash
# Analyst
ollama pull deepseek-r1:70b

# Strategist
ollama pull gpt-oss:20b

# Alternatives
ollama pull gemma3:27b
ollama pull qwen3-vl:30b
ollama pull gpt-oss:120b-cloud
```

### 3️⃣ GPU Activé (Recommandé)

- **Détection automatique** au lancement Streamlit
- Logs : `[INFO] CuPy détecté - Support GPU activé`
- Si GPU non disponible : Backtests CPU (plus lents)

---

## 🐛 Troubleshooting

### Erreur : "Connection refused (Ollama)"

**Cause** : Ollama non démarré

**Solution** :
```bash
ollama serve
# Dans un autre terminal, relancer Streamlit
streamlit run src/threadx/streamlit_app.py
```

---

### Erreur : "Model not found: deepseek-r1:70b"

**Cause** : Modèle non téléchargé

**Solution** :
```bash
ollama pull deepseek-r1:70b
ollama pull gpt-oss:20b
```

---

### Propositions Identiques / Non Créatives

**Cause** : Temperature trop basse

**Solution** : Modifier `strategist.py` ligne 129 :
```python
temperature=0.9,  # Au lieu de 0.8 (plus créatif)
```

---

### Analyse Trop Factuelle / Peu Insights

**Cause** : Temperature trop basse Analyst

**Solution** : Modifier `analyst.py` ligne 125 :
```python
temperature=0.5,  # Au lieu de 0.3 (plus nuancé)
```

---

## 📝 Exemple de Session Complète

### Contexte
- Stratégie : `MA_Crossover`
- Objectif : Optimiser Sharpe tout en limitant drawdown < 10%

### Étapes

1. **Sweep Initial** (40 configs testées)
   - Meilleure config : `fast=10, slow=30, risk=0.005, hold=20`
   - Sharpe : 1.85
   - Drawdown : -9.2%

2. **Analyse Analyst** (top 5 configs)
   ```
   Patterns détectés:
   - fast_period entre 8-12 dans 4/5 configs
   - slow_period entre 28-35 corrélé avec Sharpe > 1.7
   - risk_per_trade = 0.005 optimal (pas de gain à augmenter)
   
   Trade-offs:
   - Config #1 : Sharpe 1.85 mais drawdown -9.2% (limite)
   - Config #2 : Sharpe 1.78 mais drawdown -7.1% (plus stable)
   
   Recommandations:
   - Tester fast_period = 9 (zone peu explorée)
   - Augmenter slow_period à 32-35 pour réduire drawdown
   ```

3. **Propositions Strategist**
   
   **Conservative** :
   - `fast=11, slow=35, risk=0.005, hold=25`
   - Rationale : "Augmente slow_period +5 pour stabilité. Réduit drawdown estimé à -7.5%"
   
   **Aggressive** :
   - `fast=9, slow=28, risk=0.01, hold=18`
   - Rationale : "Exploite pattern fast < 10. Augmente risk à 1% pour rendement. Sharpe estimé 2.1"
   
   **Exploratoire** :
   - `fast=12, slow=40, risk=0.008, hold=30`
   - Rationale : "Teste zone lente (slow=40). Équilibre risque intermédiaire"

4. **Tests Automatiques**
   
   | Proposition | Sharpe | Return | Drawdown | Verdict |
   |-------------|--------|--------|----------|---------|
   | Conservative | **1.92** | 42.1% | **-6.8%** | ✅ Meilleure |
   | Aggressive | 2.05 | 58.3% | -12.4% | ❌ Drawdown trop élevé |
   | Exploratoire | 1.73 | 35.2% | -7.9% | ⚠️ Sharpe inférieur |

5. **Décision**
   - **Sélectionner** : Proposition Conservative
   - **Justification** : Améliore Sharpe (+0.07) ET réduit drawdown (-2.4%)
   - **Validation** : Drawdown < 10% (objectif respecté)

---

## 🔄 Workflow Itératif

Le système peut être utilisé en boucle :

```
1️⃣ Sweep initial (plage large) → Meilleure config A
2️⃣ Multi-LLM sur config A → Proposition B
3️⃣ Nouveau sweep centré sur B → Meilleure config C
4️⃣ Multi-LLM sur config C → Proposition D
...
```

**Convergence** : Généralement 2-3 itérations suffisent

---

## 📚 Fichiers Importants

| Fichier | Description | Lignes Clés |
|---------|-------------|-------------|
| `page_llm_optimizer.py` | Interface Streamlit | 64-67 (presets), 145-181 (consignes) |
| `analyst.py` | Agent analyse quantitative | 82-104 (system instructions) |
| `strategist.py` | Agent propositions créatives | 91-113 (system instructions) |
| `strategy_registry.py` | Définitions stratégies | 764-877 (MA_Crossover), 869-877 (overrides) |
| `backtest_bridge.py` | Wrapper GPU backtests | 187-270 (run_backtest_gpu) |

---

## ✅ Checklist Avant Lancement

- [ ] Ollama démarré (`ollama serve`)
- [ ] Modèles téléchargés (`ollama list`)
- [ ] GPU détecté (logs Streamlit)
- [ ] Données chargées (page Configuration)
- [ ] Stratégie MA_Crossover sélectionnée
- [ ] Checkbox "Analyse IA" cochée
- [ ] Bouton "🚀 Lancer" cliqué

**Temps estimé** : 2-5 minutes selon nombre de configs

---

## 🎓 Apprentissage

### Débutant
1. Lancer avec paramètres par défaut
2. Observer les patterns dans l'analyse Analyst
3. Comparer propositions Strategist avec baseline

### Intermédiaire
1. Modifier nombre de valeurs par paramètre
2. Tester différents modèles LLM
3. Analyser trade-offs dans les graphiques

### Avancé
1. Personnaliser consignes système
2. Ajouter métriques custom dans prompts
3. Implémenter boucle d'optimisation itérative

---

## 📞 Support

**Logs détaillés** : Console PowerShell où Streamlit tourne

**Erreurs communes** : Voir section Troubleshooting ci-dessus

**Documentation code** : Docstrings dans chaque fichier Python

---

**Dernière mise à jour** : 15 novembre 2025  
**Version** : v1.0 - Multi-LLM Optimizer  
**Branche Git** : `llm`
