# 🤖 Guide Système Multi-Agents Autonome ThreadX

## Vue d'ensemble

Le système multi-agents de ThreadX permet d'**optimiser automatiquement les stratégies de trading** en faisant collaborer 3 LLM spécialisés qui **conversent entre eux** pour trouver les meilleurs paramètres.

---

## 🎯 Les 3 Agents Spécialisés

### 📊 **Analyst** (deepseek-r1:70b)
**Rôle** : Diagnostiquer les problèmes
- Analyse les résultats de backtest (Sharpe, Drawdown, Win Rate...)
- Identifie les points faibles (ex: "trop de trades", "stop-loss trop serré")
- Fournit un diagnostic détaillé au Strategist

**Exemple de sortie** :
```
Diagnostic (Score 7/10):
- Sharpe ratio 1.65 (cible: 1.8) → Amélioration nécessaire
- Max Drawdown -12% (bon)
- Win Rate 58% (acceptable)
Problème principal: Profit factor insuffisant (1.4, cible > 2.0)
Recommandation: Augmenter take-profit et filtrer trades faible qualité
```

### 💡 **Strategist** (gpt-oss:20b)
**Rôle** : Proposer des solutions créatives
- Reçoit le diagnostic de l'Analyst
- Propose 3 modifications de paramètres
- Justifie chaque proposition

**Exemple de sortie** :
```
Proposition 1 (score 8.5/10):
Modifications:
  - take_profit_pct: 3.0 → 4.5  (+50%)
  - min_profit_pct: 0.6 → 1.2   (+100%)
Justification: 
  Augmenter les sorties en profit pour améliorer le profit factor.
  Filtrer les petits gains améliore la qualité des trades.
```

### ✅ **Critic** (deepseek-r1:70b)
**Rôle** : Valider et rejeter les mauvaises idées
- Teste chaque proposition sur multi-tokens (BTC, ETH, SOL...)
- Compare avec le baseline
- Rejette si Sharpe < baseline OU Tier S non atteint
- Approuve si amélioration significative

**Exemple de sortie** :
```
Validation Multi-Token:
  BTC: Sharpe 1.72 (+0.07) ✅
  ETH: Sharpe 1.68 (+0.03) ✅
  SOL: Sharpe 1.81 (+0.16) ✅
Verdict: APPROUVÉ (amélioration sur 3/3 tokens)
```

---

## 🔄 Workflow Autonome (Boucle d'Optimisation)

```
1. BACKTEST INITIAL
   ├─ Exécution stratégie avec params initiaux
   └─ Calcul métriques (Sharpe, Drawdown, Win Rate...)

2. ANALYST DIAGNOSIS
   ├─ Analyse résultats
   ├─ Identifie problèmes (score 0-10)
   └─ Génère diagnostic structuré

3. STRATEGIST PROPOSALS
   ├─ Reçoit diagnostic Analyst
   ├─ Propose 3 modifications de params
   └─ Justifie chaque proposition

4. CRITIC VALIDATION
   ├─ Teste chaque proposition (multi-token)
   ├─ Compare avec baseline
   ├─ Rejette si mauvais résultats
   └─ Approuve si amélioration

5. AUTOPSY (si tout rejeté)
   ├─ Analyse post-mortem des échecs
   ├─ Génère kill rules (éviter erreurs)
   └─ Feedback Strategist pour prochaine iteration

6. CONVERGENCE CHECK
   ├─ Si Sharpe ≥ target (1.8) → SUCCÈS ✅
   ├─ Si stagnation 3+ cycles → ARRÊT
   └─ Sinon → Retour étape 1 (nouvelle iteration)
```

---

## 🚀 Comment l'utiliser dans l'UI Streamlit ?

### Étape 1 : Charger les données
1. Naviguez vers **📊 Chargement des Données**
2. Sélectionnez un symbole (ex: BTCUSDC)
3. Timeframe: 15m ou 1h (recommandé)
4. Période: Au moins 3 mois de données
5. Cliquez sur **"Charger"**

### Étape 2 : Accéder à l'Orchestrator
1. Naviguez vers **🤖 Multi-Agents Autonome**
2. Vérifiez que les données sont chargées (message vert)

### Étape 3 : Configurer l'optimisation
1. **Stratégie** : Choisir la stratégie à optimiser
   - `ma_crossover` (recommandé pour débuter)
   - `bollinger_dual`
   - `amplitude_hunter`

2. **Paramètres** :
   - **Target Sharpe** : 1.8 (Tier S) ou plus ambitieux (2.0+)
   - **Max Iterations** : 5-20 (débuter avec 5)
   - **Convergence Threshold** : 3 (arrêt si stagnation 3 cycles)
   - **Proposals per Iteration** : 3 (nombre de propositions testées)

3. **Paramètres initiaux** : Définir le point de départ
   - Fast Period: 10
   - Slow Period: 30
   - Stop Loss: 1.5%
   - Take Profit: 3.0%

4. Cliquez sur **"💾 Sauvegarder Configuration"**

### Étape 4 : Lancer l'optimisation autonome
1. Cliquez sur **"▶️ Démarrer Optimisation Autonome"**
2. L'orchestrator démarre en arrière-plan (thread séparé)
3. **NE PAS FERMER L'APPLICATION** pendant l'exécution

### Étape 5 : Superviser en temps réel
- **📜 Logs Temps Réel** : Affiche l'activité des agents
- **💻 Code Généré** : Voir les propositions des agents
- **📊 Convergence** : Graphique Sharpe ratio par iteration
- **🎯 Tier S Score** : Validation métriques

### Étape 6 : Arrêter/Pauser
- **⏸️ Pause** : Termine l'iteration en cours puis pause
- **⏹️ Arrêter** : Termine l'iteration puis arrête définitivement
- **Auto-stop** : Si target Sharpe atteint OU convergence

---

## 📋 Prérequis Techniques

### 1. Ollama Running
```bash
# Vérifier si Ollama tourne
ollama list

# Si non installé, installer :
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger les modèles requis
ollama pull deepseek-r1:70b    # ~40GB (Analyst + Critic)
ollama pull gpt-oss:20b        # ~12GB (Strategist)
```

### 2. GPU (optionnel mais recommandé)
- **Nvidia GPU** : Pour backtests rapides
- **VRAM** : Au moins 8GB pour les LLM
- Vérifier : `nvidia-smi`

### 3. Données OHLCV
- Au moins **3 mois** de données historiques
- Timeframe : 15m ou 1h (meilleur compromis)
- Tokens recommandés : BTCUSDC, ETHUSDC, SOLUSDC

---

## 🎯 Objectifs de Performance (Tier S)

Le système vise les **10 métriques Tier S** :

| Métrique | Tier S Target | Description |
|----------|--------------|-------------|
| **Sharpe Ratio** | ≥ 1.8 | Rendement ajusté au risque |
| **Sortino Ratio** | ≥ 2.8 | Rendement vs volatilité négative |
| **Calmar Ratio** | ≥ 1.5 | Rendement vs max drawdown |
| **Profit Factor** | ≥ 2.0 | Gains totaux / Pertes totales |
| **Recovery Factor** | ≥ 3.0 | Profit net / Max drawdown |
| **Expectancy** | ≥ 1.8% | Gain moyen par trade |
| **SQN** | ≥ 3.0 | System Quality Number |
| **Outlier Sharpe** | ≥ 1.6 | Sharpe sans outliers |
| **Win Rate** | ≥ 58% | % trades gagnants |
| **Max Drawdown** | ≤ -18% | Perte maximale |

**🏆 AI-EVOLVED-GOLD TAG** : Si 10/10 métriques Tier S validées !

---

## 💡 Exemples d'Utilisation

### Exemple 1 : Optimisation rapide (Quick Test)
```
Stratégie: ma_crossover
Max Iterations: 5
Target Sharpe: 1.8
Convergence: 3

→ Durée estimée: 10-15 minutes
→ Résultat attendu: Sharpe 1.6-1.9
```

### Exemple 2 : Optimisation profonde (Research)
```
Stratégie: bollinger_dual
Max Iterations: 20
Target Sharpe: 2.0
Convergence: 5

→ Durée estimée: 1-2 heures
→ Résultat attendu: Sharpe 1.8-2.2
→ Tier S possible
```

### Exemple 3 : Multi-token robustesse
```
Stratégie: amplitude_hunter
Test sur: BTC, ETH, SOL, DOGE
Max Iterations: 15
Target Sharpe: 1.9

→ Le Critic valide sur 4 tokens simultanément
→ Garantit robustesse cross-market
```

---

## 🔬 Logs & Debugging

### Comprendre les logs

```
🚀 Orchestrator started: ma_crossover
✅ Orchestrator initialized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 ITERATION 1/5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔄 Running baseline backtest...
✅ Baseline: Sharpe 1.65, Drawdown -12%

📊 Analyst analyzing results...
✅ Diagnosis complete (score: 7.5/10)

💡 Strategist proposing modifications...
✅ 3 proposals generated

✅ Critic validating proposals...
   Proposal 1: Sharpe 1.72 (+0.07) ✅ APPROVED
   Proposal 2: Sharpe 1.68 (+0.03) ✅ APPROVED  
   Proposal 3: Sharpe 1.61 (-0.04) ❌ REJECTED

🎯 Best proposal selected: Sharpe 1.72
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Erreurs courantes

**Erreur : Ollama not running**
```
Solution: Démarrer Ollama
$ ollama serve
```

**Erreur : Model not found**
```
Solution: Télécharger les modèles
$ ollama pull deepseek-r1:70b
$ ollama pull gpt-oss:20b
```

**Erreur : Aucune donnée chargée**
```
Solution: Charger données sur page "Configuration"
```

**Stagnation (pas d'amélioration)**
```
Solution 1: Augmenter proposals_per_iteration (5 au lieu de 3)
Solution 2: Changer paramètres initiaux
Solution 3: Essayer autre stratégie
```

---

## 📊 Interpréter les Résultats

### Graphique de Convergence

```
Sharpe
  2.0 ┤           ╭─────────
      │         ╭─╯
  1.8 ┤  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Target
      │    ╭───╯
  1.6 ┤ ──╯
      └────────────────────────
       1  2  3  4  5  Iterations
```

✅ **Bonne convergence** : Progression régulière vers target
⚠️ **Stagnation** : Plateau sans amélioration → arrêt auto
❌ **Divergence** : Dégradation → vérifier paramètres

### Dashboard Tier S

- **Score 0-50** : ❌ Mauvaise stratégie
- **Score 50-70** : ⚠️ Acceptable mais améliorable
- **Score 70-85** : ✅ Bonne stratégie (Tier A/B)
- **Score 85-100** : 🏆 Excellente stratégie (Tier S)

---

## 🚀 Workflow Complet Recommandé

1. **Charger données** (3-6 mois, 15m timeframe)
2. **Backtest manuel** sur page Optimisation (comprendre baseline)
3. **Lancer Multi-Agents** avec target modeste (Sharpe 1.6)
4. **Superviser** 5-10 iterations
5. **Analyser résultats** (graphique convergence + Tier S)
6. **Si succès** : Exporter paramètres optimaux
7. **Si échec** : Consulter Autopsy feedback + retry

---

## 🎓 Apprentissage Automatique

Le système **apprend de ses erreurs** via l'**Autopsy Agent** :

1. **Après 3 rejets consécutifs** → Autopsy post-mortem
2. **Analyse patterns d'échecs** (ex: "trop de trades")
3. **Génère Kill Rules** (ex: "rejeter si trades > 30")
4. **Feedback au Strategist** → Évite erreurs futures

**Résultat** : Le système devient **plus intelligent** au fil des iterations !

---

## 📚 Pour Aller Plus Loin

- **Architecture détaillée** : `ARCHITECTURE_MULTI_LLM.md`
- **Autopsy System** : `README_AUTOPSY_SYSTEM.md`
- **Context Manager** : `README_CONTEXT_INTELLIGENT.md`
- **POC Scripts** : `tools/poc_orchestrator.py`

---

**Développé avec ❤️ par ThreadX Framework**  
Version: 2.0 - Multi-Agents Autonome
