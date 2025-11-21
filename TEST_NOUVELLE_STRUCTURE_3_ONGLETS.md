# 🧪 Test Plan - Nouvelle Structure 3 Onglets

## 📋 Objectif
Valider la nouvelle organisation de la page "Optimisation" avec 3 onglets distincts.

## 🎯 Structure implémentée

### Page 2 : 🔬 Optimisation de Stratégies

```
Onglet 1 : 🔬 Sweep Classique
├── Balayage exhaustif de paramètres
├── Configuration manuelle des plages
├── Templates (Quick Test, Standard, Exhaustif)
├── Historique des configurations
├── Contrôle GPU/Multi-GPU
└── Estimation temps d'exécution

Onglet 2 : 🤖 Sweep + LLM
├── Interface placeholder (en développement)
├── Roadmap visible
└── Redirection vers Sweep Classique ou Multi-Agents

Onglet 3 : 🧠 Multi-Agents Autonome
├── Bouton de navigation vers page dédiée
├── Aperçu fonctionnalités (orchestration, 3 agents)
└── Liste des 9 stratégies supportées
```

## ✅ Tests à effectuer

### Test 1 : Chargement de la page
**Étapes** :
1. Lancer `streamlit run src/threadx/streamlit_app.py`
2. Charger données (page 1)
3. Naviguer vers page 2 "Optimisation"

**Résultats attendus** :
- ✅ 3 onglets visibles : Sweep Classique, Sweep + LLM, Multi-Agents Autonome
- ✅ Aucun onglet "Backtest" fantôme
- ✅ Pas d'erreur `NameError: _render_config_history`

### Test 2 : Onglet Sweep Classique
**Étapes** :
1. Cliquer sur "🔬 Sweep Classique"
2. Sélectionner stratégie "Bollinger_Breakout"
3. Charger template "Quick Test"
4. Ajuster sensibilité globale à 0.5x

**Résultats attendus** :
- ✅ Titre : "Optimisation par Sweep (Grille Exhaustive)"
- ✅ Templates fonctionnels (sensibilité change)
- ✅ Historique des configurations visible
- ✅ Sliders paramètres affichés
- ✅ Estimation temps < 1 minute

### Test 3 : Onglet Sweep + LLM
**Étapes** :
1. Cliquer sur "🤖 Sweep + LLM"
2. Lire message d'information
3. Ouvrir expander "Roadmap"

**Résultats attendus** :
- ✅ Message : "Fonctionnalité en cours de développement"
- ✅ Liste roadmap avec 5 étapes
- ✅ Suggestion d'utiliser Sweep Classique ou Multi-Agents
- ✅ Pas d'erreur de rendu

### Test 4 : Onglet Multi-Agents Autonome
**Étapes** :
1. Cliquer sur "🧠 Multi-Agents Autonome"
2. Lire description
3. Cliquer bouton "Ouvrir Multi-LLM Optimizer"

**Résultats attendus** :
- ✅ Message : "Système Multi-Agents actif sur page dédiée"
- ✅ Descriptions des 3 agents (Analyst, Strategist, Critic)
- ✅ Expander "Aperçu" avec fonctionnalités
- ✅ Bouton redirige vers page `st.session_state["page"] = "multi_llm"`
- ✅ Liste 9 stratégies avec emojis ⭐🏆⚡

### Test 5 : Navigation entre onglets
**Étapes** :
1. Cliquer sur "Sweep Classique"
2. Sélectionner "MA_Crossover" + modifier plages
3. Cliquer sur "Sweep + LLM"
4. Revenir sur "Sweep Classique"

**Résultats attendus** :
- ✅ Configuration préservée (stratégie + plages)
- ✅ Session state intact
- ✅ Pas de reset intempestif
- ✅ Pas de lag/freeze

### Test 6 : Bug d'affichage (régression)
**Étapes** :
1. Naviguer vers page "Optimisation"
2. Observer onglets visibles au chargement
3. Cliquer successivement sur chaque onglet

**Résultats attendus** :
- ❌ **PAS** d'onglet "Backtest" qui apparaît/disparaît
- ❌ **PAS** d'exécution instantanée non sollicitée
- ❌ **PAS** d'erreur `_render_config_history not defined`
- ✅ Navigation fluide sans bugs visuels

## 🔧 Tests techniques

### Test 7 : Historique des configurations
**Étapes** :
1. Dans "Sweep Classique", charger template "Standard"
2. Modifier plages de 3 paramètres
3. Observer expander "Historique des Configurations"
4. (Si historique existe) Cliquer "Charger" sur une config

**Résultats attendus** :
- ✅ Historique affiche timestamp + stratégie + sensibilité
- ✅ Bouton "Charger" restaure configuration
- ✅ Bouton "Suppr." supprime entrée
- ✅ Message "Configuration chargée" si succès

### Test 8 : GPU/Multi-GPU
**Étapes** :
1. Dans "Sweep Classique", activer "Multi-GPU"
2. Mode workers : "Auto (Dynamique)"
3. Sensibilité : 0.5x (Quick Test)
4. Observer estimation temps

**Résultats attendus** :
- ✅ Estimation affiche "~2000 tests/sec" si GPU
- ✅ Boost Multi-GPU : tests/sec × 1.8
- ✅ Message : "GPU: ✅ Multi"
- ✅ Temps estimé < 1 minute pour ~100 combinaisons

## 🐞 Bugs connus (avant correction)

### Bug 1 : NameError _render_config_history ✅ CORRIGÉ
**Avant** :
```python
NameError: name '_render_config_history' is not defined
File "page_backtest_optimization.py", line 1053
```

**Après** :
- Fonction existe à ligne 154
- Appel correct avec `key_prefix="sweep_"`
- Plus d'erreur attendue

### Bug 2 : Onglet Backtest fantôme ✅ CORRIGÉ
**Avant** :
```
Onglets visibles : Backtest, Sweep, Monte-Carlo (3 onglets)
Backtest s'exécute instantanément sans interaction
Clic sur Multi-LLM → Backtest disparaît
```

**Après** :
```
Onglets visibles : Sweep Classique, Sweep + LLM, Multi-Agents (3 onglets)
Aucun backtest automatique
Navigation stable
```

### Bug 3 : UnicodeDecodeError ⚠️ NON-BLOQUANT
**Statut** : Erreur cosmétique (Ollama interne)
**Impact** : Aucun (warnings dans logs)
**Action** : Documenter dans FIX_UNICODE_SUBPROCESS_ERRORS.md

## 📊 Checklist validation complète

### Interface
- [ ] 3 onglets visibles au chargement
- [ ] Titres corrects (Sweep Classique, Sweep + LLM, Multi-Agents)
- [ ] Captions descriptives sous chaque titre
- [ ] Pas d'onglet "Backtest" ni "Monte-Carlo" visible

### Fonctionnalités
- [ ] Sweep Classique : Templates + Historique + Plages
- [ ] Sweep + LLM : Placeholder + Roadmap
- [ ] Multi-Agents : Bouton navigation + Aperçu
- [ ] Session state préservé entre onglets

### Stabilité
- [ ] Aucune erreur Python dans console
- [ ] Aucun rerun intempestif
- [ ] Aucun freeze/lag
- [ ] GPU/Multi-GPU fonctionne

### Documentation
- [ ] FIX_UNICODE_SUBPROCESS_ERRORS.md créé
- [ ] TEST_NOUVELLE_STRUCTURE_3_ONGLETS.md créé
- [ ] Code commenté et lisible

## 🚀 Commandes de test

```bash
# Terminal 1 : Démarrer Ollama (si Multi-Agents nécessaire)
ollama serve

# Terminal 2 : Lancer Streamlit
cd /workspaces/ThreadX_big
streamlit run src/threadx/streamlit_app.py

# Terminal 3 : Surveiller logs
tail -f .streamlit/logs/*.log
```

## 📝 Rapport de test (à remplir après exécution)

**Date** : _________________  
**Version ThreadX** : v2.0  
**Testeur** : _________________

### Résultats
- [ ] Test 1 : Chargement page ✅ ❌
- [ ] Test 2 : Sweep Classique ✅ ❌
- [ ] Test 3 : Sweep + LLM ✅ ❌
- [ ] Test 4 : Multi-Agents Autonome ✅ ❌
- [ ] Test 5 : Navigation onglets ✅ ❌
- [ ] Test 6 : Bug affichage ✅ ❌
- [ ] Test 7 : Historique configs ✅ ❌
- [ ] Test 8 : GPU/Multi-GPU ✅ ❌

### Bugs détectés
```
(Décrire bugs trouvés avec ligne, fichier, traceback)
```

### Notes
```
(Observations, suggestions, améliorations)
```

---

**Version** : 1.0  
**Date création** : 21 novembre 2025  
**Auteur** : GitHub Copilot
