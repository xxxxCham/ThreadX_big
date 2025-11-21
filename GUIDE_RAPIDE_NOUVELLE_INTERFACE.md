# 🚀 Guide Rapide - Nouvelle Interface Optimisation

## ✅ Changements Effectués

### Page "Optimisation" - 3 Onglets Distincts

```
AVANT (2 onglets)          →    APRÈS (3 onglets)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 Sweep                        🔬 Sweep Classique
🎲 Monte-Carlo                  🤖 Sweep + LLM (nouveau)
🐛 Backtest (bug)               🧠 Multi-Agents (nouveau)
```

## 🎯 Comment Utiliser

### 1. Lancer l'Application

```bash
# Terminal 1 (si Multi-Agents nécessaire)
ollama serve

# Terminal 2
cd D:\ThreadX_big
streamlit run src\threadx\streamlit_app.py
```

### 2. Navigation

```
Étapes :
1. Charger données (Page 1 : Configuration)
2. Cliquer "Optimisation" (Page 2)
3. Choisir un des 3 onglets
```

### 3. Choix de l'Onglet

#### 🔬 Sweep Classique (Recommandé pour débuter)
**Utiliser si** :
- ✅ Première fois sur ThreadX
- ✅ Veux contrôler manuellement
- ✅ Comprendre chaque paramètre

**Workflow** :
1. Sélectionner stratégie (ex: Bollinger_Breakout)
2. Charger template "Quick Test"
3. Ajuster sensibilité à 0.5x
4. Activer GPU + Multi-GPU
5. Cliquer "Lancer Sweep"

**Temps** : ~2 minutes

---

#### 🤖 Sweep + LLM (En développement)
**Statut** : Placeholder

**Utiliser si** :
- ⏳ Veux tester futures fonctionnalités
- ⏳ Voir roadmap LLM-Assisted

**Fonctionnalités prévues** :
- Analyse LLM post-sweep
- Suggestions amélioration
- Re-run automatique

**Temps** : Non disponible

---

#### 🧠 Multi-Agents Autonome (Avancé)
**Utiliser si** :
- ✅ Expérience avec ThreadX
- ✅ Veux système entièrement automatique
- ✅ 3 agents collaboratifs (Analyst, Strategist, Critic)

**Workflow** :
1. Cliquer "Ouvrir Multi-LLM Optimizer"
2. Sélectionner stratégie (9 disponibles)
3. Mode Simple : Paramètres auto-chargés
4. Mode Avancé : JSON personnalisé
5. Lancer orchestration autonome

**Temps** : ~5 minutes setup + boucle continue

---

## 🐛 Bugs Corrigés

### ✅ Onglet Backtest fantôme
**Avant** : Onglet "Backtest" apparaissait/disparaissait
**Après** : Supprimé, 3 onglets stables

### ✅ NameError _render_config_history
**Avant** : Erreur au chargement page
**Après** : Fonction corrigée, historique fonctionne

### ⚠️ UnicodeDecodeError
**Statut** : Non-bloquant (Ollama interne)
**Action** : Voir `FIX_UNICODE_SUBPROCESS_ERRORS.md`

---

## 📊 Quelle Approche Choisir ?

```
Débutant → 🔬 Sweep Classique
   ↓ (Quick Test, 100 combos)
   ↓
Intermédiaire → 🔬 Sweep Classique
   ↓ (Standard, 10k combos)
   ↓
Avancé → 🧠 Multi-Agents Autonome
   ↓ (Boucle infinie, 3 agents)
   ↓
Expert → 🤖 Sweep + LLM (bientôt)
```

---

## 🔧 Troubleshooting

### Problème : "Onglet Backtest encore visible"
**Solution** : 
```bash
# Recharger page Streamlit
Ctrl+R (navigateur)
# OU
streamlit cache clear
streamlit run src\threadx\streamlit_app.py
```

### Problème : "Erreur UnicodeDecodeError dans logs"
**Solution** : Ignorer (non-bloquant)
```powershell
# OU rediriger stderr Ollama
ollama serve 2>$null
```

### Problème : "Multi-Agents ne redirige pas"
**Solution** :
```python
# Vérifier session_state
st.session_state["page"] = "multi_llm"
st.rerun()
```

---

## 📚 Documentation Complète

- **Résumé visuel** : `RESUME_RESTRUCTURATION_3_ONGLETS.md`
- **Plan de test** : `TEST_NOUVELLE_STRUCTURE_3_ONGLETS.md`
- **Fix Unicode** : `FIX_UNICODE_SUBPROCESS_ERRORS.md`

---

## ✅ Checklist Validation

- [ ] Lancer Streamlit sans erreur
- [ ] 3 onglets visibles (pas de Backtest)
- [ ] Sweep Classique : Templates fonctionnent
- [ ] Sweep + LLM : Placeholder affiché
- [ ] Multi-Agents : Bouton redirige vers page dédiée
- [ ] Navigation fluide entre onglets
- [ ] Historique configurations sauvegardé

---

## 🎉 Félicitations !

Votre interface d'optimisation est maintenant **propre, stable et intuitive** avec :

- ✅ **3 onglets distincts** sans confusion
- ✅ **Bugs d'affichage corrigés**
- ✅ **Documentation complète**
- ✅ **Workflow clair** pour tous niveaux

**Prochaine étape** : Tester Sweep Classique avec vos données !

---

**Version** : ThreadX v2.0  
**Commits** : 1e0e884, 000486c  
**Date** : 21 novembre 2025
