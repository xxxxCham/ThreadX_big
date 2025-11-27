# ✅ Fonctions Système ThreadX - Arrêt & Redémarrage

## 📋 Vue d'ensemble

Deux nouvelles fonctions système disponibles dans la **Sidebar** :

1. **🔄 Redémarrer** : Réinitialise l'application (cache, GPU, session)
2. **🛑 Arrêter** : Ferme l'application proprement

---

## 🔄 Fonction Redémarrage

### Objectif

Redémarre l'application en réinitialisant **TOUT** comme si c'était le premier démarrage.

### Fichier

[src/threadx/streamlit_app.py](src/threadx/streamlit_app.py#L783-L846) : `restart_app()`

### Opérations Effectuées

#### 1. **Arrêt du Monitoring**
```python
monitor = get_global_monitor()
monitor.stop()
```
- Arrête les threads de surveillance système

#### 2. **Nettoyage GPU Manager**
```python
manager = get_default_manager()
manager.stop()
manager.__class__._default_manager = None  # Reset singleton
```
- Arrête les processus GPU
- Réinitialise le singleton pour recréation propre

#### 3. **Nettoyage Mémoire Complet**
```python
clean_all_memory()
```
- **VRAM GPU** : Vide tous les memory pools CuPy
- **Cache IndicatorBank** : Réinitialise le singleton
- **Caches Streamlit** : `cache_data.clear()` + `cache_resource.clear()`
- **RAM système** : Force garbage collection
- **Ollama** : Reset complet

#### 4. **Réinitialisation Session**
```python
all_keys = list(st.session_state.keys())
for key in all_keys:
    del st.session_state[key]
```
- Supprime **TOUTES** les clés (y compris système)
- État vierge comme au premier lancement

#### 5. **Nettoyage Fichiers Cache**
```python
shutil.rmtree("cache/indicators")
```
- Supprime tous les fichiers cache IndicatorBank sur disque
- Recrée le dossier vide

#### 6. **Rechargement Application**
```python
time.sleep(2)
st.rerun()
```
- Force un rerun complet de Streamlit
- Équivalent à un `F5` dans le navigateur

---

### Utilisation

**Sidebar** > **🔧 Actions Système** > **🔄 Redémarrer**

**Cas d'usage** :
- 🔧 Après modifications de code (sans relancer Streamlit)
- 🐛 Bugs étranges liés au cache
- 💾 Saturation mémoire GPU/RAM
- 🔄 Réinitialisation complète entre 2 sessions

---

## 🛑 Fonction Arrêt

### Objectif

Ferme l'application **proprement** en nettoyant toutes les ressources.

### Fichier

[src/threadx/streamlit_app.py](src/threadx/streamlit_app.py#L736-L780) : `shutdown_app()`

### Opérations Effectuées

#### 1. **Arrêt Monitoring**
```python
monitor = get_global_monitor()
monitor.stop()
```

#### 2. **Arrêt GPU Manager**
```python
manager = get_default_manager()
manager.stop()
```
- Libère les ressources GPU
- Arrête les threads multi-GPU

#### 3. **Nettoyage Mémoire**
```python
clean_all_memory()
```
- Même nettoyage que Redémarrage
- VRAM + RAM + Caches + Ollama

#### 4. **Arrêt Streamlit**
```python
time.sleep(1.5)
st.stop()
```
- Arrête l'exécution Streamlit
- Affiche message final

---

### Utilisation

**Sidebar** > **🔧 Actions Système** > **🛑 Arrêter**

**Cas d'usage** :
- 🏁 Fin de session de trading
- 🔌 Libération complète des ressources
- 🖥️ Avant extinction PC (libère GPU/RAM)

---

## 📊 Comparaison

| Aspect | **🔄 Redémarrer** | **🛑 Arrêter** |
|--------|-------------------|----------------|
| **Monitoring** | ✅ Arrêté | ✅ Arrêté |
| **GPU Manager** | ✅ Réinitialisé | ✅ Arrêté |
| **Mémoire** | ✅ Nettoyée | ✅ Nettoyée |
| **Session State** | ✅ **Tout supprimé** | ⚠️ Clés système conservées |
| **Cache fichiers** | ✅ **Supprimé** | ❌ Conservé |
| **Application** | 🔄 **Recharge** | 🛑 **Ferme** |
| **Équivalent** | `F5` navigateur | `Ctrl+C` terminal |

---

## 🎯 Interface Sidebar

### Localisation

**Sidebar (barre latérale)** > Bas de page > **🔧 Actions Système**

### Affichage

```
🔧 Actions Système
┌─────────────┬─────────────┐
│ 🔄 Redémarrer│  🛑 Arrêter  │
└─────────────┴─────────────┘

💡 Redémarrer : réinitialise tout (cache, GPU, session)
💡 Arrêter : ferme l'application proprement
```

---

## 🧪 Workflow de Nettoyage (Détails)

### Fonction `clean_all_memory()` (partagée)

**Fichier** : [src/threadx/streamlit_app.py](src/threadx/streamlit_app.py#L661-L733)

#### Étapes

1. **VRAM GPU (CuPy)**
   ```python
   mempool = cp.get_default_memory_pool()
   mempool.free_all_blocks()
   ```
   - Libère tous les blocs VRAM
   - Message : `✅ VRAM GPU vidée: X.XX GB libérés`

2. **Cache IndicatorBank**
   ```python
   IndicatorBank._instance = None
   ```
   - Réinitialise le singleton
   - Force recréation propre

3. **Caches Streamlit**
   ```python
   st.cache_data.clear()
   st.cache_resource.clear()
   ```
   - Vide les décorateurs @cache

4. **Session State**
   ```python
   keys_to_delete = [k for k in st.session_state.keys() if not k.startswith('_')]
   for key in keys_to_delete:
       del st.session_state[key]
   ```
   - Supprime clés utilisateur
   - **Conserve** clés système (`_FormSubmitButton_*`, etc.)

5. **Garbage Collection**
   ```python
   collected = gc.collect()
   ```
   - Force libération RAM Python
   - Message : `✅ RAM système: X objets collectés`

6. **Reset Ollama**
   ```python
   reset_ollama()
   ```
   - Arrête processus Ollama
   - Redémarre proprement
   - Décharge modèles LLM de la mémoire

---

## ⚠️ Notes Importantes

### Redémarrage

**Attention** : Le redémarrage supprime **TOUTES** les données de session :
- ❌ Données chargées (OHLCV)
- ❌ Résultats de backtest
- ❌ Configurations stratégie
- ❌ Historique LLM Optimizer

**Recommandation** : Sauvegarder les résultats importants avant redémarrage.

### Arrêt

**Note** : L'arrêt **NE FERME PAS** le processus Streamlit.

Pour fermer complètement :
1. Cliquer **🛑 Arrêter**
2. Fermer l'onglet navigateur
3. `Ctrl+C` dans le terminal Streamlit

---

## 🐛 Cas d'Usage Typiques

### Situation 1 : Saturation Mémoire GPU

**Symptômes** :
- Ralentissements
- Erreur "CUDA out of memory"

**Solution** : **🔄 Redémarrer**
- Vide VRAM
- Réinitialise GPU Manager
- État propre

### Situation 2 : Cache Corrompu

**Symptômes** :
- Résultats incohérents
- Indicateurs incorrects

**Solution** : **🔄 Redémarrer**
- Supprime cache fichiers
- Réinitialise IndicatorBank
- Recalcul propre

### Situation 3 : Ollama Bloqué

**Symptômes** :
- LLM ne répond pas
- "Connection refused"

**Solution** : **🔄 Redémarrer** OU **🧹 Nettoyage Complet**
- Reset Ollama inclus
- Décharge modèles
- Redémarrage propre

### Situation 4 : Fin de Session

**Symptômes** :
- Travail terminé
- Libération ressources avant extinction PC

**Solution** : **🛑 Arrêter**
- Nettoyage complet
- Fermeture propre
- Libération GPU/RAM

---

## 🚀 Workflow Recommandé

### Développement

```
1. Modifier code
2. 🔄 Redémarrer (au lieu de relancer Streamlit)
3. Tester modifications
4. Répéter
```

**Avantage** : Pas besoin de `Ctrl+C` et relancer Streamlit

### Session Trading

```
1. Lancer app
2. Charger données
3. Backtests / Optimisations
4. 🧹 Nettoyage Complet (entre 2 runs lourds)
5. Fin session : 🛑 Arrêter
```

### Debug Mémoire

```
1. Problème détecté
2. 🔄 Redémarrer
3. Si persiste : Relancer Streamlit
4. Si persiste : Redémarrer PC (GPU driver)
```

---

## 📝 Fichiers Modifiés

1. **MODIFIÉ** : [src/threadx/streamlit_app.py](src/threadx/streamlit_app.py)
   - Lignes 736-780 : Fonction `shutdown_app()` améliorée
   - Lignes 783-846 : Fonction `restart_app()` nouvelle
   - Lignes 1154-1182 : Section "Actions Système" sidebar

---

## ✅ Résumé Exécutif

**2 nouvelles fonctions système** disponibles dans la sidebar :

| Fonction | Icône | Type | Action | Équivalent |
|----------|-------|------|--------|------------|
| **Redémarrer** | 🔄 | Secondary | Réinitialise tout + recharge | `F5` |
| **Arrêter** | 🛑 | Primary | Nettoie + ferme | `Ctrl+C` |

**Nettoyage inclus** (les 2) :
- ✅ VRAM GPU (CuPy)
- ✅ Cache IndicatorBank
- ✅ Caches Streamlit
- ✅ RAM système (GC)
- ✅ Reset Ollama
- ✅ Monitoring arrêté
- ✅ GPU Manager arrêté

**Différence** :
- **Redémarrer** : Supprime cache fichiers + session complète → Recharge
- **Arrêter** : Conserve cache fichiers → Ferme

---

**Status** : 🟢 **OPÉRATIONNEL**

*Dernière mise à jour : 2025-11-24*
