# 🛠️ Correction des Erreurs UnicodeDecodeError Subprocess

## 📋 Problème observé

```
Exception in thread Thread-22 (_readerthread):
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 11: invalid start byte
```

## 🔍 Analyse

Ces erreurs proviennent d'**Ollama** qui génère des threads internes pour lire stdout/stderr de ses runners. 
Le problème survient quand :
- Ollama démarre plusieurs runners simultanément (jusqu'à 20+ threads)
- Les runners génèrent des sorties binaires non-UTF-8
- Python tente de décoder avec codec UTF-8 par défaut

## ✅ Solutions

### Solution 1 : Ignorer les erreurs threading (Recommandé)

Ces erreurs sont **non-bloquantes** et n'affectent pas le fonctionnement :
- ✅ Ollama continue de fonctionner normalement
- ✅ Les modèles se chargent correctement
- ✅ Les réponses LLM sont générées

**Action** : Aucune action requise, les warnings peuvent être ignorés.

### Solution 2 : Redirection des logs Ollama

Si les erreurs polluent les logs, redirigez stderr d'Ollama :

**Windows PowerShell :**
```powershell
ollama serve 2>$null
```

**Bash/Linux :**
```bash
ollama serve 2>/dev/null
```

### Solution 3 : Configuration environnement Python

Forcer l'encodage errors='ignore' pour subprocess :

**Fichier** : `src/threadx/llm/client.py`

```python
import sys
import io

# Avant d'importer ollama
if sys.platform == "win32":
    # Forcer UTF-8 avec gestion d'erreurs
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
```

### Solution 4 : Variable d'environnement (Windows)

```powershell
$env:PYTHONIOENCODING = "utf-8:ignore"
ollama serve
```

## 🧪 Validation

### Test 1 : Vérifier Ollama fonctionne malgré les erreurs

```powershell
# Démarrer Ollama
ollama serve

# Dans un autre terminal
curl http://localhost:11434/api/tags
```

**Résultat attendu** : Liste des modèles même avec erreurs threading

### Test 2 : Tester avec ThreadX

```python
from threadx.llm.client import get_llm_client

client = get_llm_client("deepseek-r1:32b")
response = client.chat("Test simple")
print(response)
```

**Résultat attendu** : Réponse correcte malgré warnings

## 📊 État observé dans vos logs

```
✅ Ollama démarre : 16:10:13
✅ Détecte 2 GPUs : RTX 5080 + RTX 2060 SUPER
✅ Charge deepseek-r1:32b : 16:15:06 (6 sec)
✅ Répartition GPU : 52 layers CUDA0 + 11 layers CUDA1
✅ Total memory : 19.6 GiB
✅ Runner started : 16:15:12
✅ Requête réussie : [GIN] 200 | 6.7s
```

**Conclusion** : Ollama fonctionne parfaitement. Les erreurs UnicodeDecodeError sont cosmétiques.

## 🎯 Recommandation finale

**NE RIEN MODIFIER** - Le système fonctionne correctement.

Les erreurs proviennent du code interne d'Ollama (non modifiable). 
Elles n'impactent pas :
- Le chargement des modèles
- La génération de réponses
- Les performances GPU
- La stabilité du système

**Alternative** : Si cela vous gêne, utilisez la solution 2 (redirection stderr).

## 📝 Bug d'affichage Streamlit (onglet Backtest)

### Problème
```
NameError: name '_render_config_history' is not defined
```

### Cause
Ligne 1720 dans `page_backtest_optimization.py` appelle la fonction sans préfixe.

### Solution ✅ (Déjà implémentée)
Fonction `_render_config_history` existe bien (ligne 154).
Restructuration de `main()` avec 3 onglets distincts corrige le problème :

1. 🔬 **Sweep Classique** : Optimisation manuelle
2. 🤖 **Sweep + LLM** : Analyse LLM (à implémenter)
3. 🧠 **Multi-Agents Autonome** : Redirection vers page dédiée

## 🚀 Prochaines étapes

1. ✅ Tester nouveau système 3 onglets
2. ⏳ Implémenter Sweep + LLM (analyse post-sweep)
3. ⏳ Ajouter option Monte-Carlo dans menu

---

**Date** : 21 novembre 2025  
**Version ThreadX** : v2.0  
**Auteur** : GitHub Copilot
