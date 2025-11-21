# 🦙 Guide Installation Ollama - Windows

## 📍 Emplacement des Modèles

Par défaut, Ollama stocke ses modèles dans :
```
C:\Users\<VotreNom>\.ollama\models\
```

Si vous avez changé l'emplacement, vérifiez la variable d'environnement `OLLAMA_MODELS`.

---

## 🔧 Installation Ollama

### 1. Téléchargement
- Téléchargez depuis : **https://ollama.com/download**
- Exécutez `OllamaSetup.exe`
- L'installation crée automatiquement un service Windows

### 2. Vérification Installation
Ouvrez PowerShell ou CMD :
```bash
ollama --version
```

Devrait afficher la version installée (ex: `ollama version is 0.3.12`)

---

## 📥 Téléchargement des Modèles Requis

Pour le système Multi-Agents de ThreadX, vous devez télécharger **3 modèles** :

```bash
# 1. Analyst Agent (DeepSeek R1 70B - ~40 GB)
ollama pull deepseek-r1:70b

# 2. Strategist Agent (GPT-OSS 20B - ~12 GB)
ollama pull gpt-oss:20b

# 3. Critic Agent (DeepSeek R1 32B - ~20 GB)
ollama pull deepseek-r1:32b
```

⚠️ **ATTENTION** : Ces modèles sont volumineux (total ~72 GB). Assurez-vous d'avoir suffisamment d'espace disque !

---

## ▶️ Démarrage du Service Ollama

### Option A : Service Windows (Automatique)
Par défaut, Ollama s'installe comme service Windows et démarre automatiquement au boot.

**Vérifier le service :**
1. Ouvrez **Gestionnaire des tâches** (Ctrl+Shift+Esc)
2. Onglet **Services**
3. Cherchez **OllamaService** → doit être "En cours d'exécution"

**Démarrer manuellement :**
```powershell
# PowerShell en Admin
Start-Service OllamaService
```

### Option B : Lancement Manuel
Si le service n'est pas configuré :
```bash
ollama serve
```

Laissez cette fenêtre ouverte pendant que vous utilisez ThreadX.

---

## ✅ Test de Connexion

```bash
# Lister les modèles téléchargés
ollama list

# Tester un modèle (devrait répondre)
ollama run deepseek-r1:70b "Bonjour, réponds en 5 mots"
```

Si tout fonctionne, vous devriez voir une réponse du modèle.

---

## 🔍 Diagnostic Erreur "WinError 10061"

Cette erreur signifie que **Ollama n'est pas démarré** ou **le port 11434 est bloqué**.

### Étape 1 : Vérifier si Ollama tourne
```bash
netstat -an | findstr "11434"
```

Devrait afficher quelque chose comme :
```
TCP    0.0.0.0:11434          0.0.0.0:0              LISTENING
```

Si rien n'apparaît → Ollama n'est PAS démarré.

### Étape 2 : Démarrer Ollama
```bash
ollama serve
```

Ou redémarrez le service :
```powershell
Restart-Service OllamaService
```

### Étape 3 : Tester l'API
```bash
curl http://localhost:11434/api/tags
```

Devrait retourner la liste des modèles installés (JSON).

---

## 🚀 Utilisation avec ThreadX

Une fois Ollama fonctionnel :

1. **Vérifiez les modèles** :
   ```bash
   ollama list
   ```
   Vous devez voir :
   - `deepseek-r1:70b`
   - `gpt-oss:20b`
   - `deepseek-r1:32b`

2. **Lancez ThreadX** :
   ```bash
   streamlit run src/threadx/streamlit_app.py
   ```

3. **Naviguez vers "🤖 Multi-Agents Autonome"** dans la sidebar

4. **Configurez et Démarrez** l'orchestrator

---

## 📊 Monitoring Ressources

Les modèles LLM consomment beaucoup de RAM/VRAM :

| Modèle             | VRAM GPU (idéal) | RAM CPU (fallback) |
|--------------------|------------------|--------------------|
| deepseek-r1:70b    | ~40 GB           | ~70 GB             |
| gpt-oss:20b        | ~12 GB           | ~24 GB             |
| deepseek-r1:32b    | ~20 GB           | ~40 GB             |

**Sans GPU puissant** → Ollama utilise la RAM système (plus lent, mais fonctionne).

---

## 🛠️ Dépannage

### Problème : "Model not found"
```bash
ollama pull <model_name>
```

### Problème : Ollama lent / freeze
- Fermez les applications gourmandes (navigateurs, IDE, etc.)
- Utilisez des quantizations plus légères : `deepseek-r1:32b` au lieu de `70b`

### Problème : Port 11434 occupé
```bash
# Windows : trouver le processus
netstat -ano | findstr "11434"

# Tuer le processus (remplacez <PID> par l'ID affiché)
taskkill /PID <PID> /F
```

---

## 📚 Ressources Officielles

- **Documentation Ollama** : https://github.com/ollama/ollama/blob/main/docs/windows.md
- **Liste Modèles** : https://ollama.com/library
- **API Reference** : https://github.com/ollama/ollama/blob/main/docs/api.md

---

## 🎯 Résumé Étapes Essentielles

1. **Installer** : Téléchargez depuis https://ollama.com/download
2. **Modèles** : `ollama pull deepseek-r1:70b gpt-oss:20b deepseek-r1:32b`
3. **Service** : Vérifiez que le service Windows "OllamaService" tourne
4. **Test** : `ollama list` → devrait afficher vos 3 modèles
5. **ThreadX** : Lancez l'UI et naviguez vers "🤖 Multi-Agents Autonome"

**En cas d'erreur WinError 10061** → `ollama serve` dans une fenêtre PowerShell séparée.

---

✅ **Vous êtes prêt !** Une fois Ollama démarré, le système Multi-Agents ThreadX peut maintenant analyser vos résultats d'optimisation et proposer des stratégies améliorées.
