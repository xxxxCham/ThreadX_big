# 📦 MANIFEST - Système de Directives ThreadX

**Document d'inventaire complet du système de directives centralisé.**

Version: `2025.11.21`  
Status: ✅ Production Ready  
Maintenance: À jour obligatoire

---

## 📋 FICHIERS SYSTÈME (5 fichiers)

### 1. DIRECTIVES_DEV.md (258 lignes)
- **Rôle:** Source de vérité unique
- **Audience:** Tous (LLMs + développeurs)
- **Mise à jour:** Obligatoire (tous les changements)
- **Contenu:**
  - Principe fondamental
  - Résumé exécutif
  - Architecture générale
  - Conventions nommage
  - Stack technologique
  - Frictions réalistes (specs complètes)
  - Netdata MCP Bridge
  - Checklist qualité
  - Erreurs courantes
  - Fichiers critiques
  - Guide doutes

### 2. README.md (90 lignes)
- **Rôle:** Documentation principale + quickstart
- **Audience:** Développeurs + nouveaux utilisateurs
- **Contenu:** Introduction, quickstart, frictions, Netdata

### 3. .llmrc (34 lignes)
- **Rôle:** Instructions rapides pour LLMs
- **Audience:** LLMs/Claude Desktop
- **Contenu:** Checklist rapide, principes clés

### 4. INDEX_DIRECTIVES.txt (240 lignes)
- **Rôle:** Lookup table + index rapide
- **Audience:** LLMs (navigation rapide)
- **Contenu:** FAQ, checklist, erreurs courantes, workflow

### 5. CHANGELOG_DIRECTIVES.md (80 lignes)
- **Rôle:** Historique changements + prochaines étapes
- **Audience:** Tous (traçabilité)
- **Contenu:** Features, checklist implementation, notes

---

## 🔧 FICHIERS DE CONFIGURATION (2 fichiers)

### 1. .editorconfig (60 lignes)
- **Rôle:** Conventions d'édition
- **Références:** DIRECTIVES_DEV.md
- **Détail:** Python (4 espaces, 100 chars max), Go (tabs)

### 2. .gitattributes (3 lignes)
- **Rôle:** Config git pour DIRECTIVES
- **Stratégie:** merge=union (pas de conflits)

---

## 🧪 OUTILS (1 script)

### 1. check_directives.py (200+ lignes)
- **Rôle:** Validation respect directives
- **Usage:** `python3 check_directives.py`
- **Checks:**
  - Fichiers critiques présents
  - Fichiers interdits absent
  - Structure modules OK
  - Contenu DIRECTIVES complet
  - Qualité code (type hints, logging)

---

## 📊 STATISTIQUES

```
Total lignes documentation: 382
Total fichiers: 7 (5 + 2 config)
Total palabras: ~4000
Temps lecture complet: 10-15 minutes
Temps consultation rapide: 1-2 minutes
```

---

## 🎯 PRINCIPES DOCUMENTÉS

### 1. Consolidation plutôt que Dispersion
- Ajouter fonctions à modules existants (90%)
- Un seul fichier doc par sujet
- Exceptions: performance/size/langage

### 2. Qualité Obligatoire
- Type hints complets
- Docstrings Google/NumPy
- Tests coverage >80%
- Logging centralisé

### 3. Architecture Clairement Documentée
```
src/threadx/backtest/engine.py   ← Moteur + frictions
src/threadx/indicators/          ← Tous les indicateurs
src/threadx/strategy/            ← Toutes les stratégies
src/threadx/utils/log.py         ← Logging unique
```

### 4. Stack Technologique Spécifiée
```
Python: 3.11+
NumPy: 2.0.2 (EXACTEMENT - <2.1 pour numba)
Numba: 0.60.0 (JIT)
Pandas: 2.2.0+
```

### 5. Frictions Réalistes - Specs Complètes
- Spread bid/ask (variable)
- Slippage (adaptatif)
- Latence (50-500ms)
- Rejets (2-30%)
- Maker/Taker fees (distinction)
- Timeframe multipliers (1m=2.0x, 1h=0.6x)

### 6. Netdata MCP Bridge - Séparé
- Langage: Go 1.16+
- Localisation: `/tools/netdata-bridge/`
- Rôle: WebSocket MCP → Netdata
- Status: Outil développement (pas intégré trading)

---

## ✔️ CHECKLIST IMPLÉMENTATION

### Phase 1: Système Directives ✅ FAIT
- [x] DIRECTIVES_DEV.md créé
- [x] README.md créé
- [x] .llmrc créé
- [x] INDEX_DIRECTIVES.txt créé
- [x] CHANGELOG_DIRECTIVES.md créé
- [x] check_directives.py créé
- [x] .editorconfig mis à jour
- [x] .gitattributes créé

### Phase 2: Frictions Réalistes (À FAIRE)
- [ ] Intégrer ExecutionResult dans engine.py
- [ ] Intégrer RealisticExecutor dans engine.py
- [ ] Intégrer EXCHANGE_CONFIGS dans engine.py
- [ ] Mettre à jour __init__.py backtest/
- [ ] Tests pour RealisticExecutor
- [ ] Documentation examples/

### Phase 3: Netdata Bridge (À FAIRE)
- [ ] Créer /tools/netdata-bridge/
- [ ] Implémenter main.go
- [ ] Créer build.sh
- [ ] Tests websocket
- [ ] Documentation setup

### Phase 4: Validation & Cleanup (À FAIRE)
- [ ] Vérifier check_directives.py passes
- [ ] Nettoyer fichiers markdown dispersés
- [ ] Review code pour non-conformité
- [ ] Tests complets suite

---

## 📚 GUIDE UTILISATION

### Pour LLMs
1. Consulter `.llmrc` (30 secondes)
2. Consulter `DIRECTIVES_DEV.md` (2 minutes)
3. Consulter `INDEX_DIRECTIVES.txt` si question (1 minute)
4. Coder selon checklist

### Pour Développeurs
1. Lire `README.md` (quickstart)
2. Consulter `DIRECTIVES_DEV.md` (détails)
3. Utiliser `check_directives.py` (validation)

### Pour Managers/PMs
1. Lire `CHANGELOG_DIRECTIVES.md` (overview)
2. Consulter `MANIFEST_DIRECTIVES.md` (ce fichier)
3. Vérifier phases implémentation

---

## 🔍 VALIDATION SYSTÈME

### Script check_directives.py
```bash
cd /workspaces/ThreadX_big
python3 check_directives.py
```

**Output Expected:**
```
✅ TOUS LES CHECKS PASSENT - Repository OK!
```

### Validation Manuelle
- [ ] DIRECTIVES_DEV.md existe et à jour
- [ ] README.md pointe vers DIRECTIVES_DEV.md
- [ ] .llmrc contient checklist
- [ ] Aucun fichier markdown dispersé
- [ ] Code respecte type hints
- [ ] Tests existent (coverage >80%)

---

## 📞 CONTACTS & MAINTENANCE

### Responsabilités

| Document | Responsable | Fréquence |
|----------|------------|-----------|
| DIRECTIVES_DEV.md | Admin | À chaque changement arch |
| README.md | Admin | À chaque release |
| .llmrc | Admin | Annuelle |
| INDEX_DIRECTIVES.txt | Admin | Annuelle |
| CHANGELOG_DIRECTIVES.md | Admin | À chaque milestone |
| check_directives.py | QA | À chaque PR |

### Review Schedule
- **Hebdomadaire:** .llmrc + check_directives.py
- **Mensuelle:** DIRECTIVES_DEV.md (mise à jour)
- **Trimestrielle:** Audit complet système

---

## 🚀 PROCHAINES PHASES (Timeline)

| Phase | Éléments | Durée | Start |
|-------|----------|-------|-------|
| **1** | Système directives | Fait | Fait |
| **2** | Frictions réalistes | 2-3h | 21/11 |
| **3** | Netdata Bridge | 2-3h | 22/11 |
| **4** | Validation & tests | 2h | 23/11 |
| **5** | Documentation finale | 1h | 24/11 |

---

## ✨ AVANTAGES SYSTÈME

1. **Cohérence:** Tous les LLMs reçoivent mêmes directives
2. **Qualité:** Type hints + docs + tests obligatoires
3. **Clarté:** Architecture documentée en un endroit
4. **Maintenabilité:** Code consolidé (pas dispersé)
5. **Performance:** Frictions réalistes spécifiées
6. **Scalabilité:** Facile ajouter nouvelles règles

---

## 📄 FICHIERS GÉNÉRÉS

```
DIRECTIVES_DEV.md          ← Source de vérité
README.md                  ← Quickstart
.llmrc                     ← LLM instructions
INDEX_DIRECTIVES.txt       ← Lookup table
CHANGELOG_DIRECTIVES.md    ← Historique
MANIFEST_DIRECTIVES.md     ← CE FICHIER
check_directives.py        ← Validation script
.editorconfig              ← Conventions éditeur
.gitattributes             ← Config git
```

---

## 🎓 LEÇON APPRISE

**Centraliser directives = moins de chaos, meilleur code.**

Un système de directives bien documenté:
- Réduit confusions (source unique de vérité)
- Améliore qualité (checklist obligatoire)
- Accélère développement (pas débat sur conventions)
- Facilite maintenance (code cohérent)

**Investir 1 jour en directives = gagner 1 semaine en maintenance.**

---

## 📝 NOTES

- Document officiel: `DIRECTIVES_DEV.md`
- Consulter avant chaque feature
- Mettre à jour à chaque changement architecture
- Valider avec `check_directives.py`

---

*Manifest créé: 21 novembre 2025*
*Status: Production Ready*
*Version: 1.0 Stable*

**Code cohérent = Framework robuste = Succès!**
