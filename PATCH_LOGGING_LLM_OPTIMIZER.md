# PATCH : Ajout Logging dans page_llm_optimizer.py

## Problème
`page_llm_optimizer.py` a **0 logs** backend alors qu'il gère le workflow critique d'optimisation LLM avec :
- 12 timing operations (time.time())
- 12 blocs try/except
- Workflow complexe sur 5 étapes (Sweep → Analyst → Strategist → Tests → Rapport)

**Impact** : Impossible de diagnostiquer problèmes, lenteurs, ou erreurs depuis les logs système.

---

## Solution

### PATCH 1 : Ajouter import logger (ligne 39)

```python
# AVANT (ligne 39)
from threadx.llm.ollama_manager import prepare_for_llm_run

# APRÈS
from threadx.llm.ollama_manager import prepare_for_llm_run
from threadx.utils.log import get_logger

logger = get_logger(__name__)
```

---

### PATCH 2 : Ajouter logging dans run_multi_llm_optimization()

#### A. Début de fonction (après ligne 736)

```python
# APRÈS ligne 736
st.session_state["llm_run_start_time"] = time.time()

# AJOUTER
logger.info(
    f"[Multi-LLM] Démarrage optimisation - "
    f"strategy:{strategy_name}, analyst:{analyst_model}, strategist:{strategist_model}, "
    f"n_proposals:{n_proposals}, gpu:{use_gpu}, multigpu:{use_multigpu}"
)
```

#### B. Après ÉTAPE 1 - Sweep (après ligne 764)

```python
# APRÈS ligne 764
st.session_state["llm_sweep_duration"] = time.time() - sweep_start

# AJOUTER
logger.info(
    f"[Multi-LLM] Étape 1/5 SWEEP terminé - "
    f"{len(sweep_results)} configs testées en {time.time() - sweep_start:.2f}s"
)
```

#### C. Avant ÉTAPE 2 - Analyst (après ligne 806, avant appel Analyst)

```python
# APRÈS ligne 806 (après sélection modèle)
current_analyst_model = model_router.get_model_for_task(TaskType.INITIALIZATION)

# AJOUTER
logger.info(f"[Multi-LLM] Étape 2/5 ANALYST démarré - modèle:{current_analyst_model}, top_n:{top_n_analysis}")
analyst_start = time.time()
```

#### D. Après appel Analyst (chercher après `analyst.run()`)

**Trouver la ligne avec** :
```python
analysis = analyst.run(...)
st.session_state["llm_analyst_duration"] = time.time() - analyst_start
```

**Ajouter après** :
```python
logger.info(
    f"[Multi-LLM] Étape 2/5 ANALYST terminé - "
    f"{len(analysis.get('insights', []))} insights en {time.time() - analyst_start:.2f}s"
)
```

#### E. Avant ÉTAPE 3 - Strategist

**Trouver** :
```python
status_text.markdown("### 🧠 Étape 3/5: Génération Propositions (Strategist)...")
```

**Ajouter avant l'appel strategist.run()** :
```python
logger.info(f"[Multi-LLM] Étape 3/5 STRATEGIST démarré - modèle:{current_strategist_model}, n_proposals:{n_proposals}")
strategist_start = time.time()
```

#### F. Après Strategist

**Après** :
```python
st.session_state["llm_strategist_duration"] = time.time() - strategist_start
```

**Ajouter** :
```python
logger.info(
    f"[Multi-LLM] Étape 3/5 STRATEGIST terminé - "
    f"{len(proposals_data.get('proposals', []))} propositions en {time.time() - strategist_start:.2f}s"
)
```

#### G. Avant ÉTAPE 4 - Test Propositions

**Trouver** :
```python
status_text.markdown("### 🧪 Étape 4/5: Test des Propositions...")
```

**Ajouter avant l'appel test_proposals()** :
```python
logger.info(f"[Multi-LLM] Étape 4/5 TESTS démarré - {len(valid_proposals)} propositions à tester")
test_start = time.time()
```

#### H. Après Tests

**Après** :
```python
test_results = test_proposals(...)
```

**Ajouter** :
```python
test_duration = time.time() - test_start
successful_tests = sum(1 for r in test_results if r.get('sharpe_ratio'))
logger.info(
    f"[Multi-LLM] Étape 4/5 TESTS terminé - "
    f"{successful_tests}/{len(valid_proposals)} propositions valides en {test_duration:.2f}s"
)
```

#### I. Fin de fonction (avant return/fin try)

**À la fin du try block, ajouter** :
```python
total_duration = time.time() - st.session_state["llm_run_start_time"]
logger.info(
    f"[Multi-LLM] Optimisation TERMINÉE - "
    f"Durée totale:{total_duration:.2f}s, "
    f"Sweep:{st.session_state.get('llm_sweep_duration', 0):.1f}s, "
    f"Analyst:{st.session_state.get('llm_analyst_duration', 0):.1f}s, "
    f"Strategist:{st.session_state.get('llm_strategist_duration', 0):.1f}s"
)
```

#### J. Gestion erreurs (dans except)

**Dans le bloc except existant, ajouter au début** :
```python
except Exception as e:
    logger.error(f"[Multi-LLM] ERREUR lors de l'optimisation: {type(e).__name__}: {str(e)}", exc_info=True)
    # ... reste du code except existant
```

---

### PATCH 3 : Ajouter logging dans execute_sweep()

**Dans la fonction execute_sweep() (ligne 1293)** :

```python
def execute_sweep(...):
    """..."""

    # AJOUTER au début
    logger.info(f"[Sweep] Démarrage sweep - strategy:{strategy_name}, gpu:{use_gpu}, workers:{max_workers}")

    # ... code existant...

    # AJOUTER après création SweepRunner
    logger.debug(f"[Sweep] {total_combos} combinaisons générées")

    # AJOUTER après runner.run()
    logger.info(
        f"[Sweep] Terminé - {len(results)} résultats, "
        f"meilleur sharpe:{max(r.get('sharpe_ratio', 0) for r in results):.3f}"
    )
```

---

### PATCH 4 : Ajouter logging dans test_proposals()

**Dans la fonction test_proposals() (ligne 1636)** :

```python
def test_proposals(...):
    """..."""

    # AJOUTER au début
    logger.info(f"[Test Proposals] Démarrage - {len(proposals)} propositions à tester")

    # AJOUTER dans la boucle for (après chaque test)
    for i, proposal in enumerate(proposals):
        # ... code test ...
        logger.debug(f"[Test Proposals] {i+1}/{len(proposals)} - {proposal.get('name', 'unnamed')}: sharpe={result.get('sharpe_ratio', 0):.3f}")

    # AJOUTER avant return
    logger.info(f"[Test Proposals] Terminé - {len(results)} résultats")
    return results
```

---

## Impact

**Avant** :
- 0 logs backend
- Impossible de diagnostiquer problèmes
- Pas de profiling des durées

**Après** :
- ~15 logs INFO structurés (workflow complet traçable)
- ~10 logs DEBUG (détails par proposition)
- Durées d'exécution de chaque étape
- Erreurs avec stack traces complets

**Overhead** : Négligeable (<0.1% runtime)

---

## Validation

```bash
# Tester avec logs DEBUG activés
export THREADX_LOG_LEVEL=DEBUG  # Linux/Mac
set THREADX_LOG_LEVEL=DEBUG     # Windows

streamlit run src/threadx/streamlit_app.py
# → Aller sur page LLM Optimizer
# → Lancer une optimisation
# → Vérifier les logs dans le terminal
```

**Logs attendus** :
```
[timestamp] threadx.ui.page_llm_optimizer - INFO - [Multi-LLM] Démarrage optimisation - strategy:bollinger_dual, analyst:qwen2.5:32b, ...
[timestamp] threadx.ui.page_llm_optimizer - INFO - [Sweep] Démarrage sweep - strategy:bollinger_dual, gpu:True, workers:30
[timestamp] threadx.ui.page_llm_optimizer - INFO - [Sweep] Terminé - 100 résultats, meilleur sharpe:2.145
[timestamp] threadx.ui.page_llm_optimizer - INFO - [Multi-LLM] Étape 1/5 SWEEP terminé - 100 configs testées en 45.23s
[timestamp] threadx.ui.page_llm_optimizer - INFO - [Multi-LLM] Étape 2/5 ANALYST démarré - modèle:qwen2.5:32b, top_n:10
...
```

---

## Fichiers à modifier

- `src/threadx/ui/page_llm_optimizer.py` (patches 1-4)

**Estimation** : ~30 lignes ajoutées, 0 lignes modifiées (seulement ajouts)
