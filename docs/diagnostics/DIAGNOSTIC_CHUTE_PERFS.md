# Diagnostic Chute de Performances (10.2 → 5.2 tests/sec)

## 🔍 Problème Identifié

**Observation**: Après corrections P0.1 + P0.5, vitesse a **CHUTÉ** de 10.2 → 5.2 tests/sec

### Logs Révélateurs

```
[2025-11-13 02:28:16] threadx.indicators.bollinger - INFO - 🔥 GPU Manager: 2 GPU(s) détectés (× 16 fois!)
[2025-11-13 02:28:16] threadx.indicators.bank - INFO - 🏦 IndicatorBank initialisé - Cache: indicators_cache (× 16 fois!)
[2025-11-13 02:28:16] threadx.indicators.xatr - INFO - 🎯 ATR initialisé - GPU: True, Multi-GPU: 2 (× 16 fois!)
```

**16 instances** = 16 workers qui chacun **recréent GPU Manager + IndicatorBank**

---

## 🎯 Cause Racine

### Flux Actuel (INCORRECT)

```python
# optimization/engine.py:431
executor.submit(
    self._evaluate_single_combination,  # ← Exécuté dans worker fork
    combo,
    computed_indicators,  # ← Dict passé MAIS...
    real_data,
    symbol,
    timeframe,
    strategy_name
)

# _evaluate_single_combination (ligne 746):
strategy = self._cached_strategy_instances[cache_key]  # ← OK: Cache marche
equity, stats = strategy.backtest(
    df=real_data,
    params=strategy_params,
    precomputed_indicators=computed_indicators  # ← PROBLÈME: Format/clés incorrects
)

# strategy/bb_atr.py:657
df_with_indicators, atr_array = self._ensure_indicators(
    df, strategy_params,
    precomputed_indicators=precomputed_indicators  # ← Reçoit dict
)

# strategy/bb_atr.py:509 (_ensure_indicators)
if precomputed_indicators:
    bb_key = json.dumps({"period": params.bb_period, "std": params.bb_std}, sort_keys=True)
    atr_key = json.dumps({"period": params.atr_period}, sort_keys=True)

    try:
        bb_result = precomputed_indicators["bollinger"][bb_key]  # ← FAIL: Clé absente !
        atr_array = precomputed_indicators["atr"][atr_key]       # ← FAIL: Clé absente !
    except KeyError:
        # Fallback: RECRÉE GPU Manager + IndicatorBank ❌
        bb_result = ensure_indicator(...)  # ← LENT: 70ms × 16 workers = overhead 1120ms !
```

### Problème: **Mismatch de Clés**

**Dans `_compute_batch_indicators` (ligne 687)**:
```python
for params_key, result in batch_results.items():
    computed[indicator_type][params_key] = result  # ← params_key est HASH interne IndicatorBank
```

**Dans `_ensure_indicators` (ligne 509)**:
```python
bb_key = json.dumps({"period": ..., "std": ...}, sort_keys=True)  # ← Clé différente !
```

**Résultat**: `bb_key` ne match jamais avec `params_key` → KeyError → Fallback recalcul → Overhead 16x

---

## 💡 Solution Immédiate (P0.2)

### Option A: Normaliser les Clés

**Modifier `_compute_batch_indicators` pour utiliser même format que stratégie**:

```python
def _compute_batch_indicators(...):
    computed = {}

    for indicator_type, params_list in unique_indicators.items():
        computed[indicator_type] = {}

        batch_results = self.indicator_bank.batch_ensure(...)

        # ✅ NORMALISER: Utiliser même format que stratégie
        for params in params_list:
            # Générer MÊME clé que _ensure_indicators
            if indicator_type == "bollinger":
                key = json.dumps({
                    "period": params.get("period", 20),
                    "std": params.get("std", 2.0)
                }, sort_keys=True)
            elif indicator_type == "atr":
                key = json.dumps({
                    "period": params.get("period", 14)
                }, sort_keys=True)

            # Récupérer depuis batch_results (mapping interne IndicatorBank)
            internal_key = self._params_to_key(params)
            computed[indicator_type][key] = batch_results[internal_key]

    return computed
```

### Option B: Passer IndicatorBank Singleton aux Workers

**Architecture préférable** (évite duplication):

```python
# optimization/engine.py
def __init__(self, indicator_bank, ...):
    self.indicator_bank = indicator_bank  # ✅ Instance unique partagée
    # Pas de recréation dans workers

# _evaluate_single_combination (modifier):
strategy = self._cached_strategy_instances[cache_key]
# ✅ FORCER réutilisation IndicatorBank du SweepRunner
strategy.indicator_bank = self.indicator_bank  # Injecter singleton

equity, stats = strategy.backtest(...)  # ← Utilisera singleton au lieu de recréer
```

### Option C: Pré-Calculer TOUS Indicateurs AVANT Fork

**Architecture optimale** (ce que P0.2 doit faire):

```python
# AVANT parallélisation (dans thread principal)
unique_indicators = self._extract_unique_indicators(combinations)
computed_indicators = self._compute_batch_indicators(unique_indicators, ...)

# ✅ Convertir en format NUMPY pur (pas d'objets Python)
precomputed_numpy = {
    "bollinger": {
        json.dumps({"period": 20, "std": 2.0}): (upper_np, middle_np, lower_np),
        ...
    },
    "atr": {
        json.dumps({"period": 14}): atr_np,
        ...
    }
}

# PUIS parallélisation (workers lisent precomputed_numpy, read-only)
with ThreadPoolExecutor(...) as executor:
    futures = [
        executor.submit(
            self._evaluate_with_precomputed,  # ← Nouvelle fonction
            combo,
            precomputed_numpy,  # ← Dict read-only, pas de lock
            real_data
        )
        for combo in combinations
    ]
```

---

## 📊 Impact Attendu de Chaque Option

### Option A (Normalisation Clés)
**Gain**: 5.2 → **10.2 tests/sec** (retour baseline)
**Effort**: 30 min
**Risque**: Faible

### Option B (Singleton IndicatorBank)
**Gain**: 5.2 → **12-15 tests/sec** (+20%)
**Effort**: 1h
**Risque**: Moyen (thread-safety IndicatorBank)

### Option C (Pré-Calcul Complet)
**Gain**: 5.2 → **25-30 tests/sec** (+500%)
**Effort**: 4h
**Risque**: Faible (architecture propre)

---

## 🚀 Recommandation

**Implémenter Option A MAINTENANT** (quick fix):
- Restaure performances baseline
- Permet de tester P0.1 + P0.5 correctement
- 30 minutes max

**Puis Option C** (P0.2 complet):
- Gain massif (6x)
- Architecture propre
- Élimine tous les overhead

---

## ✅ Checklist Quick Fix (Option A)

1. [ ] Modifier `_compute_batch_indicators()` → Normaliser clés
2. [ ] Ajouter `_normalize_indicator_key()` helper
3. [ ] Tester sur mini sweep (24 combos)
4. [ ] Vérifier logs: 0 recréation GPU Manager
5. [ ] Valider vitesse >= 10 tests/sec

**ETA**: 30 minutes
**Gain attendu**: 5.2 → 10.2 tests/sec (2x speedup)

---

**Rapport généré par**: Claude Code (Sonnet 4.5)
**Date**: 2025-11-13 02:30 UTC
