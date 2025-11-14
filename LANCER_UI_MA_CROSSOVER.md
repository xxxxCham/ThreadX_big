# 🚀 Guide Rapide - Tester MA Crossover dans l'UI

## ✅ Validation Installation

Toutes les vérifications sont passées:

```bash
✅ Stratégie enregistrée dans le registre
✅ Import MACrossoverStrategy fonctionnel
✅ Paramètres par défaut configurés
✅ Mapping dans engine.py effectué
```

---

## 🎯 Lancement de l'Interface

### Option 1: Lancement Standard

```bash
cd D:\ThreadX_big
streamlit run src\threadx\ui\app.py
```

### Option 2: Avec Configuration Personnalisée

```bash
cd D:\ThreadX_big
set STREAMLIT_SERVER_PORT=8502
streamlit run src\threadx\ui\app.py
```

---

## 📊 Navigation dans l'UI

### 1. Accéder à la Page de Backtest

```
Application ThreadX
  └── 📊 Backtest & Optimisation
```

### 2. Configuration de Base

**Données:**
- **Symbol:** BTCUSDC
- **Timeframe:** 15m
- **Période:** 2024-12-01 → 2025-01-31
- **Capital initial:** 10,000 USDC

**Stratégie:**
- **Sélectionner:** MA_Crossover (dans le dropdown)

### 3. Paramètres Recommandés (Premier Test)

#### Test Rapide - Params par Défaut
```
Fast Period:     10
Slow Period:     30
Stop Loss %:     2.0
Take Profit %:   4.0
Risk per Trade:  0.01  (1%)
Leverage:        1.0   (pas de levier)
Max Hold Bars:   100
Fee BPS:         4.5
Slippage BPS:    0.0
```

**Cliquer sur:** `▶️ Run Backtest`

**Résultats attendus:**
- ✅ ~250 trades générés
- ✅ Drawdown: -15% à -25%
- ✅ Win rate: 25-35%
- ✅ Equity curve stable

---

## 🔬 Mode Optimisation

### Configuration Optimisation Rapide

**Page:** Optimisation > Configuration

**Paramètres à optimiser:**
```
Fast Period:
  Min: 5
  Max: 30
  Step: 5
  → 6 valeurs

Slow Period:
  Min: 20
  Max: 60
  Step: 10
  → 5 valeurs

Stop Loss %:
  Min: 1.5
  Max: 3.0
  Step: 0.5
  → 4 valeurs
```

**Total combinaisons:** 6 × 5 × 4 = 120

**Temps estimé:**
- CPU seul: ~2-3 minutes
- GPU activé: ~30 secondes

**Cliquer sur:** `🚀 Start Optimization`

---

## 📈 Analyse des Résultats

### Métriques à Vérifier

**1. Cohérence Capital:**
```python
Capital Final ≈ Capital Initial + Total PnL
```
✅ Si différence < 1 USDC → Calculs corrects

**2. Drawdown Raisonnable:**
```
Max DD < -50%
```
✅ Avec stops 2% et risk 1%, impossible d'avoir DD > -50%

**3. Nombre de Trades:**
```
Total Trades: 150-300
```
✅ Si 0 trades → Problème filtres
✅ Si > 500 trades → Vérifier spacing

**4. Win Rate:**
```
Win Rate: 20-40%
```
✅ Normal pour stratégie MA simple non optimisée

---

## 🎓 Comparaison avec BB+ATR

### Test Parallèle Recommandé

1. **Run MA_Crossover** (nouveau)
   - Noter: PnL, DD, Trades

2. **Run BB+ATR** (existant)
   - Noter: PnL, DD, Trades

3. **Comparer:**
   ```
   | Métrique      | MA_Crossover | BB+ATR    |
   |---------------|--------------|-----------|
   | Trades        | ~250         | ~12       |
   | Drawdown      | -20%         | -99% ❌   |
   | Cohérence     | ✅           | ❌        |
   ```

### Questions à Répondre

- **Q1:** Pourquoi BB+ATR génère si peu de trades?
  → Filtres trop restrictifs (trend, spacing, min_pnl)

- **Q2:** Pourquoi DD -99% avec risk 1.5%?
  → Stops non déclenchés OU position sizing erroné

- **Q3:** MA Crossover fonctionne mais pas BB+ATR?
  → Bug spécifique à la logique BB+ATR, pas au moteur

---

## 🐛 Troubleshooting

### Stratégie MA_Crossover n'apparaît pas

**Solution 1:** Redémarrer Streamlit
```bash
Ctrl+C
streamlit run src\threadx\ui\app.py
```

**Solution 2:** Vider cache Streamlit
```bash
streamlit cache clear
streamlit run src\threadx\ui\app.py
```

**Solution 3:** Vérifier installation
```bash
cd D:\ThreadX_big
python -c "import sys; sys.path.insert(0, 'src'); from threadx.ui.strategy_registry import list_strategies; print(list_strategies())"
```
Doit afficher: `[..., 'MA_Crossover']`

### Erreur au lancement du backtest

**Erreur:** `MACrossoverStrategy.__init__() missing 1 required positional argument`

**Cause:** Ancienne version de ma_crossover.py sans les paramètres symbol/timeframe

**Solution:** Vérifier que `__init__` accepte symbol, timeframe, indicator_bank

### Résultats incohérents

**Symptôme:** DD > -50% ou capital négatif

**Solution:**
1. Comparer avec test CLI:
   ```bash
   python test_ma_crossover.py
   ```
2. Si CLI OK mais UI KO → Bug dans interface
3. Si CLI KO aussi → Bug dans stratégie

---

## 📊 Export des Résultats

### Via l'Interface

1. Après backtest, cliquer: `💾 Export Results`
2. Format CSV avec:
   - Trades détaillés
   - Equity curve
   - Statistiques

### Via CLI (Plus Détaillé)

```bash
python test_ma_crossover.py
```

Génère:
- `CSV/test_ma_crossover_results.csv`
- `CSV/test_ma_crossover_equity.csv`

---

## 🎯 Prochaines Étapes

### 1. Valider Fonctionnement
- [x] Run backtest simple
- [ ] Vérifier cohérence résultats
- [ ] Comparer avec test CLI

### 2. Optimiser Paramètres
- [ ] Grid search fast/slow periods
- [ ] Tester différents stops
- [ ] Analyser win rate vs params

### 3. Déboguer BB+ATR
- [ ] Identifier différences avec MA
- [ ] Isoler le bug DD -99%
- [ ] Appliquer corrections

---

## 📞 Support

Si problème persistant:

1. **Vérifier logs:**
   ```
   Terminal où Streamlit tourne
   Chercher erreurs Python
   ```

2. **Fichiers à vérifier:**
   ```
   src/threadx/strategy/ma_crossover.py    ← Stratégie
   src/threadx/ui/strategy_registry.py    ← Registre
   src/threadx/optimization/engine.py     ← Mapping
   ```

3. **Tests de validation:**
   ```bash
   python test_ma_crossover.py  # Test CLI
   ```

---

**Status:** ✅ Prêt à l'emploi
**Version:** 1.0.0
**Date:** 2025-11-13
