# ThreadX Benchmarks - Phase C

Ce répertoire contient les outils de benchmark, KPI gates, et rapports de performance pour ThreadX.

## Structure

- **`results/`** : Résultats CSV des benchmarks (`bench_cpu_gpu_<TIMESTAMP>.csv`)
- **`reports/`** : Rapports Markdown détaillés (`REPORT_<TIMESTAMP>.md`)
- **`baselines/`** : Références pour tests non-régressifs
- **`utils.py`** : Utilitaires communs pour benchmarking
- **`run_indicators.py`** : Scripts de benchmarks existants
- **`run_backtests.py`** : Scripts de benchmarks existants

## Exécution des benchmarks

Pour lancer un benchmark complet CPU vs GPU :

```powershell
python -m tools.benchmarks_cpu_gpu --indicators bollinger,atr --sizes 10000,100000,1000000 --repeats 5
```

Arguments disponibles :
- `--indicators` : Liste d'indicateurs à tester (séparés par virgules)
- `--sizes` : Tailles de données à tester (séparées par virgules)
- `--repeats` : Nombre de répétitions par test
- `--seed` : Seed pour reproductibilité (défaut: 191159)
- `--export` : Format d'export (csv, md, all)

## KPI Gates

Les benchmarks vérifient les KPI suivants :

1. **KPI_SPEEDUP_GPU**: Accélération GPU ≥ 3× vs CPU
2. **KPI_CACHE_HIT**: Taux de cache hit ≥ 80%
3. **KPI_DETERMINISM**: Hash identique sur 3 exécutions avec seed fixe
4. **KPI_PARETO**: Algorithme Pareto non-régressif (±5% vs baseline)

Ces KPI sont également vérifiés par les tests automatisés (`tests/test_kpi_gates.py`).

## Rapports

Les rapports générés contiennent :
- Badges des KPI (OK/KO)
- Tableaux de résultats pour chaque taille testée
- Graphiques des performances (optionnel)
- Diagnostics détaillés en cas d'échec des KPI
- Informations sur l'environnement d'exécution

## Tests automatisés

Pour exécuter uniquement les tests KPI :

```powershell
python -m pytest tests/test_kpi_gates.py -v
```

## Formats des fichiers

### CSV

```csv
date,indicator,N,device,repeats,mean_ms,std_ms,gain_vs_cpu,gpu_kernel_ratio
2025-10-05 12:30,bollinger,10000,cpu,5,15.2,0.8,nan,nan
2025-10-05 12:30,bollinger,10000,gpu,5,4.3,0.3,3.5,0.85
```

### Markdown

Le rapport Markdown contient les sections suivantes :
1. Résumé exécutable (badges)
2. Méthodologie (chrono CPU/GPU, warmup, repeats, seeds)
3. Tableaux synthèse & liens vers CSV
4. Résultats KPI (OK/KO, seuils, marges)
5. Diagnostics en cas de KO
6. Appendix (config, versions, device info)