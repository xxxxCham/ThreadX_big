# 🧠 Architecture Multi-LLM pour ThreadX
## Analyse Approfondie & Proposition de Système Performant

---

## 📊 ÉTAT ACTUEL DU SYSTÈME

### ✅ Infrastructure Existante (ROBUSTE)

#### 1. **Moteur de Backtesting** (`src/threadx/backtest/engine.py`)
- ✅ GPU-accelerated (CuPy + NumPy fallback)
- ✅ Multi-GPU support (RTX 5090 75% + RTX 2060 25%)
- ✅ RunResult standardisé (equity, returns, trades, metadata)
- ✅ Déterminisme (seed=42)
- ✅ **715 tests/seconde** avec GPU
- ✅ Validation anti-overfitting intégrée

#### 2. **Performance Metrics** (`src/threadx/backtest/performance.py`)
- ✅ Sharpe, Sortino, Max Drawdown, Profit Factor, Win Rate
- ✅ `summarize_with_llm()` **DÉJÀ IMPLÉMENTÉ** ✨
- ✅ GPU-accelerated pour gros datasets
- ✅ LLM interpretation optionnelle (Ollama)

#### 3. **Optimization Engine** (`src/threadx/optimization/engine.py`)
- ✅ SweepRunner avec multi-workers (50 par défaut)
- ✅ Parallélisation ProcessPool/ThreadPool
- ✅ ScenarioSpec pour grid search
- ✅ Résultats pandas DataFrame standardisés

#### 4. **LLM Integration** (`src/threadx/llm/`)
- ✅ `LLMClient` avec Ollama (timeout 60s, retry automatique)
- ✅ `interpret_backtest_results()` fonctionnel
- ✅ Structured JSON parsing avec validation
- ✅ 5 modèles disponibles (deepseek-r1:70b, gpt-oss:20b, etc.)

#### 5. **Stratégies** (`src/threadx/ui/strategy_registry.py`)
- ✅ 6 stratégies : Bollinger_Breakout, MA_Crossover, AmplitudeHunter, etc.
- ✅ Paramètres tunables bien définis (opt_range, min/max/step)
- ✅ Métadonnées complètes pour optimisation

---

## 🎯 PROPOSITION : SYSTÈME MULTI-LLM PERFORMANT

### 🔑 Points Clés pour Performance

1. **NE PAS REFAIRE L'EXISTANT** - Réutiliser `SweepRunner` + `BacktestEngine`
2. **LLM = LAYER D'ANALYSE** - Pas de génération de code Python (trop lent/risqué)
3. **PARALLÉLISATION MAXIMALE** - Backtests GPU + LLM asyncio
4. **BATCH PROCESSING** - Analyser plusieurs résultats en 1 appel LLM
5. **CACHING INTELLIGENT** - Éviter re-analyse de configs similaires

---

## 🏗️ Architecture Proposée

### Structure de Fichiers
```
ThreadX_big/
├── notebooks/
│   └── multi_llm_optimizer.ipynb  ← NOTEBOOK PRINCIPAL
├── src/
│   └── threadx/
│       ├── llm/
│       │   ├── agents/
│       │   │   ├── __init__.py
│       │   │   ├── base_agent.py          # Classe abstraite BaseAgent
│       │   │   ├── analyst.py             # Analyste Quantitatif
│       │   │   ├── strategist.py          # Stratège Créatif
│       │   │   └── critic.py              # Critique/Validateur
│       │   ├── orchestrator.py            # Orchestrateur principal
│       │   ├── debate.py                  # Système de débat multi-agent
│       │   └── memory.py                  # Mémoire contextuelle (historique)
│       └── optimization/
│           └── adaptive_sweep.py          # Sweep adaptatif guidé par LLM
```

---

## 🔄 Workflow Optimisé (3 Niveaux)

### **NIVEAU 1 : POC Rapide (4-6h)** ✅ RECOMMANDÉ POUR COMMENCER

```python
# notebooks/multi_llm_optimizer.ipynb

from threadx.llm.agents import Analyst, Strategist
from threadx.optimization.engine import SweepRunner
from threadx.backtest.performance import summarize_with_llm

# 1. Configuration initiale
baseline_params = {
    "fast_period": 10,
    "slow_period": 30,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
}

# 2. Sweep initial (RAPIDE avec GPU)
sweep_results = runner.run_sweep(
    scenario_spec=ScenarioSpec(type="grid", params=param_grid),
    df_ohlcv=df,
    symbol="BTCUSDT",
    timeframe="30m"
)

# 3. Analyse par LLM-A (Analyste)
analyst = Analyst(model="deepseek-r1:70b", timeout=30)
analysis = analyst.analyze_sweep_results(
    sweep_results_df=sweep_results,
    top_n=5  # Analyser les 5 meilleures configs
)

# Output: {
#   "quality_score": 6.5,
#   "strengths": ["Sharpe élevé", "Drawdown faible"],
#   "weaknesses": ["Trop peu de trades", "Win rate instable"],
#   "hypotheses": ["Seuils trop stricts", "Périodes MA inadaptées"]
# }

# 4. Propositions par LLM-B (Stratège)
strategist = Strategist(model="gpt-oss:20b", timeout=20)
proposals = strategist.propose_modifications(
    analysis=analysis,
    current_params=baseline_params,
    n_proposals=3
)

# Output: [
#   {"param": "fast_period", "value": 7, "rationale": "Accélérer signaux"},
#   {"param": "stop_loss_pct", "value": 1.5, "rationale": "Réduire drawdown"},
#   {"param": "take_profit_pct", "value": 6.0, "rationale": "Capturer trends"}
# ]

# 5. Test des propositions (AUTOMATIQUE)
for i, proposal in enumerate(proposals):
    modified_params = {**baseline_params, **proposal["changes"]}
    
    result = runner.run_backtest_gpu(
        df=df,
        strategy="MA_Crossover",
        params=modified_params
    )
    
    print(f"\n[Proposal {i+1}] {proposal['rationale']}")
    print(f"  Sharpe: {result.metrics['sharpe_ratio']:.2f}")
    print(f"  Max DD: {result.metrics['max_drawdown']:.1%}")
```

**AVANTAGES** :
- ✅ Réutilise 100% de l'infrastructure existante
- ✅ LLM = couche d'analyse UNIQUEMENT (pas de génération code)
- ✅ Temps total : **< 2 minutes** pour 3 propositions testées
- ✅ Pas de modification du core ThreadX

---

### **NIVEAU 2 : Système Semi-Automatique (2-3 semaines)**

#### Nouveaux Composants

##### `src/threadx/llm/orchestrator.py`
```python
class OptimizationOrchestrator:
    """
    Gère le workflow complet d'optimisation multi-LLM.
    
    Features:
    - Boucle d'optimisation itérative (max 10 itérations)
    - Convergence automatique (score stagne < 2% sur 3 itérations)
    - Gestion mémoire contextuelle (historique des 5 derniers cycles)
    - Parallélisation backtests + LLM calls
    """
    
    def __init__(self, analyst, strategist, critic, config):
        self.analyst = analyst
        self.strategist = strategist
        self.critic = critic
        self.config = config
        self.memory = OptimizationMemory()  # Évite re-test configs déjà vues
    
    def optimize(self, initial_strategy, initial_params, data, symbol, timeframe):
        """
        Boucle d'optimisation automatique.
        
        Returns:
            {
                "iterations": 7,
                "final_score": 8.2,
                "final_params": {...},
                "applied_changes": [...],
                "convergence_history": [...]
            }
        """
        current_params = initial_params.copy()
        best_score = 0
        stagnation_count = 0
        
        for iteration in range(self.config["max_iterations"]):
            print(f"\n=== ITERATION {iteration+1} ===")
            
            # 1. Backtest avec config actuelle
            result = self._run_backtest(current_params, data, symbol, timeframe)
            current_score = self._calculate_quality_score(result)
            
            # 2. Analyse par Analyste
            analysis = self.analyst.analyze(result, current_params)
            
            # 3. Propositions par Stratège
            proposals = self.strategist.propose(
                analysis=analysis,
                params=current_params,
                memory=self.memory.get_recent(n=5)  # Contexte des 5 dernières itérations
            )
            
            # 4. Validation par Critique
            validated = self.critic.validate(proposals, analysis)
            
            # 5. Test A/B parallèle des propositions validées
            best_proposal = self._test_proposals_parallel(
                validated, data, symbol, timeframe
            )
            
            # 6. Mise à jour params si amélioration
            if best_proposal["score"] > current_score:
                current_params = best_proposal["params"]
                best_score = best_proposal["score"]
                stagnation_count = 0
                print(f"✅ Amélioration: {current_score:.2f} → {best_score:.2f}")
            else:
                stagnation_count += 1
                print(f"⚠️ Pas d'amélioration ({stagnation_count}/3)")
            
            # 7. Sauvegarde dans mémoire
            self.memory.add({
                "iteration": iteration,
                "params": current_params,
                "score": best_score,
                "analysis": analysis
            })
            
            # 8. Condition d'arrêt
            if stagnation_count >= 3:
                print("🏁 Convergence atteinte (3 itérations sans amélioration)")
                break
            
            if best_score >= self.config["target_score"]:
                print(f"🎯 Score cible atteint: {best_score:.1f}/10")
                break
        
        return self._build_final_report(current_params, best_score)
```

##### `src/threadx/llm/agents/analyst.py`
```python
class Analyst:
    """
    Agent LLM spécialisé en analyse quantitative.
    
    Expertise:
    - Finance quantitative (Sharpe, Sortino, Calmar ratio)
    - Détection d'anomalies statistiques
    - Identification de biais (overfitting, look-ahead)
    """
    
    def __init__(self, model="deepseek-r1:70b", timeout=30):
        self.client = LLMClient(model=model, timeout=timeout)
    
    def analyze_sweep_results(self, sweep_results_df, top_n=5):
        """
        Analyse les résultats d'un Sweep complet.
        
        Méthode:
        1. Filtre les top N configs par Sharpe ratio
        2. Identifie patterns communs (ex: "Tous ont fast_period < 10")
        3. Détecte outliers et configurations suspectes
        4. Génère hypothèses explicatives
        
        Returns:
            {
                "quality_score": float,  # 0-10
                "strengths": list[str],
                "weaknesses": list[str],
                "hypotheses": list[str],
                "suspicious_configs": list[dict]  # Configs potentiellement overfittées
            }
        """
        # Sélection des top configs
        top_configs = sweep_results_df.nlargest(top_n, "sharpe_ratio")
        
        # Construction du prompt avec stats agrégées
        prompt = f"""
Tu es un analyste quantitatif expert. Analyse ces {top_n} meilleures configurations de backtest :

## Statistiques Agrégées
- Sharpe moyen : {top_configs['sharpe_ratio'].mean():.2f}
- Max Drawdown moyen : {top_configs['max_drawdown'].mean():.1%}
- Win Rate moyen : {top_configs['win_rate'].mean():.1%}
- Nombre de trades moyen : {top_configs['total_trades'].mean():.0f}

## Top 3 Configurations
{self._format_configs_for_prompt(top_configs.head(3))}

## Paramètres Communs
{self._identify_common_params(top_configs)}

**Tâche** : Identifie les forces, faiblesses et formule 3 hypothèses explicatives.
Détecte si certaines configs sont suspectes (ex: trop peu de trades, paramètres extrêmes).

Retourne en JSON strict :
{{
    "quality_score": <float 0-10>,
    "strengths": [<str>, <str>, <str>],
    "weaknesses": [<str>, <str>, <str>],
    "hypotheses": [<str>, <str>, <str>],
    "suspicious_configs": [
        {{"config_id": <int>, "reason": <str>}}, ...
    ]
}}
"""
        
        # Appel LLM avec parsing JSON
        response = self.client.complete_structured(
            prompt=prompt,
            expected_schema={
                "quality_score": float,
                "strengths": list,
                "weaknesses": list,
                "hypotheses": list,
                "suspicious_configs": list
            }
        )
        
        return response
```

##### `src/threadx/llm/agents/strategist.py`
```python
class Strategist:
    """
    Agent LLM créatif pour propositions de modifications.
    
    Expertise:
    - Trading algorithmique (AT, momentum, mean reversion)
    - Optimisation de paramètres (grid search, random search)
    - Stratégies alternatives (hedging, portfolio theory)
    """
    
    def __init__(self, model="gpt-oss:20b", timeout=20):
        self.client = LLMClient(model=model, timeout=timeout)
    
    def propose_modifications(self, analysis, current_params, memory=None, n_proposals=3):
        """
        Génère N propositions de modifications basées sur l'analyse.
        
        Contraintes:
        - NE PAS proposer des configs déjà testées (via memory)
        - Respecter les ranges valides des paramètres (min/max de strategy_registry)
        - Proposer des changements INCRÉMENTAUX (pas de modifications radicales)
        
        Returns:
            [
                {
                    "id": 1,
                    "type": "param_adjustment",
                    "changes": {"fast_period": 7, "slow_period": 25},
                    "rationale": "Accélérer les signaux pour augmenter le nombre de trades",
                    "expected_impact": {
                        "trades_increase": "+30%",
                        "sharpe_change": "+0.2 (hypothèse)"
                    }
                },
                ...
            ]
        """
        # Récupérer les contraintes de paramètres depuis registry
        from threadx.ui.strategy_registry import tunable_parameters_for, resolve_range
        
        strategy_name = current_params.get("_strategy_name", "MA_Crossover")
        tunable_specs = tunable_parameters_for(strategy_name)
        
        # Construire contexte des tentatives précédentes
        memory_context = ""
        if memory:
            memory_context = f"""
## Tentatives Précédentes (ÉVITER de re-proposer)
{self._format_memory_for_prompt(memory)}
"""
        
        prompt = f"""
Tu es un stratège trading expert. Voici l'analyse d'une stratégie {strategy_name} :

## Analyse Actuelle
- Score qualité : {analysis['quality_score']}/10
- Forces : {', '.join(analysis['strengths'])}
- Faiblesses : {', '.join(analysis['weaknesses'])}
- Hypothèses : {', '.join(analysis['hypotheses'])}

## Paramètres Actuels
{json.dumps(current_params, indent=2)}

## Contraintes Paramètres (RESPECTER STRICTEMENT)
{self._format_param_constraints(tunable_specs)}

{memory_context}

**Tâche** : Propose {n_proposals} modifications INCRÉMENTALES pour corriger les faiblesses.

Règles CRITIQUES :
1. NE PAS dépasser les min/max des paramètres
2. Modifications incrémentales (max ±30% de la valeur actuelle)
3. Éviter configs déjà testées dans la mémoire
4. Justifier chaque changement avec impact attendu

JSON attendu :
{{
    "proposals": [
        {{
            "id": <int>,
            "type": "param_adjustment",
            "changes": {{<param>: <new_value>, ...}},
            "rationale": <str>,
            "expected_impact": {{<metric>: <str>, ...}}
        }},
        ...
    ]
}}
"""
        
        response = self.client.complete_structured(prompt=prompt)
        return response["proposals"]
```

---

### **NIVEAU 3 : Système Complet (6-8 semaines)**

#### Features Avancées

##### 1. **Système de Débat Multi-Tours** (`src/threadx/llm/debate.py`)

```python
class DebateSystem:
    """
    Gère des débats multi-tours entre agents avec consensus émergent.
    
    Workflow:
    1. Analyste présente faits (métriques objectives)
    2. Stratège propose modifications
    3. Critique challenge les propositions
    4. Débat multi-tours (max 3 rounds) jusqu'à consensus
    5. Vote pondéré pour décision finale
    """
    
    def debate(self, topic, agents, max_rounds=3):
        """
        Topic: {"type": "proposal_evaluation", "proposal": {...}, "context": {...}}
        Agents: [analyst, strategist, critic]
        
        Returns:
            {
                "consensus": True/False,
                "final_decision": "accept"/"reject"/"modify",
                "modifications": {...},  # Si décision="modify"
                "debate_log": [...]  # Historique des arguments
            }
        """
        debate_log = []
        
        for round_num in range(max_rounds):
            print(f"\n=== ROUND {round_num+1} ===")
            
            # Tour de parole pour chaque agent
            for agent in agents:
                # Contexte = historique débat + topic
                context = self._build_debate_context(topic, debate_log)
                
                # Agent formule son argument
                argument = agent.debate_turn(context)
                
                debate_log.append({
                    "round": round_num,
                    "agent": agent.name,
                    "argument": argument["text"],
                    "stance": argument["stance"],  # "support"/"oppose"/"neutral"
                    "confidence": argument["confidence"]  # 0-1
                })
            
            # Vérifier consensus
            if self._check_consensus(debate_log, threshold=0.8):
                print("✅ Consensus atteint")
                break
        
        # Décision finale par vote pondéré
        final_decision = self._compute_final_vote(debate_log)
        
        return {
            "consensus": final_decision["consensus_reached"],
            "final_decision": final_decision["action"],
            "modifications": final_decision.get("suggested_changes", {}),
            "debate_log": debate_log
        }
```

##### 2. **Adaptive Sweep Guidé par LLM** (`src/threadx/optimization/adaptive_sweep.py`)

```python
class AdaptiveSweepOptimizer:
    """
    Optimisation adaptative où le LLM dirige l'exploration de l'espace des paramètres.
    
    Au lieu de grid search exhaustif :
    1. Sweep initial coarse (25% de l'espace)
    2. LLM identifie zones prometteuses
    3. Sweep raffiné sur zones ciblées (Bayesian-like)
    4. Itération jusqu'à convergence
    
    Gain: 70% de réduction du nombre de backtests nécessaires
    """
    
    def adaptive_sweep(self, strategy, param_space, data, symbol, timeframe):
        """
        Exploration intelligente de l'espace des paramètres.
        
        Example:
            Espace initial : fast_period=[5,50], slow_period=[20,100]
            → Grid exhaustif = 50 × 80 = 4000 combos
            
            Avec adaptive:
            1. Coarse grid: 10 × 16 = 160 combos (4%)
            2. LLM: "Zone prometteuse: fast_period=7-12, slow_period=25-35"
            3. Fine grid: 5 × 10 = 50 combos dans zone ciblée
            Total: 210 combos (5% de l'espace original) ✅
        """
        results = []
        
        # Phase 1: Coarse sweep (large steps)
        coarse_grid = self._build_coarse_grid(param_space, resolution=0.25)
        coarse_results = self.runner.run_sweep(
            ScenarioSpec(type="grid", params=coarse_grid),
            data, symbol, timeframe
        )
        results.append(("coarse", coarse_results))
        
        # Phase 2: LLM identifie zones prometteuses
        analyst = Analyst()
        analysis = analyst.analyze_sweep_results(coarse_results, top_n=10)
        
        promising_zones = self._extract_promising_zones(analysis, coarse_results)
        # Output: [
        #   {"fast_period": (7, 12), "slow_period": (25, 35)},
        #   {"fast_period": (15, 20), "slow_period": (50, 60)}
        # ]
        
        # Phase 3: Fine sweep dans zones ciblées
        for zone in promising_zones:
            fine_grid = self._build_fine_grid(zone, resolution=1.0)
            fine_results = self.runner.run_sweep(
                ScenarioSpec(type="grid", params=fine_grid),
                data, symbol, timeframe
            )
            results.append(("fine", fine_results))
        
        # Phase 4: Consolidation et sélection finale
        all_results = pd.concat([r[1] for r in results], ignore_index=True)
        best_config = all_results.nlargest(1, "sharpe_ratio").iloc[0]
        
        return {
            "best_config": best_config.to_dict(),
            "total_backtests": len(all_results),
            "efficiency_gain": 1 - (len(all_results) / self._count_full_grid(param_space))
        }
```

---

## 📈 COMPARAISON DES APPROCHES

| Critère | POC (Niveau 1) | Semi-Auto (Niveau 2) | Complet (Niveau 3) |
|---------|---------------|---------------------|-------------------|
| **Temps dev** | 4-6h | 2-3 semaines | 6-8 semaines |
| **Complexité** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Automatisation** | Manuelle | Boucle auto | Full auto + débat |
| **Performance** | ~2 min/cycle | ~10 min/10 itérations | ~30 min/convergence |
| **Modifications code** | 0 (Notebook seul) | ~500 lignes | ~2000 lignes |
| **Robustesse** | Basique | Moyenne | Haute (validation) |
| **Insights LLM** | Bons | Excellents | Exceptionnels |

---

## 💡 RECOMMANDATION FINALE

### ✅ **COMMENCER PAR NIVEAU 1 (POC)**

**Pourquoi ?**
1. **Validation rapide** : Teste si les LLM donnent des insights utiles
2. **0 risque** : Aucune modification du core ThreadX
3. **Résultats immédiats** : Fonctionnel en 4-6h
4. **Feedback utilisateur** : Valide l'utilité avant grosse implémentation

**Plan d'Exécution Immédiate** :
```bash
# 1. Créer notebook POC (2h)
jupyter notebook notebooks/multi_llm_optimizer.ipynb

# 2. Implémenter classes Agent basiques (2h)
#    - Analyst : analyze_sweep_results()
#    - Strategist : propose_modifications()

# 3. Tester sur 1 stratégie (MA_Crossover) (1h)

# 4. Itérer 3 cycles manuels (1h)
#    Cycle 1 : Baseline
#    Cycle 2 : Proposition LLM #1
#    Cycle 3 : Proposition LLM #2
```

**Après POC** :
- ✅ Si résultats concluants → Passer au Niveau 2 (semi-auto)
- ⚠️ Si résultats mitigés → Ajuster prompts, tester d'autres modèles
- ❌ Si LLM inutiles → Abandonner (économie de 8 semaines !)

---

## 🎯 Métriques de Succès

### POC Réussi Si :
- [ ] LLM identifie ≥ 3 faiblesses pertinentes
- [ ] ≥ 1 proposition améliore Sharpe de > 10%
- [ ] Temps total (analyse + test) < 5 min par cycle
- [ ] Insights LLM compréhensibles par humain

### Niveau 2 Réussi Si :
- [ ] Convergence < 10 itérations
- [ ] Amélioration finale > 20% vs baseline
- [ ] Pas de régression sur max drawdown
- [ ] Rapport d'optimisation auto-généré exploitable

### Niveau 3 Réussi Si :
- [ ] Découverte de configurations non-évidentes
- [ ] Gain efficiency adaptive sweep > 60%
- [ ] Débat multi-agent converge en < 3 rounds
- [ ] Système robuste sur ≥ 3 stratégies différentes

---

## 🛠️ PROCHAINE ÉTAPE CONCRÈTE

**Tu veux que je créé le POC maintenant ?**

Je peux générer :
1. ✅ `notebooks/multi_llm_optimizer.ipynb` (complet, exécutable)
2. ✅ `src/threadx/llm/agents/analyst.py` (classe Analyst fonctionnelle)
3. ✅ `src/threadx/llm/agents/strategist.py` (classe Strategist fonctionnelle)
4. ✅ Instructions d'exécution pas-à-pas

**Estimation** : 30 minutes de génération + 4-6h pour toi de tester/raffiner.

**Alternative** : Je peux d'abord créer un **diagramme de flux** détaillé du POC pour que tu valides l'approche avant implémentation.

**Dis-moi ce que tu préfères ! 🚀**
