# Dictionnaire du Code ThreadX

Ce document recense l'ensemble des fichiers et dossiers du projet ThreadX, avec une explication de leur utilité.

## 📂 Racine du Projet (`src/threadx`)

C'est le point d'entrée et la configuration globale.

- **`__init__.py`** : Marqueur de package Python.
- **`config.py`** : Configuration globale du projet (chemins, constantes).
- **`data_access.py`** : Gestion de l'accès aux données (chargement parquet/json).
- **`LANCER_STREAMLIT.bat`** : Script de lancement rapide de l'interface utilisateur.
- **`LANCER_STREAMLIT_DEBUG.bat`** : Script de lancement en mode debug.
- **`requirements.txt`** : Liste des dépendances Python.
- **`streamlit_app.py`** : Point d'entrée principal de l'application Streamlit (Dashboard).
- **`threadx.code-workspace`** : Configuration de l'espace de travail VS Code.

---

## 📂 Dossiers

### 📁 `backtest`
Moteur de backtesting pour simuler les stratégies sur des données historiques.
- **`engine.py`** : Cœur du moteur de backtest (boucle principale).
- **`performance.py`** : Calcul des métriques de performance (ROI, Drawdown, Sharpe...).
- **`validation.py`** : Validation des résultats de backtest.

### 📁 `configuration`
Gestion avancée de la configuration.
- **`errors.py`** : Définitions des erreurs de configuration.
- **`loaders.py`** : Chargeurs pour les fichiers de config (YAML/JSON).
- **`settings.py`** : Modèles Pydantic pour les paramètres.

### 📁 `data`
Gestion des données de marché.
- **`arbo_data_folder.txt`** : Documentation de la structure des dossiers de données.
- **`normalize.py`** : Normalisation des données brutes.
- **`schemas.py`** : Schémas de validation des données (Pydantic/Pandas).
- **`crypto_data_json/`** : (Dossier) Données brutes en JSON.
- **`crypto_data_parquet/`** : (Dossier) Données optimisées en format Parquet.
- **`indicateurs_data_parquet/`** : (Dossier) Cache des indicateurs pré-calculés.

### 📁 `dataset`
Outils pour la gestion des datasets.
- **`validate.py`** : Scripts de validation de l'intégrité des datasets.

### 📁 `gpu`
Accélération matérielle et gestion des ressources.
- **`device_manager.py`** : Gestion des périphériques (CPU/GPU/MPS).
- **`multi_gpu.py`** : Support pour l'exécution sur plusieurs GPU.
- **`profile_persistence.py`** : Sauvegarde des profils d'exécution GPU.
- **`vector_checks.py`** : Vérifications vectorielles pour le code GPU.

### 📁 `indicators`
Bibliothèque d'indicateurs techniques.
- **`bank.py`** : "Indicator Bank" - Gestion centralisée et cache des indicateurs.
- **`bollinger.py`** : Bandes de Bollinger.
- **`gpu_integration.py`** : Pont pour le calcul d'indicateurs sur GPU.
- **`xatr.py`** : Indicateur ATR étendu.

### 📁 `llm`
Intégration des modèles de langage (LLM).
- **`client.py`** : Client pour communiquer avec les modèles (Ollama, etc.).
- **`interpreters.py`** : Interprétation des réponses du LLM.
- **`prompts.py`** : Gestion des templates de prompts.
- **`agents/`** :
    - **`analyst.py`** : Agent spécialisé dans l'analyse de marché.
    - **`strategist.py`** : Agent pour la conception de stratégies.
    - **`base_agent.py`** : Classe de base pour les agents.

### 📁 `optimization`
Moteur d'optimisation des paramètres de stratégie.
- **`engine.py`** : Moteur principal d'optimisation.
- **`multi_sweep.py`** : Gestion des balayages de paramètres multiples.
- **`parallel_sweep.py`** : Exécution parallèle des optimisations.
- **`pruning.py`** : Logique d'élagage (arrêt prématuré des mauvaises configs).
- **`reporting.py`** : Génération de rapports d'optimisation.
- **`scenarios.py`** : Définition de scénarios de test.
- **`ui.py`** : Composants UI spécifiques à l'optimisation.
- **`presets/`** : Préréglages d'optimisation.
- **`templates/`** : Templates de stratégies pour l'optimisation.

### 📁 `profiling`
Analyse de performance du code.
- **`performance_analysis.py`** : Outils de profiling.

### 📁 `strategy`
Implémentation des stratégies de trading.
- **`amplitude_hunter.py`** : Stratégie basée sur l'amplitude.
- **`bb_atr.py`** : Stratégie combinant Bollinger et ATR.
- **`bollinger_dual.py`** : Stratégie double Bollinger.
- **`ma_crossover.py`** : Stratégie de croisement de moyennes mobiles.
- **`model.py`** : Modèle de base pour les stratégies.
- **`_archive/`** : Anciennes stratégies ou exemples.

### 📁 `testing`
Tests unitaires et d'intégration.
- **`mocks.py`** : Objets simulés pour les tests.

### 📁 `ui`
Interface Utilisateur (Streamlit).
- **`backtest_bridge.py`** : Pont entre l'UI et le moteur de backtest.
- **`fast_sweep.py`** : Interface pour les sweeps rapides.
- **`page_backtest_optimization.py`** : Page principale de backtest et optimisation.
- **`page_config_strategy.py`** : Page de configuration des stratégies.
- **`page_llm_optimizer.py`** : Page pour l'optimisation assistée par LLM.
- **`strategy_registry.py`** : Registre des stratégies disponibles dans l'UI.
- **`styles.py`** : Définitions CSS et styles de l'application.
- **`system_monitor.py`** : Moniteur de ressources système.
- **`components/`** :
    - **`charts.py`** : Composants graphiques.
    - **`config.py`** : Composants de configuration.
    - **`metrics.py`** : Affichage des métriques.

### 📁 `utils`
Utilitaires divers.
- **`cache.py`** : Gestion du cache.
- **`common_imports.py`** : Imports communs pour simplifier les dépendances.
- **`log.py`** : Configuration du logging.
- **`timing.py`** : Décorateurs et outils de mesure du temps.
- **`xp.py`** : Abstraction pour Numpy/CuPy (CPU/GPU).

### 📁 `visualization`
Visualisation des données.
- **`backtest_charts.py`** : Graphiques spécifiques aux backtests.
