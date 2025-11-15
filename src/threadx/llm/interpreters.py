"""
ThreadX LLM Response Interpreters
==================================

Parsers et validateurs pour structurer les réponses LLM en objets Python.

Features:
- Validation des clés requises
- Coercition de types (str → list, etc.)
- Fallback values par défaut
- Logging des erreurs de parsing
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_backtest_interpretation(response: dict[str, Any]) -> dict[str, Any]:
    """
    Parse et valide une réponse d'interprétation de backtest.

    Args:
        response: Dict brut depuis LLM

    Returns:
        Dict validé avec structure garantie:
        {
            "interpretation": str,
            "strengths": list[str],
            "weaknesses": list[str],
            "recommendations": list[str],
            "risk_level": str,
            "suitability": str
        }
    """
    # Valeurs par défaut
    defaults = {
        "interpretation": "Analyse non disponible",
        "strengths": [],
        "weaknesses": [],
        "recommendations": [],
        "risk_level": "UNKNOWN",
        "suitability": "Non analysé",
    }

    # Fusionner avec la réponse
    result = defaults.copy()

    for key in defaults:
        if key in response:
            value = response[key]

            # Validation des types
            if key in ["strengths", "weaknesses", "recommendations"]:
                # Doit être une liste
                if isinstance(value, str):
                    # Convertir string → liste
                    result[key] = [value] if value else []
                elif isinstance(value, list):
                    result[key] = [str(item) for item in value if item]
                else:
                    logger.warning(f"Invalid type for {key}: {type(value)}, using default")
                    result[key] = defaults[key]

            elif key == "risk_level":
                # Valider le niveau de risque
                valid_levels = ["LOW", "MODERATE", "HIGH", "UNKNOWN"]
                if isinstance(value, str):
                    upper_val = value.upper().strip()
                    if upper_val in valid_levels:
                        result[key] = upper_val
                    else:
                        logger.warning(f"Invalid risk_level '{value}', defaulting to UNKNOWN")
                        result[key] = "UNKNOWN"

            else:
                # Autres champs (str)
                result[key] = str(value) if value else defaults[key]

    # Log validation
    logger.debug(
        f"Parsed interpretation: {len(result['strengths'])} strengths, "
        f"{len(result['weaknesses'])} weaknesses, "
        f"{len(result['recommendations'])} recommendations"
    )

    return result


def parse_param_recommendation(response: dict[str, Any]) -> dict[str, Any]:
    """
    Parse une réponse de recommandation de paramètres.

    Args:
        response: Dict brut depuis LLM

    Returns:
        Dict validé avec structure:
        {
            "recommended_params": dict,
            "reasoning": dict,
            "confidence": float,
            "alternatives": list[dict]
        }
    """
    defaults = {
        "recommended_params": {},
        "reasoning": {},
        "confidence": 0.5,
        "alternatives": [],
    }

    result = defaults.copy()

    # recommended_params
    if "recommended_params" in response and isinstance(response["recommended_params"], dict):
        result["recommended_params"] = response["recommended_params"]

    # reasoning
    if "reasoning" in response and isinstance(response["reasoning"], dict):
        result["reasoning"] = response["reasoning"]

    # confidence
    if "confidence" in response:
        try:
            conf = float(response["confidence"])
            result["confidence"] = max(0.0, min(1.0, conf))  # Clamp [0, 1]
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence value: {response['confidence']}")
            result["confidence"] = 0.5

    # alternatives
    if "alternatives" in response and isinstance(response["alternatives"], list):
        result["alternatives"] = response["alternatives"]

    return result


def parse_anomaly_detection(response: dict[str, Any]) -> dict[str, Any]:
    """
    Parse une réponse de détection d'anomalies.

    Args:
        response: Dict brut depuis LLM

    Returns:
        Dict validé avec structure:
        {
            "anomalies_detected": bool,
            "suspicious_results": list[dict],
            "overall_quality": str,
            "warnings": list[str]
        }
    """
    defaults = {
        "anomalies_detected": False,
        "suspicious_results": [],
        "overall_quality": "UNKNOWN",
        "warnings": [],
    }

    result = defaults.copy()

    # anomalies_detected
    if "anomalies_detected" in response:
        result["anomalies_detected"] = bool(response["anomalies_detected"])

    # suspicious_results
    if "suspicious_results" in response and isinstance(response["suspicious_results"], list):
        result["suspicious_results"] = response["suspicious_results"]

    # overall_quality
    valid_qualities = ["EXCELLENT", "GOOD", "SUSPICIOUS", "POOR", "UNKNOWN"]
    if "overall_quality" in response:
        quality = str(response["overall_quality"]).upper().strip()
        if quality in valid_qualities:
            result["overall_quality"] = quality

    # warnings
    if "warnings" in response:
        if isinstance(response["warnings"], list):
            result["warnings"] = [str(w) for w in response["warnings"] if w]
        elif isinstance(response["warnings"], str):
            result["warnings"] = [response["warnings"]]

    return result


def parse_strategy_debug(response: dict[str, Any]) -> dict[str, Any]:
    """
    Parse une réponse de debugging de stratégie.

    Args:
        response: Dict brut depuis LLM

    Returns:
        Dict validé avec structure:
        {
            "diagnosis": str,
            "root_cause": str,
            "fix": str,
            "preventive_measures": list[str],
            "confidence": float
        }
    """
    defaults = {
        "diagnosis": "Non diagnostiqué",
        "root_cause": "Non identifié",
        "fix": "Aucune solution proposée",
        "preventive_measures": [],
        "confidence": 0.0,
    }

    result = defaults.copy()

    # Champs texte
    for field in ["diagnosis", "root_cause", "fix"]:
        if field in response:
            result[field] = str(response[field]) if response[field] else defaults[field]

    # preventive_measures
    if "preventive_measures" in response:
        if isinstance(response["preventive_measures"], list):
            result["preventive_measures"] = [str(m) for m in response["preventive_measures"] if m]
        elif isinstance(response["preventive_measures"], str):
            result["preventive_measures"] = [response["preventive_measures"]]

    # confidence
    if "confidence" in response:
        try:
            conf = float(response["confidence"])
            result["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            result["confidence"] = 0.0

    return result
