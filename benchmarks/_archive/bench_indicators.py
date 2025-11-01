python\n",
    "# Import principal\n",
    "from threadx.indicators.engine import enrich_indicators\n",
    "\n",
    "# Configuration GPU (optionnel)\n",
    "from threadx.utils.xp import gpu_available\n",
    "print(f\"GPU disponible: {gpu_available()}\")\n",
    "