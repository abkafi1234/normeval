# Import the main class so it's available at the top level
from .evaluator import NormalizationEvaluator

# Define the version in one place
__version__ = "0.1.1"

# Explicitly define what is exported
__all__ = ["NormalizationEvaluator"]
