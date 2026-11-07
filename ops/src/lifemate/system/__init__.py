"""Composable inference scaffold. No model downloads, network or state loader."""
from .pipeline import LifeMateSystem, SystemResult
from .retrieval import ExampleBank, ToolSelector
from .grounding import FactSource, GroundingRule

__all__ = ['LifeMateSystem', 'SystemResult', 'ToolSelector', 'ExampleBank',
           'FactSource', 'GroundingRule']
