"""
URL builders for magazine scraping.

Different magazines have different URL patterns. This module provides
a pluggable architecture for building magazine-specific download URLs.

Available builders:
- TemplateURLBuilder: Simple template-based URL building (Variety, The Nation, etc.)
- TheEconomistURLBuilder: Date-conditional patterns for The Economist
- EsquireURLBuilder: Complex period-dependent patterns for Esquire
"""

from .base import TemplateURLBuilder
from .the_economist import TheEconomistURLBuilder
from .esquire import EsquireURLBuilder

__all__ = [
    'TemplateURLBuilder',
    'TheEconomistURLBuilder',
    'EsquireURLBuilder'
]
