"""Internationalization (i18n) support for LM Watermark Lab."""

from .translator import get_translator, translate, set_locale, get_supported_locales
from .compliance import ComplianceManager, GDPRCompliance, CCPACompliance, PDPACompliance

__all__ = [
    "get_translator",
    "translate", 
    "set_locale",
    "get_supported_locales",
    "ComplianceManager",
    "GDPRCompliance", 
    "CCPACompliance",
    "PDPACompliance"
]