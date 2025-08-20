"""Watermarking methods implementation."""

from watermark_lab.methods.base import BaseWatermark
from watermark_lab.methods.kirchenbauer import KirchenbauerWatermark

__all__ = [
    "BaseWatermark",
    "KirchenbauerWatermark",
    "list_available_methods"
]

def list_available_methods():
    """List all available watermarking methods."""
    methods = [
        {
            "name": "kirchenbauer",
            "class": KirchenbauerWatermark,
            "description": "Green-red list watermark for language models",
            "paper_url": "https://arxiv.org/abs/2301.10226",
            "watermark_type": "statistical"
        }
    ]
    return methods