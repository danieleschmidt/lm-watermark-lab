"""
WatermarkDetector: Unified detection interface for all watermarking schemes.
"""

from typing import List, Optional, Union
from .green_red import GreenRedWatermark
from .semantic import SemanticWatermark
from .statistical import StatisticalWatermark


class WatermarkDetector:
    """
    Unified detector for GreenRedWatermark, SemanticWatermark, and StatisticalWatermark.

    Parameters
    ----------
    watermarker : GreenRedWatermark | SemanticWatermark | StatisticalWatermark
        The watermarker instance (carries secret key and parameters for detection).
    """

    def __init__(
        self,
        watermarker: Union[GreenRedWatermark, SemanticWatermark, StatisticalWatermark],
    ):
        self.watermarker = watermarker

    def detect(self, text: str) -> dict:
        """
        Detect watermark in *text* by tokenizing and running the appropriate detector.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        dict
            Detection result including `is_watermarked` and method-specific stats.
        """
        tokens = self._tokenize(text)
        result = self.watermarker.detect(tokens)
        result["method"] = type(self.watermarker).__name__
        result["num_tokens"] = len(tokens)
        return result

    def detect_tokens(self, tokens: List[str]) -> dict:
        """Detect watermark from pre-tokenized token list."""
        result = self.watermarker.detect(tokens)
        result["method"] = type(self.watermarker).__name__
        result["num_tokens"] = len(tokens)
        return result

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenizer with punctuation splitting."""
        import re
        # Split on whitespace, keep punctuation attached for now
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens


def detect_all(
    text: str,
    green_red: Optional[GreenRedWatermark] = None,
    semantic: Optional[SemanticWatermark] = None,
    statistical: Optional[StatisticalWatermark] = None,
) -> dict:
    """
    Convenience function: run all provided detectors on *text* and return results.

    Parameters
    ----------
    text : str
    green_red : GreenRedWatermark, optional
    semantic : SemanticWatermark, optional
    statistical : StatisticalWatermark, optional

    Returns
    -------
    dict
        Keys: "green_red", "semantic", "statistical" (whichever were provided).
    """
    results = {}
    if green_red is not None:
        results["green_red"] = WatermarkDetector(green_red).detect(text)
    if semantic is not None:
        results["semantic"] = WatermarkDetector(semantic).detect(text)
    if statistical is not None:
        results["statistical"] = WatermarkDetector(statistical).detect(text)
    return results
