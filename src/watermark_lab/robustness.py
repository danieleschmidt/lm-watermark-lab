"""
RobustnessEvaluator: Test how much text modification breaks each watermark.

Simulates paraphrasing attacks by randomly swapping words with synonyms at
various noise levels (swap rate = fraction of tokens replaced).
"""

import random
from typing import List, Optional, Union, Dict, Any

from .green_red import GreenRedWatermark
from .semantic import SemanticWatermark
from .statistical import StatisticalWatermark
from .detector import WatermarkDetector


# Default synonym pool used for paraphrasing simulation
PARAPHRASE_SYNONYMS: Dict[str, List[str]] = {
    # Common substitution pairs for noise injection
    "important": ["crucial", "vital", "key", "critical"],
    "large": ["big", "huge", "enormous", "substantial"],
    "small": ["tiny", "little", "minor", "minimal"],
    "good": ["fine", "solid", "decent", "positive"],
    "great": ["excellent", "superb", "outstanding", "wonderful"],
    "make": ["produce", "build", "construct", "craft"],
    "show": ["display", "present", "exhibit", "reveal"],
    "use": ["utilize", "apply", "employ", "leverage"],
    "find": ["locate", "discover", "identify", "uncover"],
    "get": ["acquire", "obtain", "receive", "gain"],
    "said": ["stated", "mentioned", "noted", "remarked"],
    "think": ["believe", "consider", "suppose", "assume"],
    "work": ["function", "operate", "perform", "labor"],
    "problem": ["issue", "challenge", "difficulty", "obstacle"],
    "result": ["outcome", "consequence", "effect", "product"],
    "method": ["technique", "approach", "strategy", "procedure"],
    "system": ["framework", "structure", "setup", "arrangement"],
    "time": ["period", "duration", "moment", "interval"],
    "also": ["additionally", "furthermore", "moreover", "besides"],
    "however": ["nevertheless", "yet", "still", "but"],
    "therefore": ["thus", "hence", "consequently", "accordingly"],
    "often": ["frequently", "regularly", "repeatedly", "commonly"],
    "different": ["distinct", "varied", "diverse", "unlike"],
    "new": ["fresh", "recent", "novel", "modern"],
    "high": ["elevated", "raised", "lofty", "tall"],
    "fast": ["quick", "swift", "rapid", "speedy"],
    "strong": ["robust", "powerful", "sturdy", "firm"],
    "clear": ["obvious", "apparent", "evident", "plain"],
    "simple": ["easy", "basic", "straightforward", "plain"],
    "common": ["typical", "usual", "ordinary", "standard"],
    "real": ["actual", "true", "genuine", "authentic"],
    "long": ["extended", "lengthy", "prolonged", "extensive"],
    "place": ["location", "site", "spot", "position"],
    "number": ["count", "quantity", "amount", "total"],
    "point": ["aspect", "element", "factor", "detail"],
    "part": ["section", "segment", "portion", "piece"],
    "level": ["degree", "extent", "stage", "tier"],
    "area": ["region", "zone", "domain", "field"],
    "world": ["realm", "sphere", "universe", "domain"],
    "group": ["set", "collection", "cluster", "ensemble"],
    "type": ["kind", "sort", "variety", "category"],
    "case": ["instance", "example", "scenario", "situation"],
    "example": ["instance", "sample", "illustration", "case"],
    "study": ["research", "analysis", "examination", "investigation"],
    "change": ["shift", "modification", "alteration", "transformation"],
    "begin": ["start", "commence", "initiate", "launch"],
    "help": ["support", "aid", "assist", "facilitate"],
    "need": ["require", "demand", "necessitate", "call for"],
    "feel": ["experience", "sense", "perceive", "notice"],
    "seem": ["appear", "look", "come across"],
}


class RobustnessEvaluator:
    """
    Evaluate how robust watermarks are under paraphrasing attacks.

    Parameters
    ----------
    seed : int
        Random seed for reproducible experiments.
    paraphrase_synonyms : dict, optional
        Custom synonym pool for word swapping.
    """

    def __init__(
        self,
        seed: int = 42,
        paraphrase_synonyms: Optional[Dict[str, List[str]]] = None,
    ):
        self.seed = seed
        self.paraphrase_synonyms = (
            paraphrase_synonyms if paraphrase_synonyms is not None else PARAPHRASE_SYNONYMS
        )

    def paraphrase(self, tokens: List[str], swap_rate: float) -> List[str]:
        """
        Simulate paraphrasing by randomly replacing a fraction of tokens with synonyms.

        Parameters
        ----------
        tokens : list[str]
            Input tokens.
        swap_rate : float
            Fraction of tokens to replace [0.0, 1.0].

        Returns
        -------
        list[str]
            Paraphrased token list.
        """
        rng = random.Random(self.seed)
        result = list(tokens)
        indices = list(range(len(tokens)))
        rng.shuffle(indices)
        n_swap = int(len(tokens) * swap_rate)
        for i in indices[:n_swap]:
            token_lower = tokens[i].lower()
            if token_lower in self.paraphrase_synonyms:
                syns = self.paraphrase_synonyms[token_lower]
                replacement = rng.choice(syns)
                # Preserve capitalization
                if tokens[i][0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                result[i] = replacement
        return result

    def evaluate(
        self,
        watermarker: Union[GreenRedWatermark, SemanticWatermark, StatisticalWatermark],
        tokens: List[str],
        swap_rates: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run robustness evaluation at multiple swap rates.

        Parameters
        ----------
        watermarker : watermarker instance
        tokens : list[str]
            *Watermarked* tokens to test against.
        swap_rates : list[float], optional
            Noise levels. Default [0.0, 0.1, 0.2, 0.3].

        Returns
        -------
        list[dict]
            One record per swap rate with detection results.
        """
        if swap_rates is None:
            swap_rates = [0.0, 0.1, 0.2, 0.3]

        detector = WatermarkDetector(watermarker)
        results = []

        for rate in swap_rates:
            noisy_tokens = self.paraphrase(tokens, rate)
            detection = detector.detect_tokens(noisy_tokens)
            results.append({
                "swap_rate": rate,
                "is_watermarked": detection["is_watermarked"],
                **{k: v for k, v in detection.items() if k not in ("is_watermarked", "method")},
            })

        return results

    def evaluate_all(
        self,
        watermarked_tokens: List[str],
        green_red: Optional[GreenRedWatermark] = None,
        semantic: Optional[SemanticWatermark] = None,
        statistical: Optional[StatisticalWatermark] = None,
        swap_rates: Optional[List[float]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run robustness evaluation for all provided watermarkers.

        Returns
        -------
        dict
            Keys: method names → list of per-swap-rate results.
        """
        results = {}
        if green_red is not None:
            results["GreenRedWatermark"] = self.evaluate(green_red, watermarked_tokens, swap_rates)
        if semantic is not None:
            results["SemanticWatermark"] = self.evaluate(semantic, watermarked_tokens, swap_rates)
        if statistical is not None:
            results["StatisticalWatermark"] = self.evaluate(statistical, watermarked_tokens, swap_rates)
        return results
