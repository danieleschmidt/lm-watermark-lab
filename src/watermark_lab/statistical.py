"""
StatisticalWatermark: Token probability distribution modification.

Algorithm:
- Maintain a "marker set" — tokens that have special significance (derived from key).
- At every N-th token position, prefer tokens from the marker set.
- The watermark pattern: positions 0, N, 2N, ... should have elevated marker-set usage.
- Detection: chi-square test comparing marker-set frequency at marked vs unmarked positions.
"""

import hashlib
import math
from typing import List, Optional, Tuple


class StatisticalWatermark:
    """
    Statistical distribution-based watermarking.

    Parameters
    ----------
    secret_key : str
        Secret key to derive the marker token set.
    period : int
        Every N-th token is a "marked" position. Default 4.
    marker_fraction : float
        Fraction of vocabulary treated as marker tokens. Default 0.2.
    vocab : list[str], optional
        Vocabulary. If None, built from first watermarked text.
    """

    def __init__(
        self,
        secret_key: str = "default_key",
        period: int = 4,
        marker_fraction: float = 0.2,
        vocab: Optional[List[str]] = None,
    ):
        self.secret_key = secret_key
        self.period = period
        self.marker_fraction = marker_fraction
        self._vocab: Optional[List[str]] = vocab
        self._marker_set: Optional[set] = None

        if vocab is not None:
            self._build_marker_set(vocab)

    # ------------------------------------------------------------------
    # Vocab / marker set helpers
    # ------------------------------------------------------------------

    def _build_marker_set(self, vocab: List[str]) -> None:
        self._vocab = sorted(set(vocab))
        n = max(1, int(len(self._vocab) * self.marker_fraction))
        # Deterministically select marker tokens using key hash
        order = self._key_shuffle(self._vocab)
        self._marker_set = set(order[:n])

    def _key_shuffle(self, items: List[str]) -> List[str]:
        """Return items in a deterministic order based on secret_key."""
        def sort_key(item: str) -> int:
            h = hashlib.sha256(f"{self.secret_key}|{item}".encode()).hexdigest()
            return int(h, 16)
        return sorted(items, key=sort_key)

    def _ensure_vocab(self, tokens: List[str]) -> None:
        if self._marker_set is None:
            self._build_marker_set(tokens)

    def is_marked_position(self, index: int) -> bool:
        """Return True if this token position is a 'marked' position."""
        return (index % self.period) == 0

    def is_marker_token(self, token: str) -> bool:
        """Return True if this token is in the marker set."""
        if self._marker_set is None:
            return False
        return token.lower() in self._marker_set

    # ------------------------------------------------------------------
    # Watermarking
    # ------------------------------------------------------------------

    def watermark(self, tokens: List[str]) -> List[str]:
        """
        At marked positions, replace non-marker tokens with the nearest marker token.

        Parameters
        ----------
        tokens : list[str]

        Returns
        -------
        list[str]
        """
        self._ensure_vocab(tokens)
        result = list(tokens)
        for i, token in enumerate(tokens):
            if self.is_marked_position(i) and not self.is_marker_token(token):
                replacement = self._find_marker_replacement(token)
                if replacement is not None:
                    result[i] = replacement
        return result

    def _find_marker_replacement(self, token: str) -> Optional[str]:
        """Find the closest marker token by string similarity."""
        if not self._marker_set:
            return None
        # Find marker token with most shared leading characters
        best = None
        best_score = -1
        for marker in self._marker_set:
            score = self._similarity(token.lower(), marker)
            if score > best_score:
                best_score = score
                best = marker
        return best

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Simple length-normalized common prefix score."""
        common = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                common += 1
            else:
                break
        return common / max(len(a), len(b), 1)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, tokens: List[str]) -> dict:
        """
        Detect watermark using chi-square test.

        Compare marker-token rate at marked positions vs unmarked positions.

        Returns
        -------
        dict with keys:
            marked_marker_rate, unmarked_marker_rate, chi2, p_value, is_watermarked
        """
        self._ensure_vocab(tokens)

        marked_total = 0
        marked_hits = 0
        unmarked_total = 0
        unmarked_hits = 0

        for i, token in enumerate(tokens):
            in_marker = self.is_marker_token(token)
            if self.is_marked_position(i):
                marked_total += 1
                if in_marker:
                    marked_hits += 1
            else:
                unmarked_total += 1
                if in_marker:
                    unmarked_hits += 1

        marked_rate = marked_hits / marked_total if marked_total > 0 else 0.0
        unmarked_rate = unmarked_hits / unmarked_total if unmarked_total > 0 else 0.0

        # Chi-square test on 2x2 contingency table:
        # [marked_hits, marked_misses]
        # [unmarked_hits, unmarked_misses]
        chi2, p_value = self._chi2_test(
            marked_hits, marked_total - marked_hits,
            unmarked_hits, unmarked_total - unmarked_hits,
        )

        return {
            "marked_marker_rate": marked_rate,
            "unmarked_marker_rate": unmarked_rate,
            "chi2": chi2,
            "p_value": p_value,
            "is_watermarked": p_value < 0.05 and marked_rate > unmarked_rate,
        }

    @staticmethod
    def _chi2_test(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
        """
        2x2 chi-square test.
        a = marked hits, b = marked misses
        c = unmarked hits, d = unmarked misses
        """
        n = a + b + c + d
        if n == 0:
            return 0.0, 1.0

        # Expected values
        row1 = a + b
        row2 = c + d
        col1 = a + c
        col2 = b + d

        if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
            return 0.0, 1.0

        e_a = row1 * col1 / n
        e_b = row1 * col2 / n
        e_c = row2 * col1 / n
        e_d = row2 * col2 / n

        chi2 = 0.0
        for obs, exp in [(a, e_a), (b, e_b), (c, e_c), (d, e_d)]:
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp

        # chi2 with 1 degree of freedom → p-value via regularized incomplete gamma
        p_value = StatisticalWatermark._chi2_sf(chi2, df=1)
        return chi2, p_value

    @staticmethod
    def _chi2_sf(x: float, df: int = 1) -> float:
        """Survival function of chi-squared distribution (1 df) via regularized gamma."""
        if x <= 0:
            return 1.0
        # For df=1: chi2_sf(x) = erfc(sqrt(x/2))
        return math.erfc(math.sqrt(x / 2))
