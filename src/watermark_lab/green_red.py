"""
GreenRedWatermark: Kirchenbauer et al. 2023 hash-based watermarking.

Reference: "A Watermark for Large Language Models", Kirchenbauer et al. 2023
https://arxiv.org/abs/2301.10226

Algorithm:
- Tokenize text into words.
- For each token position, use the previous token to seed a hash.
- Deterministically partition vocabulary into "green" (gamma fraction) and "red" lists.
- During watermarking, bias selection toward green tokens.
- Detection: count green token ratio, perform z-test.
"""

import hashlib
import math
from typing import List, Optional, Tuple


class GreenRedWatermark:
    """
    Hash-based green/red list watermarking (Kirchenbauer et al. 2023).

    Parameters
    ----------
    secret_key : str
        Secret key for watermark. Changes which tokens land in green list.
    gamma : float
        Fraction of vocabulary in green list (0 < gamma < 1). Default 0.5.
    delta : float
        Logit boost for green tokens (higher = stronger watermark, worse quality).
    vocab : list[str], optional
        Vocabulary to use. If None, will be built from text seen during watermarking.
    """

    def __init__(
        self,
        secret_key: str = "default_key",
        gamma: float = 0.5,
        delta: float = 2.0,
        vocab: Optional[List[str]] = None,
    ):
        self.secret_key = secret_key
        self.gamma = gamma
        self.delta = delta
        self._vocab: Optional[List[str]] = vocab
        self._vocab_index: Optional[dict] = None

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    def _build_vocab(self, tokens: List[str]) -> List[str]:
        seen = sorted(set(tokens))
        return seen

    def _ensure_vocab(self, tokens: List[str]) -> None:
        if self._vocab is None:
            self._vocab = self._build_vocab(tokens)
            self._vocab_index = {w: i for i, w in enumerate(self._vocab)}
        elif self._vocab_index is None:
            self._vocab_index = {w: i for i, w in enumerate(self._vocab)}

    def _token_hash(self, prev_token: str, candidate: str) -> int:
        """Deterministic hash of (prev_token, secret_key, candidate)."""
        payload = f"{self.secret_key}|{prev_token}|{candidate}"
        return int(hashlib.sha256(payload.encode()).hexdigest(), 16)

    def _is_green(self, prev_token: str, token: str) -> bool:
        """Return True if *token* is in the green list given *prev_token*."""
        h = self._token_hash(prev_token, token)
        # Map hash to [0, 1) range and compare to gamma
        return (h % 100000) / 100000 < self.gamma

    # ------------------------------------------------------------------
    # Watermarking
    # ------------------------------------------------------------------

    def watermark(self, tokens: List[str]) -> List[str]:
        """
        Watermark a token sequence.

        For each position we want to replace a token with a green-list alternative.
        Since we work post-generation (not during decoding), we simulate the bias
        by: if the current token is already green → keep it; otherwise replace it
        with the *most similar* green candidate from the vocab (first green token
        with the same leading character, falling back to the first green token).

        Parameters
        ----------
        tokens : list[str]
            Input word/token list.

        Returns
        -------
        list[str]
            Watermarked token list (may differ from input).
        """
        self._ensure_vocab(tokens)
        result = list(tokens)
        for i, token in enumerate(tokens):
            prev = tokens[i - 1] if i > 0 else "<BOS>"
            if not self._is_green(prev, token):
                # Try to find a green replacement
                green_candidate = self._find_green_replacement(prev, token)
                if green_candidate is not None:
                    result[i] = green_candidate
        return result

    def _find_green_replacement(self, prev_token: str, token: str) -> Optional[str]:
        """Find the best green-list replacement for a red token."""
        # Prefer same-first-letter token → best approximate synonym
        candidates_same = [w for w in self._vocab if w[0] == token[0] and self._is_green(prev_token, w)]
        if candidates_same:
            return min(candidates_same, key=lambda w: abs(len(w) - len(token)))
        # Fall back to any green token (won't replace — return None to preserve meaning)
        return None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, tokens: List[str]) -> dict:
        """
        Detect watermark in a token sequence.

        Counts how many tokens are in the green list given their preceding context.
        Performs a one-sided z-test against the null hypothesis (gamma fraction green).

        Parameters
        ----------
        tokens : list[str]
            Token sequence to test.

        Returns
        -------
        dict with keys:
            green_count, total_tokens, green_fraction, z_score, p_value, is_watermarked
        """
        self._ensure_vocab(tokens)
        if len(tokens) < 2:
            return {
                "green_count": 0,
                "total_tokens": len(tokens),
                "green_fraction": 0.0,
                "z_score": 0.0,
                "p_value": 0.5,
                "is_watermarked": False,
            }

        green_count = 0
        total = 0
        for i, token in enumerate(tokens[1:], start=1):
            prev = tokens[i - 1]
            if self._is_green(prev, token):
                green_count += 1
            total += 1

        green_fraction = green_count / total if total > 0 else 0.0

        # One-sided z-test: H0: fraction = gamma, H1: fraction > gamma
        # z = (observed - expected) / std  where std = sqrt(gamma*(1-gamma)/n)
        expected = self.gamma
        std = math.sqrt(expected * (1 - expected) / total) if total > 0 else 1.0
        z_score = (green_fraction - expected) / std if std > 0 else 0.0

        # Approximate p-value from standard normal CDF (upper tail)
        p_value = self._normal_sf(z_score)

        return {
            "green_count": green_count,
            "total_tokens": total,
            "green_fraction": green_fraction,
            "z_score": z_score,
            "p_value": p_value,
            "is_watermarked": p_value < 0.05,
        }

    @staticmethod
    def _normal_sf(z: float) -> float:
        """Survival function (1 - CDF) of standard normal using erf approximation."""
        # P(Z > z) = 0.5 * erfc(z / sqrt(2))
        return 0.5 * math.erfc(z / math.sqrt(2))
