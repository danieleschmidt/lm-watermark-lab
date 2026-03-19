"""
SemanticWatermark: Embed a bit pattern in synonym choices.

Algorithm:
- Maintain a synonym dictionary: {canonical_word → [synonym_0, synonym_1, ...]}
- The watermark bit for each canonical word is derived deterministically from
  (secret_key, canonical_word) — NOT from a positional counter.
  This makes detection word-order-independent and robust to noise.
- Encoding: when a canonical word appears, replace it with synonym[bit % len(syns)]
  where bit = key_bit(canonical_word).
- Detection: for each token that maps to a canonical word, check if the synonym
  chosen matches the expected bit. Count matches vs mismatches.
"""

import hashlib
from typing import List, Optional, Dict


# A curated synonym dictionary. Each entry: canonical_word → [syn_bit0, syn_bit1]
DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    # Articles / determiners
    "a": ["a", "one"],
    "the": ["the", "this"],
    # Verbs
    "said": ["said", "stated"],
    "use": ["use", "employ"],
    "show": ["show", "demonstrate"],
    "help": ["help", "assist"],
    "make": ["make", "create"],
    "get": ["get", "obtain"],
    "give": ["give", "provide"],
    "find": ["find", "discover"],
    "know": ["know", "understand"],
    "think": ["think", "believe"],
    "see": ["see", "observe"],
    "look": ["look", "appear"],
    "want": ["want", "desire"],
    "seem": ["seem", "appear"],
    "need": ["need", "require"],
    "feel": ["feel", "sense"],
    "become": ["become", "grow"],
    "leave": ["leave", "depart"],
    "put": ["put", "place"],
    "mean": ["mean", "signify"],
    "keep": ["keep", "maintain"],
    "let": ["let", "allow"],
    "begin": ["begin", "start"],
    "reveal": ["reveal", "uncover"],
    "hear": ["hear", "perceive"],
    "play": ["play", "engage"],
    "run": ["run", "operate"],
    "move": ["move", "shift"],
    "live": ["live", "exist"],
    "happen": ["happen", "occur"],
    "change": ["change", "alter"],
    "differ": ["differ", "vary"],
    # Adjectives
    "important": ["important", "significant"],
    "large": ["large", "substantial"],
    "small": ["small", "tiny"],
    "good": ["good", "excellent"],
    "great": ["great", "remarkable"],
    "high": ["high", "elevated"],
    "different": ["different", "distinct"],
    "new": ["new", "novel"],
    "old": ["old", "ancient"],
    "big": ["big", "major"],
    "possible": ["possible", "feasible"],
    "real": ["real", "genuine"],
    "right": ["right", "correct"],
    "long": ["long", "extended"],
    "hard": ["hard", "difficult"],
    "early": ["early", "initial"],
    "little": ["little", "slight"],
    "public": ["public", "open"],
    "free": ["free", "unrestricted"],
    "simple": ["simple", "straightforward"],
    "clear": ["clear", "evident"],
    "strong": ["strong", "robust"],
    "fast": ["fast", "rapid"],
    "common": ["common", "widespread"],
    "special": ["special", "particular"],
    "certain": ["certain", "specific"],
    # Nouns
    "problem": ["problem", "challenge"],
    "method": ["method", "approach"],
    "system": ["system", "framework"],
    "result": ["result", "outcome"],
    "number": ["number", "quantity"],
    "point": ["point", "aspect"],
    "fact": ["fact", "reality"],
    "time": ["time", "period"],
    "work": ["work", "effort"],
    "group": ["group", "collection"],
    "area": ["area", "domain"],
    "case": ["case", "instance"],
    "example": ["example", "instance"],
    "type": ["type", "kind"],
    "place": ["place", "location"],
    "world": ["world", "realm"],
    "part": ["part", "component"],
    "level": ["level", "degree"],
    "end": ["end", "conclusion"],
    # Adverbs / connectors
    "also": ["also", "additionally"],
    "however": ["however", "nevertheless"],
    "therefore": ["therefore", "consequently"],
    "often": ["often", "frequently"],
    "usually": ["usually", "typically"],
    "only": ["only", "solely"],
    "very": ["very", "extremely"],
    "well": ["well", "effectively"],
    "just": ["just", "simply"],
    "even": ["even", "indeed"],
    "still": ["still", "yet"],
    "already": ["already", "previously"],
    "always": ["always", "consistently"],
    "now": ["now", "currently"],
    "actually": ["actually", "in reality"],
    "again": ["again", "once more"],
    "human": ["human", "personal"],
}


class SemanticWatermark:
    """
    Synonym-choice based watermarking.

    Each canonical word is assigned a fixed watermark bit derived from
    (secret_key, canonical_word). This makes encoding and detection
    independent of token position and robust to small noise.

    Parameters
    ----------
    secret_key : str
        Secret key used to derive per-word watermark bits.
    synonyms : dict, optional
        Custom synonym dictionary. Defaults to DEFAULT_SYNONYMS.
    """

    def __init__(
        self,
        secret_key: str = "default_key",
        synonyms: Optional[Dict[str, List[str]]] = None,
    ):
        self.secret_key = secret_key
        self.synonyms = synonyms if synonyms is not None else DEFAULT_SYNONYMS

        # Build reverse lookup: synonym_form → (canonical, synonym_index)
        self._reverse: Dict[str, tuple] = {}
        for canonical, syns in self.synonyms.items():
            for idx, syn in enumerate(syns):
                if syn not in self._reverse:
                    self._reverse[syn] = (canonical, idx)

    # ------------------------------------------------------------------
    # Per-word bit derivation
    # ------------------------------------------------------------------

    def _word_bit(self, canonical: str) -> int:
        """
        Derive a deterministic watermark bit for a canonical word.
        Uses HMAC-style: hash(secret_key || canonical).
        """
        payload = f"{self.secret_key}:{canonical}"
        digest = hashlib.sha256(payload.encode()).digest()
        return digest[0] & 1  # LSB of first byte

    # ------------------------------------------------------------------
    # Watermarking
    # ------------------------------------------------------------------

    def watermark(self, tokens: List[str]) -> List[str]:
        """
        Encode watermark by choosing synonyms based on per-word secret bits.

        Parameters
        ----------
        tokens : list[str]

        Returns
        -------
        list[str]
            Watermarked tokens.
        """
        result = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.synonyms:
                syns = self.synonyms[token_lower]
                bit = self._word_bit(token_lower)
                chosen_idx = bit % len(syns)
                chosen = syns[chosen_idx]
                # Preserve original capitalization
                if token[0].isupper():
                    chosen = chosen[0].upper() + chosen[1:]
                result.append(chosen)
            else:
                result.append(token)
        return result

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, tokens: List[str]) -> dict:
        """
        Detect watermark by checking synonym choices against expected word bits.

        For each token that maps to a canonical word, check:
        - Which synonym index was used (chosen_idx)
        - What index the secret key predicts (expected_idx)
        - Count matches vs total.

        Returns
        -------
        dict with keys:
            matched_bits, total_bits, match_fraction, is_watermarked
        """
        matched = 0
        total = 0

        for token in tokens:
            token_lower = token.lower()
            canonical = None
            chosen_idx = None

            if token_lower in self.synonyms:
                # Canonical form used (index 0)
                canonical = token_lower
                chosen_idx = 0
            elif token_lower in self._reverse:
                # A synonym was used
                canonical, chosen_idx = self._reverse[token_lower]
                if chosen_idx == 0:
                    # Index 0 synonyms are the same as canonical, skip ambiguous
                    continue

            if canonical is not None and chosen_idx is not None:
                expected_bit = self._word_bit(canonical)
                syns = self.synonyms[canonical]
                expected_idx = expected_bit % len(syns)
                if chosen_idx == expected_idx:
                    matched += 1
                total += 1

        if total == 0:
            return {
                "matched_bits": 0,
                "total_bits": 0,
                "match_fraction": 0.0,
                "is_watermarked": False,
            }

        match_fraction = matched / total

        # Watermarked if >70% bits match (vs 50% chance for random)
        return {
            "matched_bits": matched,
            "total_bits": total,
            "match_fraction": match_fraction,
            "is_watermarked": match_fraction > 0.70,
        }

    def decode_bits(self, tokens: List[str]) -> Dict[str, int]:
        """Return dict of {canonical_word: decoded_bit} for all matched tokens."""
        decoded = {}
        for token in tokens:
            token_lower = token.lower()
            canonical = None
            chosen_idx = None

            if token_lower in self.synonyms:
                canonical = token_lower
                chosen_idx = 0
            elif token_lower in self._reverse:
                canonical, chosen_idx = self._reverse[token_lower]
                if chosen_idx == 0:
                    continue

            if canonical is not None and chosen_idx is not None:
                decoded[canonical] = chosen_idx % 2
        return decoded
