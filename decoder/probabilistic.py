import re
import numpy as np
from typing import List, Dict, Tuple, Optional

# Typical character confusion tables
DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A',
    '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
}

LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'Z': '2', 'E': '3', 'A': '4',
    'S': '5', 'G': '6', 'T': '7', 'B': '8', 'Q': '0', 'D': '0'
}

class ProbabilisticDecoder:
    """Corrects typical OCR errors by aligning character types with expected Indian Plate layouts."""

    @classmethod
    def generate_alternatives(cls, text: str, max_changes: int = 2) -> List[str]:
        """Generates possible alternative OCR interpretations by permuting commonly confused characters."""
        cleaned = text.upper().strip()
        cleaned = re.sub(r"[^A-Z0-9↑^]", "", cleaned)
        if not cleaned:
            return []

        # Define character confusion groups
        confusion_groups = [
            set(['0', 'O', 'D', 'Q', 'C']),
            set(['1', 'I', 'T', 'L', 'J']),
            set(['2', 'Z']),
            set(['3', 'E']),
            set(['4', 'A']),
            set(['5', 'S']),
            set(['6', 'G']),
            set(['7', 'T']),
            set(['8', 'B', 'S']),
            set(['9', 'G', 'P'])
        ]

        # Build confusion map
        confusion_map = {}
        for group in confusion_groups:
            for char in group:
                if char not in confusion_map:
                    confusion_map[char] = set()
                confusion_map[char].update(group)

        # Ensure all characters are in map (at least mapping to themselves)
        for char in cleaned:
            if char not in confusion_map:
                confusion_map[char] = {char}

        results = set()
        n = len(cleaned)

        def backtrack(idx, current_chars, changes):
            if idx == n:
                results.add("".join(current_chars))
                return
            
            orig_char = cleaned[idx]
            
            # Option 1: Keep original character (no change cost)
            current_chars[idx] = orig_char
            backtrack(idx + 1, current_chars, changes)
            
            # Option 2: Try alternatives if we haven't hit the max_changes limit
            if changes < max_changes:
                for alt in confusion_map.get(orig_char, {}):
                    if alt != orig_char:
                        current_chars[idx] = alt
                        backtrack(idx + 1, current_chars, changes + 1)

        backtrack(0, [None] * n, 0)
        
        # Sort candidates to return the original text first, and then other variations
        sorted_results = sorted(list(results), key=lambda x: (
            cls.edit_distance(cleaned, x), # prioritize fewer edits
            x != cleaned # keep original text first
        ))
        return sorted_results

    @classmethod
    def edit_distance(cls, s1: str, s2: str) -> int:
        """Computes Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return cls.edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @classmethod
    def correct_confusions(cls, text: str) -> str:
        """Contextually swaps characters based on typical Indian plate positional requirements."""
        cleaned = text.upper().strip()
        cleaned = re.sub(r"[^A-Z0-9↑^]", "", cleaned)
        if not cleaned:
            return ""

        # Let's map military arrow keys to a special character class 'S'
        char_types = []
        for char in cleaned:
            if char in ('↑', '^'):
                char_types.append('S')
            elif char.isalpha():
                char_types.append('L')
            else:
                char_types.append('D')

        n = len(cleaned)

        # Define template dictionary for standard Indian plate layouts
        templates_dict = {
            11: [
                ['S', 'D', 'D', 'L', 'D', 'D', 'D', 'D', 'D', 'D', 'L'], # ↑15D123456K
            ],
            10: [
                ['L', 'L', 'D', 'D', 'L', 'L', 'D', 'D', 'D', 'D'],  # MH12GP1234
                ['L', 'L', 'D', 'L', 'L', 'L', 'D', 'D', 'D', 'D'],  # DL3CAY1111
                ['L', 'L', 'D', 'D', 'L', 'L', 'L', 'D', 'D', 'D'],  # MH12GPX123
                ['D', 'D', 'L', 'L', 'D', 'D', 'D', 'D', 'L', 'L'],  # 21BH1234AA
            ],
            9: [
                ['L', 'L', 'D', 'D', 'L', 'L', 'D', 'D', 'D'],      # MH12GP123
                ['L', 'L', 'D', 'D', 'L', 'D', 'D', 'D', 'D'],      # MH12G1234
                ['L', 'L', 'D', 'L', 'L', 'D', 'D', 'D', 'D'],      # DL3CA1111
                ['L', 'L', 'D', 'L', 'L', 'L', 'D', 'D', 'D'],      # DL3CAY111
            ],
            8: [
                ['L', 'L', 'D', 'D', 'L', 'L', 'D', 'D'],          # MH12GP12
                ['L', 'L', 'D', 'D', 'L', 'D', 'D', 'D'],          # MH12G123
                ['L', 'L', 'D', 'L', 'L', 'D', 'D', 'D'],          # DL3CA111
                ['L', 'L', 'D', 'L', 'L', 'L', 'D', 'D'],          # DL3CAY11
                ['L', 'L', 'D', 'D', 'D', 'D', 'D', 'D'],          # MH121234
                ['D', 'D', 'D', 'L', 'L', 'D', 'D', 'D'],          # 111CD123
            ],
            7: [
                ['L', 'L', 'D', 'D', 'L', 'L', 'D'],              # MH12GP1
                ['L', 'L', 'D', 'D', 'L', 'D', 'D'],              # MH12G12
                ['L', 'L', 'D', 'L', 'L', 'D', 'D'],              # DL3CA12
                ['L', 'L', 'D', 'D', 'D', 'D', 'D'],              # MH12123
                ['D', 'D', 'L', 'L', 'D', 'D', 'D'],              # 11CD123
            ],
            6: [
                ['L', 'L', 'D', 'D', 'L', 'D'],                  # MH12G1
                ['L', 'L', 'D', 'L', 'L', 'D'],                  # DL3CA1
                ['L', 'L', 'D', 'D', 'D', 'D'],                  # MH1212
                ['D', 'D', 'L', 'L', 'D', 'D'],                  # 11CD12
            ],
        }

        best_template = None
        best_penalty = 9999
        best_conversions = []

        if n in templates_dict:
            for template in templates_dict[n]:
                penalty = 0
                conversions = []
                for idx, t_type in enumerate(template):
                    c = cleaned[idx]
                    if t_type == 'S':
                        if c not in ('↑', '^'):
                            penalty += 10
                    elif t_type == 'L':
                        if not c.isalpha():
                            # Digit needs to be converted to Letter
                            if c in DIGIT_TO_LETTER:
                                conversions.append((idx, DIGIT_TO_LETTER[c]))
                                penalty += 1
                            else:
                                penalty += 10 # non-convertible
                        else:
                            conversions.append((idx, c))
                    elif t_type == 'D':
                        if not c.isdigit():
                            # Letter needs to be converted to Digit
                            if c in LETTER_TO_DIGIT:
                                conversions.append((idx, LETTER_TO_DIGIT[c]))
                                penalty += 1
                            else:
                                penalty += 10 # non-convertible
                        else:
                            conversions.append((idx, c))
                
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_template = template
                    best_conversions = conversions

        # If we found a template matching with acceptable edits (penalty < 5)
        if best_template and best_penalty < 5:
            chars = list(cleaned)
            for idx, new_char in best_conversions:
                chars[idx] = new_char
            return "".join(chars)

        # Fallback to general letter/digit heuristics if no length-based template matches
        chars = list(cleaned)
        
        # Rule 1: First two characters must be letters (State code)
        for i in range(min(2, len(chars))):
            if chars[i].isdigit() and chars[i] in DIGIT_TO_LETTER:
                chars[i] = DIGIT_TO_LETTER[chars[i]]

        # Rule 2: Last characters are usually digits. Let's find trailing letters and correct them
        # only if they are preceded by digits and appear towards the end
        if len(chars) >= 4:
            # Check the last character
            if chars[-1].isalpha() and chars[-1] in LETTER_TO_DIGIT:
                chars[-1] = LETTER_TO_DIGIT[chars[-1]]
            # Check the second to last character if it is a letter and the last one is a digit (or was corrected to one)
            if len(chars) >= 5 and chars[-2].isalpha() and chars[-1].isdigit():
                # Correct only if it makes sense contextually
                if chars[-2] in LETTER_TO_DIGIT:
                    chars[-2] = LETTER_TO_DIGIT[chars[-2]]

        return "".join(chars)

class WatchlistMatcher:
    """Checks plate readings against a database watchlist using exact and similarity matching."""

    def __init__(self, watchlist: List[str]):
        # Standardize watchlist entries
        self.watchlist = [w.replace(" ", "").upper() for w in watchlist]
        self._init_faiss()

    def _init_faiss(self):
        """Pre-calculates representation vectors for FAISS searches if available."""
        self.use_faiss = False
        try:
            import faiss
            # Build a simple character occurrence matrix as embeddings
            # (or simple 2-gram frequencies) for fast indexing
            if len(self.watchlist) > 0:
                self.vocab = sorted(list(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789↑^")))
                self.vocab_char_to_idx = {char: i for i, char in enumerate(self.vocab)}
                self.dimension = len(self.vocab)
                
                # Create bag-of-character vectors
                vectors = []
                for plate in self.watchlist:
                    vec = np.zeros(self.dimension, dtype=np.float32)
                    for char in plate:
                        if char in self.vocab_char_to_idx:
                            vec[self.vocab_char_to_idx[char]] += 1
                    vectors.append(vec)
                
                self.vectors = np.array(vectors).astype('float32')
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(self.vectors)
                self.use_faiss = True
        except ImportError:
            # Fall back to NumPy
            pass

    def search(self, query_plate: str, threshold_distance: int = 2) -> List[Tuple[str, float]]:
        """Searches the watchlist and returns a list of (matched_plate, confidence_score)."""
        query_std = query_plate.replace(" ", "").upper()
        matches = []

        # 1. Check for exact match first
        if query_std in self.watchlist:
            matches.append((query_plate, 1.0))
            return matches

        # 2. Check using Levenshtein distance
        for watched in self.watchlist:
            dist = ProbabilisticDecoder.edit_distance(query_std, watched)
            if dist <= threshold_distance:
                # Calculate confidence score based on match similarity
                max_len = max(len(query_std), len(watched))
                similarity = 1.0 - (dist / max_len) if max_len > 0 else 0.0
                matches.append((watched, similarity))

        # Sort matches by similarity score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
