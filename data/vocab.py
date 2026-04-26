from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = set()
        for ch in list(text):
            chars.add(ch)

        chars = list(chars)
        chars.sort()

        stoi = {}
        itos = {}
        counter = 0
        for ch in chars:
            stoi[ch] = counter
            itos[counter] = ch
            counter += 1
        
        print(stoi, itos)
        # return
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoding = []
        for ch in list(text):
            encoding.append(stoi[ch])
        return encoding

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoding = ""
        for id in ids:
            decoding += itos[id]
        return decoding
