# app/tamil_romanizer.py
"""
Custom Tamil -> ASCII Romanization (grapheme-based, library-free).
"""

_VOWELS = {
    "அ": "a",  "ஆ": "aa", "இ": "i",  "ஈ": "ii",
    "உ": "u",  "ஊ": "uu", "எ": "e",  "ஏ": "ae",
    "ஐ": "ai", "ஒ": "o",  "ஓ": "oo", "ஔ": "au",
}

_VOWEL_SIGNS = {
    "\u0BBE": "aa",
    "\u0BBF": "i",
    "\u0BC0": "ii",
    "\u0BC1": "u",
    "\u0BC2": "uu",
    "\u0BC6": "e",
    "\u0BC7": "ae",
    "\u0BC8": "ai",
    "\u0BCA": "o",
    "\u0BCB": "oo",
    "\u0BCC": "au",
}

_CONSONANTS = {
    "க": "k",  "ங": "ng", "ச": "s",  "ஞ": "nj",
    "ட": "t",  "ண": "N",  "த": "t",  "ந": "n",
    "ப": "p",  "ம": "m",  "ய": "y",  "ர": "r",
    "ல": "l",  "வ": "v",  "ழ": "L",  "ள": "L",
    "ற": "R",  "ன": "n",
    "ஜ": "j",  "ஶ": "sh", "ஷ": "Sh", "ஸ": "s",  "ஹ": "h",
}

_PULLI = "\u0BCD"
_AYTHAM = "ஃ"
_INHERENT = "a"


def tamil_to_ascii(text: str) -> str:
    if not text:
        return ""

    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]

        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            if i + 1 < n:
                nxt = text[i + 1]
                if nxt in _VOWEL_SIGNS:
                    out.append(_VOWEL_SIGNS[nxt])
                    i += 2
                    continue
                if nxt == _PULLI:
                    i += 2
                    continue
            out.append(_INHERENT)
            i += 1
            continue

        if ch in _VOWELS:
            out.append(_VOWELS[ch])
            i += 1
            continue

        if ch == _AYTHAM:
            out.append("h")
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)
