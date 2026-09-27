"""Text normalization shared by blocking and feature extraction."""
import re
import unicodedata

import numpy as np
import pandas as pd

_NON_WORD = re.compile(r"[\W_]+")
_DIGITS = re.compile(r"\d+")

# Generic abbreviation expansions (street types, units, directions, legal forms).
# Applied identically to every record, so both spellings collapse to one token.
ABBREVIATIONS = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "cir": "circle",
    "ter": "terrace", "trl": "trail", "fwy": "freeway", "expy": "expressway",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "rm": "room", "no": "number", "num": "number",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "nr": "near", "opp": "opposite", "mkt": "market", "ngr": "nagar",
    "pvt": "private", "prvt": "private", "ltd": "limited",
    "corp": "corporation", "co": "company", "inc": "incorporated", "intl": "international",
    "mfg": "manufacturing", "svc": "services", "svcs": "services", "bros": "brothers",
    "assoc": "associates", "mgmt": "management",
}

# Legal forms and function words: carry almost no identity, so they are dropped
# from the "core" name used for blocking and fuzzy name features.
NAME_STOPWORDS = {
    "private", "limited", "company", "corporation", "incorporated", "llc", "llp", "lp",
    "pc", "pllc", "plc", "pa", "the", "and", "of", "dba", "aka",
    "sa", "sas", "sasu", "sarl", "eurl", "snc", "gmbh", "ag",
    "et", "de", "du", "des", "la", "le", "les", "d", "l",
}


def normalize(text: str) -> str:
    """Lowercase, strip accents (Cáble -> cable), drop punctuation, expand abbreviations."""
    if not text:
        return ""
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace("’", "")
    return " ".join(ABBREVIATIONS.get(t, t) for t in _NON_WORD.sub(" ", text).split())


def core_name(norm_name: str) -> str:
    """Normalized name with legal forms and function words removed."""
    return " ".join(t for t in norm_name.split() if t not in NAME_STOPWORDS)


def number_tokens(norm_addr: str) -> str:
    """Sorted unique numbers in an address (street/unit numbers, PIN/ZIP codes)."""
    return " ".join(sorted(set(_DIGITS.findall(norm_addr))))


def first_number(norm_addr: str) -> int:
    """First number in the address (usually the street number), or -1 if none."""
    m = _DIGITS.search(norm_addr)
    return int(m.group(0)[:15]) if m else -1


def non_ascii_fraction(text: str) -> float:
    """Share of non-ASCII characters; high for names written in a native script."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 127) / len(text)


def prepare_records(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the normalized columns the pipeline needs and drop the raw text.
    Memory matters here: Source 2/3 have millions of rows each."""
    names = df["business_name"].tolist()
    name_core = [core_name(normalize(s)) for s in names]
    addr_norm = [normalize(s) for s in df["business_address"].tolist()]

    return pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy(),
        "country": df["country"].str.strip().str.lower().to_numpy(),
        "name_core": name_core,
        "addr_norm": addr_norm,
        "addr_nums": [number_tokens(s) for s in addr_norm],
        "addr_first_num": np.array([first_number(s) for s in addr_norm], dtype=np.int64),
        "name_ntok": np.array([s.count(" ") + 1 if s else 0 for s in name_core], dtype=np.int16),
        "name_non_ascii": np.array([non_ascii_fraction(s) for s in names], dtype=np.float32),
    })


def load_source(path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read a source TSV in chunks and normalize each chunk, so the raw text
    of a multi-million-row file is never held in memory all at once."""
    parts = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=chunksize, quoting=3):
        parts.append(prepare_records(chunk))
    out = pd.concat(parts, ignore_index=True)
    out["country"] = out["country"].astype("category")
    return out


def name_keys(core: str):
    """Blocking keys from a core name: tokens and adjacent-token bigrams."""
    toks = [t for t in core.split() if len(t) > 1]
    keys = ["n:" + t for t in toks]
    keys += ["nb:" + a + "_" + b for a, b in zip(toks, toks[1:])]
    return keys


def addr_keys(norm_addr: str):
    """Blocking keys from a normalized address: tokens and adjacent-token bigrams."""
    toks = [t for t in norm_addr.split() if len(t) > 1 or t.isdigit()]
    keys = ["a:" + t for t in toks]
    keys += ["ab:" + a + "_" + b for a, b in zip(toks, toks[1:])]
    return keys
