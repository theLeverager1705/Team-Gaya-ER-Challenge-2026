"""Text normalization for business names and addresses.

Country is treated as an OPEN set of labels: country-specific maps are applied when
the label is known, and a generic map is used for anything else (e.g. France or any
unseen country still gets the generic cleaning).
"""
import re
import unicodedata

# ---- legal suffixes (removed from the core name, kept as a class) ------------
LEGAL_SUFFIXES = {
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co",
    "limited": "ltd", "ltd": "ltd",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc",
    "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "opc": "opc",
    # French legal forms
    "sa": "sa", "sas": "sas", "sasu": "sas", "sarl": "sarl", "eurl": "sarl",
    "sci": "sci", "snc": "snc", "scop": "scop",
}

# ---- generic name abbreviations ------------------------------------------------
NAME_ABBR = {
    "intl": "international", "int'l": "international", "natl": "national",
    "mfg": "manufacturing", "svc": "services", "svcs": "services",
    "srvs": "services", "tech": "technologies", "techs": "technologies",
    "assoc": "associates", "bros": "brothers", "grp": "group",
    "ent": "enterprises", "entp": "enterprises", "ind": "industries",
    "sri": "shri", "shree": "shri", "laxmi": "lakshmi",
}

# ---- address abbreviations: generic + per-country overrides --------------------
ADDR_ABBR_GENERIC = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place",
    "sq": "square", "ste": "suite", "fl": "floor", "flr": "floor",
    "apt": "apartment", "bldg": "building", "no": "number",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "nr": "near", "opp": "opposite", "bhd": "behind",
}
ADDR_ABBR_BY_COUNTRY = {
    "india": {"mg": "mahatma gandhi", "ngr": "nagar", "clny": "colony", "sec": "sector"},
    # in French addresses "st" usually means "saint", and "r" means "rue"
    "france": {"st": "saint", "ste": "sainte", "r": "rue", "chem": "chemin",
               "fbg": "faubourg", "imp": "impasse", "rte": "route", "all": "allee"},
}

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_POSTAL = re.compile(r"\b(\d{6}|\d{5}(?:\s?\d{4})?)\b")  # IN PIN 6, US ZIP 5(+4), FR 5
_DIGITS = re.compile(r"\d+")


def strip_accents(s):
    """Unicode NFKD, then drop combining marks: 'Société' -> 'Societe'."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def basic_clean(s):
    """Lowercase, remove accents, '&' -> 'and', drop punctuation, collapse spaces."""
    s = strip_accents(str(s or "")).lower()
    s = s.replace("&", " and ").replace("+", " and ")
    s = s.replace("'", "")
    s = _PUNCT.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def country_key(country):
    """Normalized country label used for lookups and blocking (open set)."""
    return basic_clean(country) or "unknown"


def normalize_name(name):
    """Return (full_clean_name, core_name, legal_class).

    core_name = name with abbreviations expanded and trailing legal suffixes removed.
    """
    tokens = [NAME_ABBR.get(t, t) for t in basic_clean(name).split()]
    full = " ".join(tokens)
    legal = []
    # strip legal suffix tokens from the end (e.g. "... pvt ltd")
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        legal.append(LEGAL_SUFFIXES[tokens.pop()])
    # drop a leading "the"
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    core = " ".join(tokens) or full
    return full, core, " ".join(sorted(set(legal)))


def normalize_address(address, country):
    """Clean an address and expand abbreviations (country-specific map first)."""
    ck = country_key(country)
    local = ADDR_ABBR_BY_COUNTRY.get(ck, {})
    tokens = []
    for t in basic_clean(address).split():
        tokens.append(local.get(t, ADDR_ABBR_GENERIC.get(t, t)))
    return " ".join(tokens)


def postal_code(address):
    """Last 5/6-digit number in the raw address, spaces removed; '' if none."""
    found = _POSTAL.findall(str(address or ""))
    return found[-1].replace(" ", "")[:6] if found else ""


def acronym(core_name):
    """Initials of the core name: 'international business machines' -> 'ibm'."""
    toks = [t for t in core_name.split() if t not in {"and", "of", "the"}]
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


def add_normalized_columns(df):
    """Add all normalized fields used by blocking and features to a source frame."""
    df = df.copy()
    names = df["business_name"].map(normalize_name)
    df["name_full"] = names.str[0]
    df["name_core"] = names.str[1]
    df["legal"] = names.str[2]
    df["addr"] = [normalize_address(a, c) for a, c in zip(df["business_address"], df["country"])]
    df["postal"] = df["business_address"].map(postal_code)
    df["numbers"] = df["addr"].map(lambda s: " ".join(sorted(set(_DIGITS.findall(s)))))
    df["acronym"] = df["name_core"].map(acronym)
    df["ckey"] = df["country"].map(country_key)
    df["first_tok"] = df["name_core"].str.split().str[0].fillna("")
    return df
