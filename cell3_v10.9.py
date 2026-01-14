# @title 3. Patterns + Normalization { display-mode: "form" }
# @markdown BIP v10.9: Complete native patterns for moral concepts in 7 languages
# @markdown - Added: Sanskrit, Pali patterns
# @markdown - Added: NLP improvements (negation detection, modal classification)

import re
import unicodedata
from enum import Enum, auto

print("=" * 60)
print("TEXT NORMALIZATION & PATTERNS")
print("=" * 60)


# ===== TEXT NORMALIZATION =====
def normalize_hebrew(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u0591-\u05C7]", "", text)  # Remove nikud
    for final, regular in [
        ("\u05da", "\u05db"),
        ("\u05dd", "\u05de"),
        ("\u05df", "\u05e0"),
        ("\u05e3", "\u05e4"),
        ("\u05e5", "\u05e6"),
    ]:
        text = text.replace(final, regular)
    return text


def normalize_arabic(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u064B-\u065F]", "", text)  # Remove tashkeel
    text = text.replace("\u0640", "")  # Remove tatweel
    for v in ["\u0623", "\u0625", "\u0622", "\u0671"]:
        text = text.replace(v, "\u0627")
    text = text.replace("\u0629", "\u0647").replace("\u0649", "\u064a")
    return text


# NEW in v10.9: Sanskrit normalization
def normalize_sanskrit(text):
    """Normalize Sanskrit/Devanagari text."""
    text = unicodedata.normalize("NFC", text)
    # Remove vedic accents and other diacriticals
    text = re.sub(r"[\u0951-\u0954]", "", text)  # Vedic tone marks
    text = re.sub(r"[\u0900-\u0902]", "", text)  # Chandrabindu variants
    return text


# NEW in v10.9: Pali normalization
def normalize_pali(text):
    """Normalize Pali text (romanized or script)."""
    text = unicodedata.normalize("NFC", text)
    # Normalize romanized Pali diacritics
    text = text.lower()
    # Handle common Pali romanization variations
    text = text.replace("ṃ", "m").replace("ṅ", "n").replace("ñ", "n")
    text = text.replace("ṭ", "t").replace("ḍ", "d").replace("ṇ", "n")
    text = text.replace("ḷ", "l").replace("ā", "a").replace("ī", "i").replace("ū", "u")
    return text


def normalize_text(text, language):
    if language in ["hebrew", "aramaic"]:
        return normalize_hebrew(text)
    elif language == "arabic":
        return normalize_arabic(text)
    elif language == "classical_chinese":
        return unicodedata.normalize("NFKC", text)
    elif language == "sanskrit":
        return normalize_sanskrit(text)
    elif language == "pali":
        return normalize_pali(text)
    else:
        return unicodedata.normalize("NFKC", text.lower())


# ===== BOND AND HOHFELD TYPES =====
class BondType(Enum):
    HARM_PREVENTION = auto()
    RECIPROCITY = auto()
    AUTONOMY = auto()
    PROPERTY = auto()
    FAMILY = auto()
    AUTHORITY = auto()
    CARE = auto()
    FAIRNESS = auto()
    CONTRACT = auto()
    NONE = auto()


class HohfeldState(Enum):
    OBLIGATION = auto()
    RIGHT = auto()
    LIBERTY = auto()
    NO_RIGHT = auto()


# ===== COMPLETE BOND PATTERNS =====
ALL_BOND_PATTERNS = {
    "hebrew": {
        BondType.HARM_PREVENTION: [
            r"\u05d4\u05e8\u05d2",
            r"\u05e8\u05e6\u05d7",
            r"\u05e0\u05d6\u05e7",
            r"\u05d4\u05db\u05d4",
            r"\u05d4\u05e6\u05d9\u05dc",
            r"\u05e9\u05de\u05e8",
            r"\u05e4\u05e7\u05d5\u05d7.\u05e0\u05e4\u05e9",
        ],
        BondType.RECIPROCITY: [
            r"\u05d2\u05de\u05d5\u05dc",
            r"\u05d4\u05e9\u05d9\u05d1",
            r"\u05e4\u05e8\u05e2",
            r"\u05e0\u05ea\u05df.*\u05e7\u05d1\u05dc",
            r"\u05de\u05d3\u05d4.\u05db\u05e0\u05d2\u05d3",
        ],
        BondType.AUTONOMY: [
            r"\u05d1\u05d7\u05e8",
            r"\u05e8\u05e6\u05d5\u05df",
            r"\u05d7\u05e4\u05e9",
            r"\u05e2\u05e6\u05de",
        ],
        BondType.PROPERTY: [
            r"\u05e7\u05e0\u05d4",
            r"\u05de\u05db\u05e8",
            r"\u05d2\u05d6\u05dc",
            r"\u05d2\u05e0\u05d1",
            r"\u05de\u05de\u05d5\u05df",
            r"\u05e0\u05db\u05e1",
            r"\u05d9\u05e8\u05e9",
        ],
        BondType.FAMILY: [
            r"\u05d0\u05d1",
            r"\u05d0\u05de",
            r"\u05d1\u05e0",
            r"\u05db\u05d1\u05d3.*\u05d0\u05d1",
            r"\u05db\u05d1\u05d3.*\u05d0\u05de",
            r"\u05de\u05e9\u05e4\u05d7\u05d4",
            r"\u05d0\u05d7",
            r"\u05d0\u05d7\u05d5\u05ea",
        ],
        BondType.AUTHORITY: [
            r"\u05de\u05dc\u05db",
            r"\u05e9\u05d5\u05e4\u05d8",
            r"\u05e6\u05d5\u05d4",
            r"\u05ea\u05d5\u05e8\u05d4",
            r"\u05de\u05e6\u05d5\u05d4",
            r"\u05d3\u05d9\u05df",
            r"\u05d7\u05e7",
        ],
        BondType.CARE: [
            r"\u05d7\u05e1\u05d3",
            r"\u05e8\u05d7\u05de",
            r"\u05e2\u05d6\u05e8",
            r"\u05ea\u05de\u05db",
            r"\u05e6\u05d3\u05e7\u05d4",
        ],
        BondType.FAIRNESS: [
            r"\u05e6\u05d3\u05e7",
            r"\u05de\u05e9\u05e4\u05d8",
            r"\u05d9\u05e9\u05e8",
            r"\u05e9\u05d5\u05d4",
        ],
        BondType.CONTRACT: [
            r"\u05d1\u05e8\u05d9\u05ea",
            r"\u05e0\u05d3\u05e8",
            r"\u05e9\u05d1\u05d5\u05e2",
            r"\u05d4\u05ea\u05d7\u05d9\u05d1",
            r"\u05e2\u05e8\u05d1",
        ],
    },
    "aramaic": {
        BondType.HARM_PREVENTION: [
            r"\u05e7\u05d8\u05dc",
            r"\u05e0\u05d6\u05e7",
            r"\u05d7\u05d1\u05dc",
            r"\u05e9\u05d6\u05d9\u05d1",
            r"\u05e4\u05e6\u05d9",
        ],
        BondType.RECIPROCITY: [r"\u05e4\u05e8\u05e2", r"\u05e9\u05dc\u05de", r"\u05d0\u05d2\u05e8"],
        BondType.AUTONOMY: [r"\u05e6\u05d1\u05d9", r"\u05e8\u05e2\u05d5"],
        BondType.PROPERTY: [
            r"\u05d6\u05d1\u05e0",
            r"\u05e7\u05e0\u05d4",
            r"\u05d2\u05d6\u05dc",
            r"\u05de\u05de\u05d5\u05e0\u05d0",
            r"\u05e0\u05db\u05e1\u05d9",
        ],
        BondType.FAMILY: [
            r"\u05d0\u05d1\u05d0",
            r"\u05d0\u05de\u05d0",
            r"\u05d1\u05e8\u05d0",
            r"\u05d1\u05e8\u05ea\u05d0",
            r"\u05d9\u05e7\u05e8",
            r"\u05d0\u05d7\u05d0",
        ],
        BondType.AUTHORITY: [
            r"\u05de\u05dc\u05db\u05d0",
            r"\u05d3\u05d9\u05e0\u05d0",
            r"\u05d3\u05d9\u05d9\u05e0\u05d0",
            r"\u05e4\u05e7\u05d5\u05d3\u05d0",
            r"\u05d0\u05d5\u05e8\u05d9\u05ea",
        ],
        BondType.CARE: [r"\u05d7\u05e1\u05d3", r"\u05e8\u05d7\u05de", r"\u05e1\u05e2\u05d3"],
        BondType.FAIRNESS: [
            r"\u05d3\u05d9\u05e0\u05d0",
            r"\u05e7\u05e9\u05d5\u05d8",
            r"\u05ea\u05e8\u05d9\u05e6",
        ],
        BondType.CONTRACT: [
            r"\u05e7\u05d9\u05de\u05d0",
            r"\u05e9\u05d1\u05d5\u05e2\u05d4",
            r"\u05e0\u05d3\u05e8\u05d0",
            r"\u05e2\u05e8\u05d1\u05d0",
        ],
    },
    "classical_chinese": {
        BondType.HARM_PREVENTION: [
            r"\u6bba",
            r"\u5bb3",
            r"\u50b7",
            r"\u6551",
            r"\u8b77",
            r"\u885b",
            r"\u66b4",
        ],
        BondType.RECIPROCITY: [r"\u5831", r"\u9084", r"\u511f", r"\u8ced", r"\u7b54"],
        BondType.AUTONOMY: [r"\u81ea", r"\u7531", r"\u4efb", r"\u610f", r"\u5fd7"],
        BondType.PROPERTY: [
            r"\u8ca1",
            r"\u7269",
            r"\u7522",
            r"\u76dc",
            r"\u7aca",
            r"\u8ce3",
            r"\u8cb7",
        ],
        BondType.FAMILY: [
            r"\u5b5d",
            r"\u7236",
            r"\u6bcd",
            r"\u89aa",
            r"\u5b50",
            r"\u5f1f",
            r"\u5144",
            r"\u5bb6",
        ],
        BondType.AUTHORITY: [
            r"\u541b",
            r"\u81e3",
            r"\u738b",
            r"\u547d",
            r"\u4ee4",
            r"\u6cd5",
            r"\u6cbb",
        ],
        BondType.CARE: [r"\u4ec1", r"\u611b", r"\u6148", r"\u60e0", r"\u6069", r"\u6190"],
        BondType.FAIRNESS: [r"\u7fa9", r"\u6b63", r"\u516c", r"\u5e73", r"\u5747"],
        BondType.CONTRACT: [r"\u7d04", r"\u76df", r"\u8a93", r"\u8afe", r"\u4fe1"],
    },
    "arabic": {
        BondType.HARM_PREVENTION: [
            r"\u0642\u062a\u0644",
            r"\u0636\u0631\u0631",
            r"\u0627\u0630[\u064a\u0649]",
            r"\u0638\u0644\u0645",
            r"\u0627\u0646\u0642\u0630",
            r"\u062d\u0641\u0638",
            r"\u0627\u0645\u0627\u0646",
        ],
        BondType.RECIPROCITY: [
            r"\u062c\u0632\u0627",
            r"\u0631\u062f",
            r"\u0642\u0635\u0627\u0635",
            r"\u0645\u062b\u0644",
            r"\u0639\u0648\u0636",
        ],
        BondType.AUTONOMY: [
            r"\u062d\u0631",
            r"\u0627\u0631\u0627\u062f\u0629",
            r"\u0627\u062e\u062a\u064a\u0627\u0631",
            r"\u0645\u0634\u064a\u0626",
        ],
        BondType.PROPERTY: [
            r"\u0645\u0627\u0644",
            r"\u0645\u0644\u0643",
            r"\u0633\u0631\u0642",
            r"\u0628\u064a\u0639",
            r"\u0634\u0631\u0627",
            r"\u0645\u064a\u0631\u0627\u062b",
            r"\u063a\u0635\u0628",
        ],
        BondType.FAMILY: [
            r"\u0648\u0627\u0644\u062f",
            r"\u0627\u0628\u0648",
            r"\u0627\u0645",
            r"\u0627\u0628\u0646",
            r"\u0628\u0646\u062a",
            r"\u0627\u0647\u0644",
            r"\u0642\u0631\u0628[\u064a\u0649]",
            r"\u0631\u062d\u0645",
        ],
        BondType.AUTHORITY: [
            r"\u0637\u0627\u0639",
            r"\u0627\u0645\u0631",
            r"\u062d\u0643\u0645",
            r"\u0633\u0644\u0637\u0627\u0646",
            r"\u062e\u0644\u064a\u0641",
            r"\u0627\u0645\u0627\u0645",
            r"\u0634\u0631\u064a\u0639",
        ],
        BondType.CARE: [
            r"\u0631\u062d\u0645",
            r"\u0627\u062d\u0633\u0627\u0646",
            r"\u0639\u0637\u0641",
            r"\u0635\u062f\u0642",
            r"\u0632\u0643\u0627",
        ],
        BondType.FAIRNESS: [
            r"\u0639\u062f\u0644",
            r"\u0642\u0633\u0637",
            r"\u062d\u0642",
            r"\u0627\u0646\u0635\u0627\u0641",
            r"\u0633\u0648[\u064a\u0649]",
        ],
        BondType.CONTRACT: [
            r"\u0639\u0647\u062f",
            r"\u0639\u0642\u062f",
            r"\u0646\u0630\u0631",
            r"\u064a\u0645\u064a\u0646",
            r"\u0648\u0641\u0627",
            r"\u0627\u0645\u0627\u0646",
        ],
    },
    "english": {
        BondType.HARM_PREVENTION: [
            r"\bkill",
            r"\bmurder",
            r"\bharm",
            r"\bhurt",
            r"\bsave",
            r"\bprotect",
            r"\bviolence",
        ],
        BondType.RECIPROCITY: [
            r"\breturn",
            r"\brepay",
            r"\bexchange",
            r"\bgive.*back",
            r"\breciproc",
        ],
        BondType.AUTONOMY: [
            r"\bfree",
            r"\bchoice",
            r"\bchoose",
            r"\bconsent",
            r"\bautonomy",
            r"\bright to",
        ],
        BondType.PROPERTY: [
            r"\bsteal",
            r"\btheft",
            r"\bown",
            r"\bproperty",
            r"\bbelong",
            r"\binherit",
        ],
        BondType.FAMILY: [
            r"\bfather",
            r"\bmother",
            r"\bparent",
            r"\bchild",
            r"\bfamily",
            r"\bhonor.*parent",
        ],
        BondType.AUTHORITY: [
            r"\bobey",
            r"\bcommand",
            r"\bauthority",
            r"\blaw",
            r"\brule",
            r"\bgovern",
        ],
        BondType.CARE: [r"\bcare", r"\bhelp", r"\bkind", r"\bcompassion", r"\bcharity", r"\bmercy"],
        BondType.FAIRNESS: [r"\bfair", r"\bjust", r"\bequal", r"\bequity", r"\bright\b"],
        BondType.CONTRACT: [
            r"\bpromise",
            r"\bcontract",
            r"\bagreem",
            r"\bvow",
            r"\boath",
            r"\bcommit",
        ],
    },
    # NEW in v10.9: Sanskrit patterns (Devanagari)
    "sanskrit": {
        BondType.HARM_PREVENTION: [
            r"हिंसा",
            r"अहिंसा",
            r"वध",
            r"रक्षा",
            r"त्राण",
        ],  # himsa, ahimsa, vadha, raksha, trana
        BondType.RECIPROCITY: [
            r"प्रतिदान",
            r"प्रत्युपकार",
            r"दान",
            r"ऋण",
        ],  # pratidana, pratyupakara, dana, rna
        BondType.AUTONOMY: [r"स्वतंत्र", r"मोक्ष", r"स्वेच्छा"],  # swatantra, moksha, sveccha
        BondType.PROPERTY: [r"धन", r"स्व", r"चोर", r"दाय"],  # dhana, sva, chora, daya
        BondType.FAMILY: [r"पितृ", r"मातृ", r"पुत्र", r"कुल", r"गृह"],  # pitri, matri, putra, kula, grha
        BondType.AUTHORITY: [
            r"राज",
            r"धर्म",
            r"विधि",
            r"नियम",
            r"शास्त्र",
        ],  # raja, dharma, vidhi, niyama, shastra
        BondType.CARE: [
            r"करुणा",
            r"दया",
            r"प्रेम",
            r"मैत्री",
            r"सेवा",
        ],  # karuna, daya, prema, maitri, seva
        BondType.FAIRNESS: [r"न्याय", r"समता", r"धर्म", r"ऋत"],  # nyaya, samata, dharma, rta
        BondType.CONTRACT: [
            r"प्रतिज्ञा",
            r"संविद",
            r"वचन",
            r"शपथ",
        ],  # pratijna, samvid, vachana, shapatha
    },
    # NEW in v10.9: Pali patterns (romanized)
    "pali": {
        BondType.HARM_PREVENTION: [r"himsa", r"ahimsa", r"panatipata", r"rakkhati"],
        BondType.RECIPROCITY: [r"dana", r"patidana", r"ina"],
        BondType.AUTONOMY: [r"vimutti", r"nibbana", r"attadhipa"],
        BondType.PROPERTY: [r"dhana", r"theyya", r"adinnadana"],
        BondType.FAMILY: [r"mata", r"pita", r"putta", r"kula"],
        BondType.AUTHORITY: [r"raja", r"dhamma", r"vinaya", r"sikkhapada"],
        BondType.CARE: [r"karuna", r"metta", r"mudita", r"upekkha"],
        BondType.FAIRNESS: [r"samma", r"dhamma", r"sacca"],
        BondType.CONTRACT: [r"patijna", r"vacana", r"sacca"],
    },
}

# ===== COMPLETE HOHFELD PATTERNS =====
ALL_HOHFELD_PATTERNS = {
    "hebrew": {
        HohfeldState.OBLIGATION: [
            r"\u05d7\u05d9\u05d9\u05d1",
            r"\u05e6\u05e8\u05d9\u05db",
            r"\u05de\u05d5\u05db\u05e8\u05d7",
            r"\u05de\u05e6\u05d5\u05d5\u05d4",
        ],
        HohfeldState.RIGHT: [
            r"\u05d6\u05db\u05d5\u05ea",
            r"\u05e8\u05e9\u05d0\u05d9",
            r"\u05d6\u05db\u05d0\u05d9",
            r"\u05de\u05d2\u05d9\u05e2",
        ],
        HohfeldState.LIBERTY: [
            r"\u05de\u05d5\u05ea\u05e8",
            r"\u05e8\u05e9\u05d5\u05ea",
            r"\u05e4\u05d8\u05d5\u05e8",
            r"\u05d9\u05db\u05d5\u05dc",
        ],
        HohfeldState.NO_RIGHT: [
            r"\u05d0\u05e1\u05d5\u05e8",
            r"\u05d0\u05d9\u05e0\u05d5 \u05e8\u05e9\u05d0\u05d9",
            r"\u05d0\u05d9\u05df.*\u05d6\u05db\u05d5\u05ea",
        ],
    },
    "aramaic": {
        HohfeldState.OBLIGATION: [
            r"\u05d7\u05d9\u05d9\u05d1",
            r"\u05de\u05d7\u05d5\u05d9\u05d1",
            r"\u05d1\u05e2\u05d9",
        ],
        HohfeldState.RIGHT: [
            r"\u05d6\u05db\u05d5\u05ea",
            r"\u05e8\u05e9\u05d0\u05d9",
            r"\u05d6\u05db\u05d9",
        ],
        HohfeldState.LIBERTY: [
            r"\u05e9\u05e8\u05d9",
            r"\u05de\u05d5\u05ea\u05e8",
            r"\u05e4\u05d8\u05d5\u05e8",
        ],
        HohfeldState.NO_RIGHT: [
            r"\u05d0\u05e1\u05d5\u05e8",
            r"\u05dc\u05d0.*\u05e8\u05e9\u05d0\u05d9",
        ],
    },
    "classical_chinese": {
        HohfeldState.OBLIGATION: [r"\u5fc5", r"\u9808", r"\u7576", r"\u61c9", r"\u5b9c"],
        HohfeldState.RIGHT: [r"\u53ef", r"\u5f97", r"\u6b0a", r"\u5b9c"],
        HohfeldState.LIBERTY: [r"\u8a31", r"\u4efb", r"\u807d", r"\u514d"],
        HohfeldState.NO_RIGHT: [r"\u4e0d\u53ef", r"\u52ff", r"\u7981", r"\u83ab", r"\u975e"],
    },
    "arabic": {
        HohfeldState.OBLIGATION: [
            r"\u064a\u062c\u0628",
            r"\u0648\u0627\u062c\u0628",
            r"\u0641\u0631\u0636",
            r"\u0644\u0627\u0632\u0645",
            r"\u0648\u062c\u0648\u0628",
        ],
        HohfeldState.RIGHT: [
            r"\u062d\u0642",
            r"\u064a\u062d\u0642",
            r"\u062c\u0627\u0626\u0632",
            r"\u064a\u062c\u0648\u0632",
        ],
        HohfeldState.LIBERTY: [
            r"\u0645\u0628\u0627\u062d",
            r"\u062d\u0644\u0627\u0644",
            r"\u062c\u0627\u0626\u0632",
            r"\u0627\u0628\u0627\u062d",
        ],
        HohfeldState.NO_RIGHT: [
            r"\u062d\u0631\u0627\u0645",
            r"\u0645\u062d\u0631\u0645",
            r"\u0645\u0645\u0646\u0648\u0639",
            r"\u0644\u0627 \u064a\u062c\u0648\u0632",
            r"\u0646\u0647[\u064a\u0649]",
        ],
    },
    "english": {
        HohfeldState.OBLIGATION: [r"\bmust\b", r"\bshall\b", r"\bobligat", r"\bduty", r"\brequir"],
        HohfeldState.RIGHT: [r"\bright\b", r"\bentitle", r"\bdeserve", r"\bclaim"],
        HohfeldState.LIBERTY: [r"\bmay\b", r"\bpermit", r"\ballow", r"\bfree to"],
        HohfeldState.NO_RIGHT: [r"\bforbid", r"\bprohibit", r"\bmust not", r"\bshall not"],
    },
    # NEW in v10.9: Sanskrit Hohfeld patterns (Devanagari)
    "sanskrit": {
        HohfeldState.OBLIGATION: [r"कर्तव्य", r"अवश्य", r"नियम", r"विधि"],  # kartavya, avashya
        HohfeldState.RIGHT: [r"अधिकार", r"स्वत्व"],  # adhikara, svatva
        HohfeldState.LIBERTY: [r"शक्य", r"अनुज्ञा", r"उचित"],  # shakya, anujña
        HohfeldState.NO_RIGHT: [r"निषिद्ध", r"वर्जित", r"अकर्तव्य"],  # nishiddha, varjita
    },
    # NEW in v10.9: Pali Hohfeld patterns (romanized)
    "pali": {
        HohfeldState.OBLIGATION: [r"kicca", r"karaniiya", r"dhammo"],
        HohfeldState.RIGHT: [r"adhikaara", r"bhaaga"],
        HohfeldState.LIBERTY: [r"anujaanati", r"kappati"],
        HohfeldState.NO_RIGHT: [r"nisiddha", r"akaraniya", r"na kappati"],
    },
}


# ===== CONTEXT MARKERS FOR GRAMMAR-AWARE EXTRACTION =====
# These help distinguish "thou shalt not kill" from "he killed"
CONTEXT_MARKERS = {
    "hebrew": {
        "negation": [r"לא", r"אל", r"אין", r"בלי", r"אינ"],
        "obligation": [r"חייב", r"צריך", r"מוכרח", r"צווה"],
        "prohibition": [r"אסור", r"אל.*ת"],
        "permission": [r"מותר", r"רשאי", r"פטור"],
    },
    "aramaic": {
        "negation": [r"לא", r"לית", r"לאו"],
        "obligation": [r"חייב", r"בעי"],
        "prohibition": [r"אסור"],
        "permission": [r"שרי", r"מותר"],
    },
    "classical_chinese": {
        "negation": [r"不", r"非", r"無", r"未", r"毋"],
        "obligation": [r"必", r"當", r"須", r"應", r"宜"],
        "prohibition": [r"勿", r"禁", r"莫", r"不可"],
        "permission": [r"可", r"得", r"許"],
    },
    "arabic": {
        "negation": [r"لا", r"ما", r"ليس", r"لم", r"غير"],
        "obligation": [r"يجب", r"واجب", r"فرض", r"عليه"],
        "prohibition": [r"حرام", r"محرم", r"لا يجوز", r"نهى"],
        "permission": [r"حلال", r"مباح", r"جائز"],
    },
    "english": {
        "negation": [r"not", r"no", r"never", r"neither", r"n't"],
        "obligation": [r"must", r"shall", r"should", r"ought", r"required"],
        "prohibition": [r"forbid", r"prohibit", r"must not", r"shall not", r"don't"],
        "permission": [r"may", r"can", r"allowed", r"permit"],
    },
    # NEW in v10.9: Sanskrit context markers
    "sanskrit": {
        "negation": [r"न", r"मा", r"अ"],  # na, mā, a- prefix
        "obligation": [r"कर्तव्य", r"अवश्य", r"विधि"],
        "prohibition": [r"निषिद्ध", r"वर्जित", r"मा"],
        "permission": [r"शक्य", r"अनुज्ञा"],
    },
    # NEW in v10.9: Pali context markers
    "pali": {
        "negation": [r"na", r"ma", r"a-"],
        "obligation": [r"kicca", r"karaniya"],
        "prohibition": [r"nisiddha", r"akaraniya"],
        "permission": [r"anujaanati", r"kappati"],
    },
}


def detect_context(text, language, match_pos, window=30):
    """
    Detect grammatical context around a pattern match.
    Returns: ('prescriptive'/'descriptive'/'unknown', marker_type or None)
    """
    markers = CONTEXT_MARKERS.get(language, {})
    if not markers:
        return "unknown", None

    start = max(0, match_pos - window)
    end = min(len(text), match_pos + window)
    window_text = text[start:end]

    # Check for deontic markers (prescriptive = moral statement)
    for marker_type in ["prohibition", "obligation", "permission"]:
        for pattern in markers.get(marker_type, []):
            if re.search(pattern, window_text):
                return "prescriptive", marker_type

    # Check for simple negation (may be descriptive)
    for pattern in markers.get("negation", []):
        if re.search(pattern, window_text):
            return "descriptive", "negated"

    return "descriptive", None


# ===== NLP IMPROVEMENTS (v10.9 Phase 1) =====
# These provide negation detection and modal classification without external dependencies

NEGATION_CUES = {
    "english": ["not", "no", "never", "neither", "nor", "n't", "without", "lack", "none"],
    "classical_chinese": ["不", "非", "無", "莫", "勿", "未", "弗", "毋", "否"],
    "arabic": ["لا", "ما", "لم", "لن", "ليس", "غير", "بدون"],
    "hebrew": ["לא", "אל", "בלי", "אין", "מבלי"],
    "aramaic": ["לא", "לית", "לאו"],
    "sanskrit": ["न", "मा", "अ"],  # na, mā, a- (privative prefix)
    "pali": ["na", "ma", "a", "an"],
}

MODAL_CLASSIFICATION = {
    "english": {
        "obligation": ["must", "shall", "have to", "ought to", "need to", "required", "obligated"],
        "permission": ["may", "can", "allowed", "permitted", "free to", "entitled"],
        "prohibition": ["must not", "shall not", "cannot", "forbidden", "prohibited", "banned"],
        "supererogation": ["should", "ought", "would be good", "ideally", "preferably"],
    },
    "classical_chinese": {
        "obligation": ["必", "當", "宜", "須", "應", "要"],
        "permission": ["可", "得", "許", "容", "能"],
        "prohibition": ["不可", "不得", "勿", "莫", "禁", "不許", "不宜"],
        "supererogation": ["善", "美", "德", "宜"],
    },
    "arabic": {
        "obligation": ["يجب", "فرض", "واجب", "لازم", "فريضة"],
        "permission": ["يجوز", "مباح", "حلال", "جائز"],
        "prohibition": ["حرام", "محرم", "ممنوع", "لا يجوز", "محظور"],
        "supererogation": ["مستحب", "سنة", "مندوب", "نافلة"],
    },
    "hebrew": {
        "obligation": ["חייב", "מצווה", "צריך", "מוכרח", "חובה"],
        "permission": ["מותר", "רשאי", "יכול", "היתר"],
        "prohibition": ["אסור", "לא יעשה", "אל", "איסור"],
        "supererogation": ["ראוי", "טוב", "מידת חסידות", "לפנים משורת הדין"],
    },
    "sanskrit": {
        "obligation": ["कर्तव्य", "अवश्य", "नियम"],  # kartavya, avashya, niyama
        "permission": ["शक्य", "अनुज्ञा"],  # shakya, anujña
        "prohibition": ["निषिद्ध", "वर्जित", "मा"],  # nishiddha, varjita, mā
    },
    "pali": {
        "obligation": ["kicca", "karaniya", "dhamma"],
        "permission": ["kappati", "anujanati"],
        "prohibition": ["akappiya", "akaraniya", "na kappati"],
    },
}


def enhanced_extract_bond(text: str, language: str) -> dict:
    """
    Enhanced bond extraction with negation + modal detection.
    Phase 1 implementation - no external NLP dependencies required.

    Returns dict with:
        - bond_type: BondType or None
        - hohfeld_state: str (OBLIGATION/RIGHT/LIBERTY/NO_RIGHT)
        - negated: bool
        - modal: str or None (the matched modal marker)
        - confidence: float
        - context: str (prescriptive/descriptive/unknown)
    """
    # 1. Normalize text
    normalized = normalize_text(text, language)

    # 2. Check negation
    negation_cues = NEGATION_CUES.get(language, [])
    is_negated = any(cue in normalized for cue in negation_cues)

    # 3. Check modal and classify deontic status
    modal_status = "unknown"
    modal_text = None
    for status, markers in MODAL_CLASSIFICATION.get(language, {}).items():
        for marker in markers:
            if marker in normalized:
                modal_status = status
                modal_text = marker
                break
        if modal_status != "unknown":
            break

    # 4. Map modal to Hohfeld state
    hohfeld_map = {
        "obligation": "OBLIGATION",
        "permission": "LIBERTY",
        "prohibition": "NO_RIGHT",
        "supererogation": "LIBERTY",
        "unknown": "OBLIGATION",  # Default assumption
    }
    hohfeld = hohfeld_map[modal_status]

    # 5. Pattern matching for bond type
    bond_type = None
    confidence = 0.5
    for bt, patterns in ALL_BOND_PATTERNS.get(language, {}).items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                bond_type = bt
                confidence = 0.9
                break
        if bond_type:
            break

    # 6. Adjust confidence for negation
    if is_negated:
        confidence *= 0.8  # Lower confidence for negated statements

    # 7. Determine context
    if modal_status in ["obligation", "prohibition"]:
        context = "prescriptive"
    elif modal_status == "permission":
        context = "descriptive"  # Permissions are often statements of fact
    else:
        context = "unknown"

    return {
        "bond_type": bond_type,
        "hohfeld_state": hohfeld,
        "negated": is_negated,
        "modal": modal_text,
        "confidence": confidence,
        "context": context,
    }


print("\nContext markers defined for grammar-aware extraction")
print("  Detects: negation, obligation, prohibition, permission")

print(f"\nPatterns defined for {len(ALL_BOND_PATTERNS)} languages:")
for lang in ALL_BOND_PATTERNS:
    n = sum(len(p) for p in ALL_BOND_PATTERNS[lang].values())
    print(f"  {lang}: {n} bond patterns")

print("\nNLP improvements (Phase 1):")
print(f"  NEGATION_CUES: {len(NEGATION_CUES)} languages")
print(f"  MODAL_CLASSIFICATION: {len(MODAL_CLASSIFICATION)} languages")
print("  enhanced_extract_bond() ready")

print("\n" + "=" * 60)
