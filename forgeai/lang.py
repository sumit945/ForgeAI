"""Language support for the major Indian languages (offline, no downloads).

Supported: Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Urdu, Kannada, Malayalam, Odia, Punjabi, English, and romanised
Hindi ("Hinglish"). Detection is by Unicode script plus function-word markers (to separate Hindi/Marathi and Urdu/Arabic).

What this module provides to the rest of the system
  * detect_language / corpus_language          which language is this text / this document folder
  * normalize_text                             one canonical form: NFC, native digits -> ASCII digits, danda/Arabic punctuation ->
                                               ASCII, zero-width joiners and nukta removed, Arabic-script variants unified
  * TOKEN_RE / TEXT_CLASS                      Unicode-aware word characters (matras and viramas stay inside words)
  * STOP_INDIC / stem_indic                    function-word lists and a light truncation stemmer for agglutinative scripts

The stopword lists and markers are compact, hand-written approximations (not linguistic resources): good enough for retrieval
and coverage checks, and meant to be extended by a native speaker.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# (script name, first code point, last code point, default language)
SCRIPTS = [
    ("devanagari", 0x0900, 0x097F, "hi"), ("bengali", 0x0980, 0x09FF, "bn"), ("gurmukhi", 0x0A00, 0x0A7F, "pa"),
    ("gujarati", 0x0A80, 0x0AFF, "gu"), ("odia", 0x0B00, 0x0B7F, "or"), ("tamil", 0x0B80, 0x0BFF, "ta"),
    ("telugu", 0x0C00, 0x0C7F, "te"), ("kannada", 0x0C80, 0x0CFF, "kn"), ("malayalam", 0x0D00, 0x0D7F, "ml"),
    ("arabic", 0x0600, 0x06FF, "ur"), ("arabic", 0x0750, 0x077F, "ur"), ("arabic", 0xFB50, 0xFDFF, "ur"), ("arabic", 0xFE70, 0xFEFF, "ur"),
]

LANGS = {
    "en": {"name": "English", "native": "English", "script": "latin"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "script": "devanagari"},
    "hi-Latn": {"name": "Hinglish (romanised Hindi)", "native": "Hinglish", "script": "latin"},
    "bn": {"name": "Bengali", "native": "বাংলা", "script": "bengali"},
    "mr": {"name": "Marathi", "native": "मराठी", "script": "devanagari"},
    "te": {"name": "Telugu", "native": "తెలుగు", "script": "telugu"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "script": "tamil"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "script": "gujarati"},
    "ur": {"name": "Urdu", "native": "اردو", "script": "arabic"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "script": "kannada"},
    "ml": {"name": "Malayalam", "native": "മലയാളം", "script": "malayalam"},
    "or": {"name": "Odia", "native": "ଓଡ଼ିଆ", "script": "odia"},
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "script": "gurmukhi"},
}
INDIAN = [c for c in LANGS if c not in ("en",)]

# word characters of every supported script (combining marks / matras / viramas are inside these blocks)
TEXT_CLASS = r"À-ɏऀ-ൿ؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿"
TOKEN_RE = re.compile(rf"[0-9a-z{TEXT_CLASS}]+(?:[.,][0-9]+)*")

# ------------------------------------------------------------------------------------------ digits & normalisation
_DIGIT_BLOCKS = [0x0660, 0x06F0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66]
DIGIT_MAP = {ord(chr(b + i)): ord(str(i)) for b in _DIGIT_BLOCKS for i in range(10)}
_PUNCT_MAP = {0x0964: ".", 0x0965: ".", 0x06D4: ".", 0x061F: "?", 0x060C: ",", 0x061B: ";", 0x0970: ".", 0x200C: None, 0x200D: None, 0x200B: None,
              0x093C: None, 0x0901: 0x0902, 0x064A: 0x06CC, 0x0649: 0x06CC, 0x0643: 0x06A9, 0x0640: None}
for _cp in range(0x064B, 0x0660):  # Arabic diacritics (harakat)
    _PUNCT_MAP[_cp] = None
_LIGATURES = {0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl", 0xFB05: "st", 0xFB06: "st"}
_TRANS = {**DIGIT_MAP, **{k: v for k, v in _PUNCT_MAP.items()}, **_LIGATURES}
_CTRL = re.compile(r"[‪-‮⁦-⁩﻿]")  # bidi controls and BOM


def normalize_text(text: str) -> str:
    """Canonical form used for matching: lower-case, NFC, native digits -> ASCII, danda & Arabic punctuation -> ASCII,
    joiners / nukta / diacritics dropped, Arabic-script letter variants unified."""
    t = unicodedata.normalize("NFC", _CTRL.sub("", text or "")).lower()
    return t.translate(_TRANS)


# ------------------------------------------------------------------------------------------ stopwords & stemming
_SW = {
    "hi": "का की के को में से पर है हैं था थे थी और या भी तो ने एक यह वह ये वे इस उस जो कि लिए तक साथ नहीं क्या कैसे कितने कितना कितनी कौन कब कहाँ कहां होगा होगी होंगे हो गया गई गए कर करें करना सकता सकते सकती रहा रही रहे अपने अपनी",
    "mr": "आहे आहेत होते होता नाही आणि व किंवा या हा ही हे तो ती ते च चा ची चे चा मध्ये साठी पासून वर काय किती कसे कोण केव्हा कुठे असे असेल करावा करू शकतो शकते त्या",
    "bn": "এর এই সেই এবং ও বা কি কী না নেই আছে ছিল হবে হয় হয়েছে থেকে জন্য সাথে মধ্যে একটি একটা তার তাদের কত কোন কখন কিভাবে করতে পারে করা",
    "ta": "மற்றும் ஒரு இந்த அந்த என்ன எவ்வளவு எத்தனை எப்போது எங்கே யார் எப்படி இல்லை உள்ளது உள்ளன இருக்கும் வேண்டும் செய்ய முடியும் அல்லது ஆனால் இது அது",
    "te": "మరియు ఒక ఈ ఆ ఏమి ఎంత ఎన్ని ఎప్పుడు ఎక్కడ ఎవరు ఎలా లేదు ఉంది ఉన్నాయి ఉంటుంది కావాలి చేయవచ్చు లేదా కానీ ఇది అది",
    "gu": "અને અથવા આ તે શું કેટલા કેટલું ક્યારે ક્યાં કોણ કેવી રીતે નથી છે હતું હશે માટે થી માં નો ની નું ના ને પર",
    "kn": "ಮತ್ತು ಅಥವಾ ಈ ಆ ಏನು ಎಷ್ಟು ಯಾವಾಗ ಎಲ್ಲಿ ಯಾರು ಹೇಗೆ ಇಲ್ಲ ಇದೆ ಇವೆ ಇರುತ್ತದೆ ಬೇಕು ಮಾಡಬಹುದು ಆದರೆ ಇದು ಅದು",
    "ml": "ഒപ്പം അല്ലെങ്കിൽ ഈ ആ എന്ത് എത്ര എപ്പോൾ എവിടെ ആര് എങ്ങനെ ഇല്ല ഉണ്ട് ആണ് വേണം ചെയ്യാം പക്ഷേ ഇത് അത് കൂടാതെ",
    "pa": "ਅਤੇ ਜਾਂ ਇਹ ਉਹ ਕੀ ਕਿੰਨੇ ਕਿੰਨਾ ਕਦੋਂ ਕਿੱਥੇ ਕੌਣ ਕਿਵੇਂ ਨਹੀਂ ਹੈ ਹਨ ਸੀ ਹੋਵੇਗਾ ਲਈ ਤੋਂ ਵਿੱਚ ਦਾ ਦੀ ਦੇ ਨੂੰ ਤੇ",
    "or": "ଏବଂ କିମ୍ବା ଏହି ସେହି କଣ କେତେ କେବେ କେଉଁଠି କିଏ କିପରି ନାହିଁ ଅଛି ଥିଲା ହେବ ପାଇଁ ରୁ ରେ ର କୁ",
    "ur": "اور یا یہ وہ کا کی کے کو میں سے پر ہے ہیں تھا تھی تھے کیا کتنے کتنا کب کہاں کون کیسے نہیں ہو گا گی گے لیے تک ساتھ",
    "hi-Latn": "hai hain kya kaise kitne kitna kitni nahi nahin mein main aap ka ki ke ko se par aur toh tha thi hoga karna karne chahiye kab kahan kaun kyun liye bhi",
}
STOP_INDIC: set[str] = {normalize_text(w) for words in _SW.values() for w in words.split()}

AGGLUTINATIVE = re.compile(r"[஀-௿ఀ-౿ಀ-೿ഀ-ൿ]")  # Tamil, Telugu, Kannada, Malayalam


def stem_indic(tok: str) -> str:
    """Truncation stemming (first 5 code points) for the agglutinative Dravidian scripts, where case/plural suffixes are
    glued to the word. A standard, robust baseline for retrieval; other scripts are left as they are."""
    if len(tok) > 6 and AGGLUTINATIVE.match(tok):
        return tok[:5]
    return tok


# ------------------------------------------------------------------------------------------ detection
_MARATHI = {normalize_text(w) for w in "आहे आहेत नाही आणि व मध्ये साठी पासून काय किती कसे असेल करावा त्या च्या ची चे कोणती कोणता कोणते लिहा करा छापा द्या माझ्या मला तुमचा तुम्ही आम्ही ज्या जे".split()}
_HINDI = {normalize_text(w) for w in "है हैं और में नहीं का की के को से क्या कितने कितना लिए होता होगा करें कौन करो लिखिए दें मुझे आपका हम".split()}
_HINGLISH = set(_SW["hi-Latn"].split()) - {"main", "par", "ka"}
_URDU_LETTERS = set("ٹڈڑںےہھگچپژکی")
_ENGLISH = set("the and of to in is are for with on at by from that this be as it an or not".split())


def _script_of(ch: str) -> str | None:
    cp = ord(ch)
    for name, lo, hi, _ in SCRIPTS:
        if lo <= cp <= hi:
            return name
    return None


def detect_language(text: str) -> dict:
    """{'code','name','script','confidence','scores'} - never raises; empty/symbol-only text -> English with confidence 0."""
    t = normalize_text(text or "")
    counts: Counter = Counter()
    latin = 0
    for ch in t:
        if not ch.isalpha():
            continue
        s = _script_of(ch)
        if s:
            counts[s] += 1
        elif ch.isascii():
            latin += 1
    indic_total = sum(counts.values())
    total = indic_total + latin
    if total < 3:
        return {"code": "en", **_info("en"), "confidence": 0.0, "scores": {}}
    if latin >= indic_total:
        toks = re.findall(r"[a-z]+", t)
        hits = sum(1 for w in toks if w in _HINGLISH)
        eng = sum(1 for w in toks if w in _ENGLISH)
        if len(set(w for w in toks if w in _HINGLISH)) >= 3 and hits / max(1, len(toks)) >= 0.15 and hits > eng:
            return {"code": "hi-Latn", **_info("hi-Latn"), "confidence": round(min(1.0, hits / max(1, len(toks)) * 3), 2), "scores": {"latin": latin}}
        return {"code": "en", **_info("en"), "confidence": round(latin / total, 2), "scores": {"latin": latin}}
    script = counts.most_common(1)[0][0]
    code = next(c for n, _, _, c in SCRIPTS if n == script)
    words = set(TOKEN_RE.findall(t))
    if script == "devanagari":
        code = "mr" if len(words & _MARATHI) > len(words & _HINDI) else "hi"
    elif script == "arabic":
        code = "ur"
    return {"code": code, **_info(code), "confidence": round(counts[script] / total, 2), "scores": dict(counts)}


def _info(code: str) -> dict:
    return {"name": LANGS[code]["name"], "script": LANGS[code]["script"]}


def corpus_language(texts: list[str], sample_chars: int = 4000) -> dict:
    """Dominant language of a document set (weighted by sampled text), plus the share of every language seen."""
    tally: Counter = Counter()
    for t in texts:
        if not t.strip():
            continue
        d = detect_language(t[:sample_chars])
        tally[d["code"]] += min(len(t), sample_chars)
    if not tally:
        return {"code": "en", "name": "English", "shares": {}, "mixed": False}
    total = sum(tally.values())
    shares = {c: round(v / total, 2) for c, v in tally.most_common()}
    top = next(iter(shares))
    return {"code": top, "name": LANGS[top]["name"], "shares": shares, "mixed": len([s for s in shares.values() if s >= 0.2]) > 1}


def same_script(a: str, b: str) -> bool:
    """Do two texts use the same dominant script? (used to detect an answer written in the wrong language)"""
    return detect_language(a)["script"] == detect_language(b)["script"]
