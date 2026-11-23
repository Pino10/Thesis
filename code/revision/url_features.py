"""Recompute PhiUSIIL's URL-string features directly from a raw URL.

Used for comment 2 (end-to-end deployment timing): a deployed detector receives only
the URL text, so the features have to be computed before the model can run. Each
function here is checked against the dataset's own column values by
check_url_features.py, which reports how often they match exactly.

TLDLegitimateProb and URLCharProb are not computable from the URL alone: they need
lookup tables (a probability per top-level domain, and a probability per character).
Bank, Pay and Crypto are page-content features in PhiUSIIL and cannot be computed
from the URL at all.
"""
import ipaddress
import re
from urllib.parse import urlsplit

_SPECIAL = set("=?&")


def _host(url):
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _longest_runs(s):
    """Length of the longest run of letters, of digits and of other characters."""
    best = {"a": 0, "d": 0, "o": 0}
    cur_kind, cur_len = None, 0
    for ch in s:
        kind = "a" if ch.isalpha() else "d" if ch.isdigit() else "o"
        cur_len = cur_len + 1 if kind == cur_kind else 1
        cur_kind = kind
        best[kind] = max(best[kind], cur_len)
    return best


def extract(url, tld_table, char_table):
    """Return the 24 Baseline U feature values for one URL, in features_U.txt order.

    tld_table: dict top-level domain -> TLDLegitimateProb.
    char_table: dict character -> probability, used for URLCharProb.
    Bank, Pay and Crypto are set to 0 because they need the page content.
    """
    host = _host(url)
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    try:
        ipaddress.ip_address(host); is_ip = 1
    except ValueError:
        is_ip = 0
    n = len(url)
    rest = url.split("://", 1)[-1]
    letters = sum(c.isalpha() for c in rest)
    digits = sum(c.isdigit() for c in rest)
    eq, qm, amp = rest.count("="), rest.count("?"), rest.count("&")
    other_special = sum(1 for c in rest if not c.isalnum() and c not in _SPECIAL)
    obf = len(re.findall(r"%[0-9a-fA-F]{2}", url))
    runs = _longest_runs(rest)
    char_prob = sum(char_table.get(c, 0.0) for c in url.lower()) / max(n, 1)
    return [
        n,                                         # URLLength
        len(host),                                 # DomainLength
        is_ip,                                     # IsDomainIP
        (runs["a"] + runs["d"] + runs["o"]) / max(len(rest), 1),  # CharContinuationRate
        tld_table.get(tld, 0.0),                   # TLDLegitimateProb (lookup)
        char_prob,                                 # URLCharProb (lookup)
        len(tld),                                  # TLDLength
        max(host.count(".") - 1, 0),               # NoOfSubDomain
        int(obf > 0),                              # HasObfuscation
        obf,                                       # NoOfObfuscatedChar
        obf / max(n, 1),                           # ObfuscationRatio
        letters,                                   # NoOfLettersInURL
        letters / max(n, 1),                       # LetterRatioInURL
        digits,                                    # NoOfDegitsInURL
        digits / max(n, 1),                        # DegitRatioInURL
        eq,                                        # NoOfEqualsInURL
        qm,                                        # NoOfQMarkInURL
        amp,                                       # NoOfAmpersandInURL
        other_special,                             # NoOfOtherSpecialCharsInURL
        (other_special + eq + qm + amp) / max(n, 1),  # SpacialCharRatioInURL
        int(url.lower().startswith("https://")),   # IsHTTPS
        0, 0, 0,                                   # Bank, Pay, Crypto (page content, unavailable)
    ]
