from __future__ import annotations

from difflib import get_close_matches
import re


HXH_ALIASES: dict[str, list[str]] = {
    "Hunter x Hunter": ["hunter x hunter", "hunter hunter", "hxh", "hunterxhunter"],
    "Killua Zoldyck": ["killua", "kilua", "killwa", "killua zoldyck", "kilwa"],
    "Gon Freecss": ["gon", "gon freecss", "gon freaks", "gon freeks", "gohn"],
    "Kurapika": ["kurapika", "kurapica", "kurapicka"],
    "Leorio Paradinight": ["leorio", "leorio paradinight", "leorio paladiknight"],
    "Hisoka Morow": ["hisoka", "hisuca", "hisoka morow", "hysoka"],
    "Chrollo Lucilfer": ["chrollo", "chrolo", "chrollo lucilfer", "kuroro"],
    "Phantom Troupe": ["phantom troupe", "phantom troop", "spiders", "genei ryodan"],
    "Hunter Association": ["hunter association", "hunters association", "hunters", "hunter", "hunter exam"],
    "Greed Island": ["greed island", "green island", "greed island game"],
    "Chimera Ant arc": ["chimera ant", "chimera ants", "kimera ant", "chimera ant arc"],
    "Nen": ["nen", "aura", "hatsu", "ten", "ren", "zetsu", "gyo", "en", "ko", "ken", "ryu", "shu", "in"],
    "Zoldyck Family": ["zoldyck", "zoldyck family", "zoldick", "killua family"],
    "Meruem": ["meruem", "merum", "mereum"],
    "Komugi": ["komugi"],
    "Neferpitou": ["neferpitou", "pitou", "neferpito"],
    "Shaiapouf": ["shaiapouf", "pouf"],
    "Menthuthuyoupi": ["menthuthuyoupi", "youpi", "yupi"],
    "Isaac Netero": ["netero", "isaac netero"],
    "Illumi Zoldyck": ["illumi", "illumi zoldyck"],
    "Silva Zoldyck": ["silva", "silva zoldyck"],
    "Zeno Zoldyck": ["zeno", "zeno zoldyck"],
    "Biscuit Krueger": ["biscuit", "bisky", "biscuit krueger"],
    "Kite": ["kite", "kaito"],
    "Ging Freecss": ["ging", "ging freecss"],
    "Alluka Zoldyck": ["alluka", "alluka zoldyck"],
    "Nanika": ["nanika"],
    "Bungee Gum": ["bungee gum", "bunge gum"],
    "Emperor Time": ["emperor time"],
    "Godspeed": ["godspeed", "god speed"],
    "Jajanken": ["jajanken", "janken", "jajanken rock", "jajanken scissors", "jajanken paper"],
    "Chain Jail": ["chain jail"],
    "Judgment Chain": ["judgment chain", "judgement chain"],
    "Skill Hunter": ["skill hunter"],
    "100-Type Guanyin Bodhisattva": ["100 type guanyin", "hyakushiki kannon", "guanyin bodhisattva"],
    "Yorknew City arc": ["yorknew", "yorknew city", "yorknew arc"],
    "Heavens Arena": ["heavens arena", "heaven arena", "heaven's arena"],
    "Hunter Exam arc": ["hunter exam", "exam arc"],
    "Succession Contest arc": ["succession contest", "succession war"],
    "Dark Continent": ["dark continent"],
}

HXH_KEYWORDS = {
    "hunter x hunter", "hunterxhunter", "hxh", "hunter", "hunters", "hunter exam",
    "nen", "aura", "ten", "zetsu", "ren", "hatsu", "gyo", "en", "ko", "ken",
    "ryu", "shu", "enhancement", "transmutation", "emission", "conjuration",
    "manipulation", "specialization", "arc", "arcs", "ability", "abilities", "card",
    "cards", "spell", "greed", "island", "exam", "troupe", "chimera", "ant",
    "zoldyck", "kurta", "scarlet eyes", "royal guard", "zodiacs", "kakin",
    "black whale", "meteor city", "dark continent",
}

BROAD_DETECT_ONLY_ALIASES = {
    "hunter", "hunters", "troupe", "greed", "chimera", "ten", "ren", "en", "ko", "ken", "ryu", "shu",
}

ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in HXH_ALIASES.items()
    for alias in aliases
}

FUZZY_ALIASES = sorted([alias for alias in ALIAS_TO_CANONICAL if len(alias) >= 4], key=len, reverse=True)


def basic_normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword or "'" in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def contains_any(text: str, keywords: set[str]) -> bool:
    return any(contains_keyword(text, keyword) for keyword in keywords)


def normalize_hxh_query(message: str) -> tuple[str, list[str]]:
    original = basic_normalize(message)
    normalized = original
    detected: list[str] = []

    for alias in sorted(ALIAS_TO_CANONICAL, key=len, reverse=True):
        canonical = ALIAS_TO_CANONICAL[alias]
        if contains_keyword(original, alias):
            if canonical in detected and alias != canonical.lower():
                continue
            if any(alias in existing.lower() and canonical != existing for existing in detected):
                continue
            if canonical not in detected:
                detected.append(canonical)
            if alias not in BROAD_DETECT_ONLY_ALIASES:
                normalized = re.sub(rf"\b{re.escape(alias)}\b", canonical.lower(), normalized)

    corrected_words: list[str] = []
    for word in original.split():
        if word in ALIAS_TO_CANONICAL:
            canonical = ALIAS_TO_CANONICAL[word]
            if any(word in existing.lower().split() and canonical != existing for existing in detected):
                corrected_words.append(word)
                continue
            corrected_words.append(word if word in BROAD_DETECT_ONLY_ALIASES else canonical.lower())
            if canonical not in detected:
                detected.append(canonical)
            continue

        replacement = None
        if len(word) >= 4:
            matches = get_close_matches(word, FUZZY_ALIASES, n=1, cutoff=0.78)
            if matches:
                replacement = ALIAS_TO_CANONICAL[matches[0]]
        if replacement:
            corrected_words.append(replacement.lower())
            if replacement not in detected:
                detected.append(replacement)
        else:
            corrected_words.append(word)

    fuzzy_normalized = re.sub(r"\s+", " ", " ".join(corrected_words)).strip()
    # Prefer exact phrase replacements when available; otherwise use fuzzy word corrections.
    if detected and normalized != original:
        return _dedupe_repeated_terms(re.sub(r"\s+", " ", normalized).strip()), detected
    return _dedupe_repeated_terms(fuzzy_normalized), detected


def _dedupe_repeated_terms(text: str) -> str:
    text = re.sub(r"\b(family|arc|freecss|morow|lucilfer|zoldyck)\s+\1\b", r"\1", text)
    text = text.replace("zoldyck family family", "zoldyck family")
    text = text.replace("chimera ant arc arc", "chimera ant arc")
    return re.sub(r"\s+", " ", text).strip()


def enrich_retrieval_query(query: str, detected_entities: list[str], question_type: str | None = None) -> str:
    additions: list[str] = []
    for entity in detected_entities:
        additions.append(entity)
        if entity == "Killua Zoldyck":
            additions.append("Zoldyck family Godspeed Transmutation Gon Freecss")
        elif entity == "Gon Freecss":
            additions.append("Jajanken Enhancement Ging Freecss Killua Zoldyck")
        elif entity == "Kurapika":
            additions.append("Scarlet Eyes Chain Jail Judgment Chain Emperor Time Phantom Troupe")
        elif entity == "Nen":
            additions.append("aura Ten Zetsu Ren Hatsu six Nen categories restrictions vows")
        elif entity == "Phantom Troupe":
            additions.append("Chrollo Lucilfer Spiders Yorknew City Kurta Clan")
        elif entity == "Greed Island":
            additions.append("cards spell cards Biscuit Razor Accompany Leave Angel's Breath")
        elif entity == "Chimera Ant arc":
            additions.append("Meruem Royal Guards Netero Neferpitou Shaiapouf Menthuthuyoupi Komugi")
        elif entity == "Hunter Association":
            additions.append("Hunter Exam Hunter license chairman Netero Zodiacs")

    if question_type == "list":
        additions.append("list categories members arcs types")
    elif question_type == "comparison":
        additions.append("compare difference similarities")
    elif question_type == "ability":
        additions.append("ability user Nen category effect limitation")

    return " ".join([query, *additions]).strip()
