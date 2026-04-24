import re
from backend.app.config import ALLOWED_SMALLTALK

HXH_KEYWORDS = {
    "hunter x hunter", "hunterxhunter", "gon", "killua", "kurapika", "leorio", "hisoka",
    "chrollo", "phantom troupe", "nen", "zoldyck", "alluka", "nanika", "netero", "meruem",
    "chimera ant", "greed island", "yorknew", "heavens arena", "heaven's arena", "biscuit",
    "kite", "feitan", "shaiapouf", "pitou", "youpi", "hunter exam", "illum", "illumi"
}
TEAM_KEYWORDS = {
    "team", "member", "members", "full name", "academic level", "age", "university",
    "field of study", "study", "who are you", "your team"
}
SMALLTALK_KEYWORDS = {"hello", "hi", "hey", "help", "what can you do", "how do i use", "usage"}


def classify_intent(message: str) -> str:
    text = message.strip().lower()
    normalized = re.sub(r"\s+", " ", text)

    if any(k in normalized for k in TEAM_KEYWORDS):
        return "team_info"

    if any(k in normalized for k in HXH_KEYWORDS):
        return "hxh_knowledge"

    if ALLOWED_SMALLTALK and any(k in normalized for k in SMALLTALK_KEYWORDS):
        return "allowed_smalltalk"

    # Follow-up detection: if the user is asking a short pronoun-based follow-up,
    # we let retrieval + memory handle it inside the allowed domain.
    followup_markers = ["he", "she", "they", "it", "his", "her", "their", "that", "those", "this"]
    if len(normalized.split()) <= 12 and normalized.split()[:1] and normalized.split()[0] in followup_markers:
        return "hxh_knowledge"

    return "out_of_scope"
