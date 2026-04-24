from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

from ..config import TEAM_FILE
from ..models.schemas import TeamMember


def _clean(value: object) -> str:
    if value is None:
        return "Not provided"
    text = str(value).strip()
    if not text or text.lower().startswith("replace with"):
        return "Not provided"
    return text


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact(text: str) -> str:
    return normalize_text(text).replace(" ", "")


FUZZY_STOPWORDS = {
    "about", "tell", "know", "what", "who", "which", "where", "from", "does", "do",
    "you", "your", "the", "is", "are", "me", "give", "info", "describe", "how",
    "old", "age", "study", "studies", "university", "academic", "level",
}


@dataclass(frozen=True)
class TeamMatch:
    member: TeamMember
    matched_alias: str
    score: float


@dataclass(frozen=True)
class TeamAnswer:
    answer: str
    matched_member: str | None
    matched_alias: str | None


class TeamService:
    def __init__(self, team_file: Path = TEAM_FILE) -> None:
        self.team_file = Path(team_file)
        self._members: list[TeamMember] = []
        self._alias_entries: list[tuple[str, str, TeamMember]] = []
        self.reload()

    def reload(self) -> None:
        self._members = self._load_members()
        self._alias_entries = self._build_alias_entries(self._members)

    def load_team(self) -> list[TeamMember]:
        self.reload()
        return self.get_all_members()

    def _load_members(self) -> list[TeamMember]:
        if not self.team_file.exists():
            return []
        try:
            raw = json.loads(self.team_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        members: list[TeamMember] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            member = TeamMember(
                full_name=_clean(item.get("full_name")),
                academic_level=_clean(item.get("academic_level")),
                age=_clean(item.get("age")),
                university_name=_clean(item.get("university_name")),
                field_of_study=_clean(item.get("field_of_study")),
                aliases=[_clean(alias) for alias in item.get("aliases", []) if _clean(alias) != "Not provided"],
            )
            if member.full_name != "Not provided":
                members.append(member)
        return members

    def _build_alias_entries(self, members: list[TeamMember]) -> list[tuple[str, str, TeamMember]]:
        entries: list[tuple[str, str, TeamMember]] = []
        seen: set[tuple[str, str]] = set()
        for member in members:
            aliases = {member.full_name, *member.aliases}
            tokens = normalize_text(member.full_name).split()
            aliases.update(tokens)
            if len(tokens) >= 2:
                aliases.add(tokens[0])
                aliases.add(tokens[-1])
            for alias in list(aliases):
                compact_alias = _compact(alias)
                if len(compact_alias) >= 5:
                    aliases.add(compact_alias)

            for alias in aliases:
                normalized_alias = normalize_text(alias)
                if not normalized_alias:
                    continue
                key = (member.full_name, normalized_alias)
                if key not in seen:
                    entries.append((normalized_alias, alias, member))
                    seen.add(key)
        return sorted(entries, key=lambda item: len(item[0]), reverse=True)

    def get_all_members(self) -> list[TeamMember]:
        return list(self._members)

    def members(self) -> list[TeamMember]:
        return self.get_all_members()

    def clear_cache(self) -> None:
        self.reload()

    def as_context(self) -> str:
        if not self._members:
            return "No structured team data is currently available."
        lines = []
        for index, member in enumerate(self._members, start=1):
            lines.append(
                f"Member {index}: full_name={member.full_name}; "
                f"academic_level={member.academic_level}; age={member.age}; "
                f"university_name={member.university_name}; "
                f"field_of_study={member.field_of_study}; "
                f"aliases={', '.join(member.aliases) if member.aliases else 'none'}"
            )
        return "\n".join(lines)

    def find_member(self, query: str) -> TeamMatch | None:
        normalized_query = normalize_text(query)
        compact_query = _compact(query)
        best: TeamMatch | None = None

        for normalized_alias, original_alias, member in self._alias_entries:
            score = 0.0
            if re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_query):
                score = 1.0 + len(normalized_alias) / 100
            elif len(normalized_alias) >= 5 and normalized_alias in compact_query:
                score = 0.98
            elif len(normalized_alias) >= 4:
                query_tokens = [token for token in normalized_query.split() if token not in FUZZY_STOPWORDS]
                candidates = query_tokens + [_compact(" ".join(query_tokens[i : i + 2])) for i in range(len(query_tokens) - 1)]
                close = get_close_matches(normalized_alias, candidates, n=1, cutoff=0.78)
                if close:
                    score = SequenceMatcher(None, normalized_alias, close[0]).ratio()

            if score and (best is None or score > best.score):
                best = TeamMatch(member=member, matched_alias=original_alias, score=score)

        return best

    def find_member_by_full_name(self, full_name: str) -> TeamMatch | None:
        normalized = normalize_text(full_name)
        for member in self._members:
            if normalize_text(member.full_name) == normalized:
                return TeamMatch(member=member, matched_alias=member.full_name, score=1.0)
        return None

    def detect_team_intent(self, query: str) -> bool:
        normalized = normalize_text(query)
        if self.find_member(normalized):
            return True

        explicit_terms = [
            "your team", "project team", "team members", "teammates", "creator", "creators",
            "developer", "developers", "built you", "made you", "author", "authors",
            "who built you", "who made you", "team university", "list all team",
        ]
        if any(term in normalized for term in explicit_terms):
            return True

        if "team" in normalized and not any(term in normalized for term in ["killua", "gon", "kurapika", "phantom"]):
            return True

        field_terms = ["university", "school", "academic", "level", "age", "old", "study", "field", "students"]
        user_terms = ["you", "your", "from"]
        if "field of study" in normalized:
            return True

        pronoun_terms = ["you", "your", "he", "his", "him", "they", "their", "this member"]
        if any(term in normalized for term in field_terms) and any(term in normalized for term in user_terms + pronoun_terms):
            return True

        if "computer science" in normalized or "esi sba" in normalized:
            return True

        return False

    def extract_requested_team_field(self, query: str) -> str | None:
        normalized = normalize_text(query)
        if "age" in normalized or "old" in normalized or "ages" in normalized:
            return "age"
        if "university" in normalized or "school" in normalized or "from" in normalized:
            return "university"
        if "academic" in normalized or "level" in normalized or "year" in normalized:
            return "academic_level"
        if "field" in normalized or "study" in normalized or "studies" in normalized or "computer science" in normalized:
            return "field_of_study"
        return None

    def _ambiguous_member_question(self, normalized: str) -> TeamAnswer | None:
        if "mohamed" not in normalized or any(term in normalized for term in ["ouail", "essadik", "benbait"]):
            return None
        matches = [member.full_name for member in self._members if "mohamed" in normalize_text(member.full_name).split()]
        if len(matches) > 1:
            return TeamAnswer(
                answer="I found more than one Mohamed in the team. Do you mean Mohamed Ouail or Benbait Mohamed Essadik?",
                matched_member=None,
                matched_alias="mohamed",
            )
        return None

    def answer_team_question(self, query: str, last_member_name: str | None = None) -> TeamAnswer:
        if not self._members:
            return TeamAnswer(
                answer="No team information is available yet. Add the project team details in data/team/team.json.",
                matched_member=None,
                matched_alias=None,
            )

        normalized = normalize_text(query)
        ambiguous = self._ambiguous_member_question(normalized)
        if ambiguous:
            return ambiguous

        match = self.find_member(query)
        if match is None and last_member_name and self._is_pronoun_followup(normalized):
            match = self.find_member_by_full_name(last_member_name)

        wants_list = any(
            term in normalized
            for term in ["who are", "list", "members", "team", "creators", "built you", "made you", "students"]
        )
        requested_field = self.extract_requested_team_field(query)
        wants_age = requested_field == "age"
        wants_level = requested_field == "academic_level"
        wants_university = requested_field == "university"
        wants_field = requested_field == "field_of_study"

        if match:
            member = match.member
            if wants_age and not (wants_level or wants_university or wants_field):
                return self._matched(f"{member.full_name} is {member.age} years old.", match)
            if wants_university and not (wants_age or wants_level or wants_field):
                return self._matched(f"{member.full_name} studies at {member.university_name}.", match)
            if wants_field and not (wants_age or wants_level or wants_university):
                return self._matched(f"{member.full_name} studies {member.field_of_study}.", match)
            if wants_level and not (wants_age or wants_university or wants_field):
                return self._matched(f"{member.full_name} is in {member.academic_level}.", match)
            return self._matched(self._format_member(member), match)

        if wants_age and ("their" in normalized or "ages" in normalized):
            lines = ["Team member ages:"]
            lines.extend(f"- {member.full_name}: {member.age}." for member in self._members)
            return TeamAnswer("\n".join(lines), None, None)

        if wants_field and "who" in normalized:
            matching = [member for member in self._members if member.field_of_study.lower() in normalized or "computer science" in normalized]
            if matching:
                lines = ["Team members who study computer science:"]
                lines.extend(f"- {member.full_name}." for member in matching)
                return TeamAnswer("\n".join(lines), None, None)

        if wants_university:
            universities = sorted({member.university_name for member in self._members})
            return TeamAnswer("Team university: " + "; ".join(universities) + ".", None, None)

        if wants_field:
            fields = sorted({member.field_of_study for member in self._members})
            return TeamAnswer("Team field of study: " + "; ".join(fields) + ".", None, None)

        if wants_list or True:
            return TeamAnswer(self._format_all_members(), None, None)

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        return self.answer_team_question(question).answer

    def _matched(self, answer: str, match: TeamMatch) -> TeamAnswer:
        return TeamAnswer(answer=answer, matched_member=match.member.full_name, matched_alias=match.matched_alias)

    def _format_member(self, member: TeamMember) -> str:
        return (
            f"{member.full_name} is a project team member. Academic level: {member.academic_level}; age: {member.age}; "
            f"university: {member.university_name}; field of study: {member.field_of_study}."
        )

    def _format_all_members(self) -> str:
        lines = ["The project team members are:"]
        for index, member in enumerate(self._members, start=1):
            lines.append(
                f"{index}. {member.full_name} - {member.academic_level}, {member.age}, "
                f"{member.university_name}, {member.field_of_study}."
            )
        return "\n".join(lines)

    def _is_pronoun_followup(self, normalized: str) -> bool:
        return any(term in normalized.split() for term in ["he", "his", "him", "they", "their"]) or "this member" in normalized


team_service = TeamService()


