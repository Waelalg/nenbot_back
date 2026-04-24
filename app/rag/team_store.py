import json
from pathlib import Path
from app.config import TEAM_FILE


def load_team_data() -> list[dict]:
    if not Path(TEAM_FILE).exists():
        return []
    with open(TEAM_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def format_team_context() -> str:
    members = load_team_data()
    if not members:
        return "No team data is currently available."

    lines = []
    for idx, member in enumerate(members, start=1):
        lines.append(
            f"Member {idx}: full_name={member.get('full_name', 'N/A')}; "
            f"academic_level={member.get('academic_level', 'N/A')}; "
            f"age={member.get('age', 'N/A')}; "
            f"university_name={member.get('university_name', 'N/A')}; "
            f"field_of_study={member.get('field_of_study', 'N/A')}"
        )
    return "\n".join(lines)
