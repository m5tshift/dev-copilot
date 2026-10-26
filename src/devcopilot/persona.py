"""Who the agents are and what they know how to do, kept in Markdown files.

persona/*.md: identity, values, users and team rules (the OpenClaw idea: the agent's
character is a document people can read and review, not a string in code):
- IDENTITY.md: who the agent is; SOUL.md: values and boundaries for every agent;
- USER.md: who the answers are for; AGENTS.md: the team and the lead's rules.

skills/<name>/SKILL.md: playbooks for typical tasks, in the format of Agent Skills:
YAML front matter (name, description, agents) and Markdown steps. Progressive
disclosure: the system prompt lists only names and descriptions; the agent loads
a playbook with the load_skill tool when its task matches. So ten skills cost ten
lines of prompt, not ten pages.

Skills vs tools vs the system prompt: a tool acts on the world (reads a file, closes
an issue); a skill is knowledge of HOW to do a task with the tools; the system prompt
is what holds always, for every task (role, values, output format).
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool, StructuredTool

from devcopilot.config import PROJECT_ROOT, get_settings

PERSONA_DIR = PROJECT_ROOT / "persona"
SKILLS_DIR = PROJECT_ROOT / "skills"
LOAD_SKILL = "load_skill"
# Pseudo-server name of in-process tools; trusted like our MCP servers (policy.py).
LOCAL_SERVER = "local"

_COMMENT = re.compile(r"<!--.*?-->\s*", re.DOTALL)


@lru_cache
def persona(name: str) -> str:
    """Text of persona/<NAME>.md for the model: without the title and <!-- notes -->."""
    text = (PERSONA_DIR / f"{name.upper()}.md").read_text(encoding="utf-8")
    text = _COMMENT.sub("", text)
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip() + "\n"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str  # when to use it: the only part the model sees up front
    agents: tuple[str, ...]  # who may load it: "single" or specialist names
    body: str
    path: Path


def parse_skill(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        raise ValueError(f"{path}: no YAML front matter")
    meta = yaml.safe_load(match.group(1))
    name = meta.get("name")
    if name != path.parent.name:
        raise ValueError(f"{path}: name {name!r} must match the directory {path.parent.name!r}")
    if not meta.get("description"):
        raise ValueError(f"{path}: description is required: the model chooses skills by it")
    return Skill(name, meta["description"], tuple(meta.get("agents") or ()), match.group(2).strip(), path)


@lru_cache
def all_skills() -> dict[str, Skill]:
    skills = [parse_skill(p) for p in sorted(SKILLS_DIR.glob("*/SKILL.md"))]
    return {s.name: s for s in skills}


def skills_for(agent: str) -> dict[str, Skill]:
    if not get_settings().skills:
        return {}
    return {n: s for n, s in all_skills().items() if agent in s.agents}


def skills_index(agent: str) -> str:
    """The prompt section listing the agent's skills; empty if it has none."""
    skills = skills_for(agent)
    if not skills:
        return ""
    lines = "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
    return (
        "\nSkills: step-by-step playbooks for typical tasks. If your task matches one, call "
        f"{LOAD_SKILL} with its name first, then follow it.\n{lines}\n"
    )


def skill_tool(agent: str) -> BaseTool | None:
    """load_skill for one agent: it can load only its own skills."""
    skills = skills_for(agent)
    if not skills:
        return None

    def load_skill(name: str) -> str:
        skill = skills.get(name)
        if skill is None:
            return f"Error: unknown skill {name!r}. Available: {', '.join(skills)}."
        return skill.body

    return StructuredTool.from_function(
        load_skill,
        name=LOAD_SKILL,
        description="Load the step-by-step playbook of a skill by its name. Names: " + ", ".join(skills) + ".",
        # Read-only and ours: no approval needed (policy.py trusts LOCAL_SERVER).
        metadata={"mcp_server": LOCAL_SERVER, "readOnlyHint": True},
    )
