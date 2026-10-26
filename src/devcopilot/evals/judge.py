"""LLM-as-judge: checks the answer against a rubric, criterion by criterion.

Design choices, each against a known failure of LLM judges:
- binary criteria instead of a 1-10 score: a 9B model cannot keep a scale stable,
  and binary verdicts can be compared with human labels (kappa, labels.py);
- evidence before verdict (field order): the model has to find the quote first;
- a reference answer: without it the judge grades fluency, not correctness;
- the judge never sees the trajectory or the checks: it grades only the answer.

A judge is just another model and makes mistakes: trust it only as far as its
agreement with human labels shows (`devcopilot eval calibrate`).
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from devcopilot.agents import prompts
from devcopilot.decide import Decider, JevError, Noul
from devcopilot.evals.dataset import Task
from devcopilot.llm import StructuredOutputError, structured


class CriterionCheck(BaseModel):
    evidence: str = Field(description="Shortest quote from the answer that addresses the criterion, or 'none'.")
    met: bool


class Judgement(BaseModel):
    """Submit your grading. Call this exactly once."""

    criteria: list[CriterionCheck] = Field(description="One entry per rubric criterion, in the rubric's order.")


class OnlineJudgement(BaseModel):
    """Submit your check. Call this exactly once."""

    reason: str = Field(description="One or two sentences.")
    answers_question: bool
    admits_gaps: bool


def wrap_answer(answer: str) -> str:
    return "<answer>\n" + answer.replace("</answer", "&lt;/answer") + "\n</answer>"


def judge_messages(task: Task, answer: str) -> list:
    rubric = "\n".join(f"{i}. {c}" for i, c in enumerate(task.rubric, 1))
    body = (
        f"Question:\n{task.question}\n\nReference answer:\n{task.reference.strip()}\n\n"
        f"Rubric ({len(task.rubric)} criteria):\n{rubric}\n\n{wrap_answer(answer)}"
    )
    return [SystemMessage(prompts.JUDGE), HumanMessage(body)]


async def judge_answer(model: BaseChatModel, task: Task, answer: str) -> dict:
    """{"pass": bool | None, "criteria": [...], "error": str | None, "rubric": hash}."""
    grade = structured(model, Judgement)
    messages = judge_messages(task, answer)
    verdict = {"rubric": task.rubric_hash, "pass": None, "criteria": [], "error": None}
    for _ in range(2):
        try:
            judgement = await grade(messages)
        except StructuredOutputError as e:
            verdict["error"] = str(e)
            return verdict
        except Exception as e:  # Ollama errors: record and move on; `eval judge` retries errors
            verdict["error"] = f"{type(e).__name__}: {e}"
            return verdict
        if len(judgement.criteria) == len(task.rubric):
            verdict["criteria"] = [c.model_dump() for c in judgement.criteria]
            verdict["pass"] = all(c.met for c in judgement.criteria)
            return verdict
        # A list of the wrong length cannot be matched to criteria: ask again.
        messages = [
            *messages,
            HumanMessage(f"You returned {len(judgement.criteria)} entries, the rubric has {len(task.rubric)}."),
        ]
    verdict["error"] = "criteria count mismatch"
    return verdict


async def judge_answer_jev(decider: Decider, task: Task, answer: str) -> dict:
    """The same verdict from Jev: one yes/no question per criterion, all in one request.

    Each criterion gets a probability, so the verdict also has one: the lowest of them
    (all criteria must be met). `eval calibrate` uses it to pick a threshold from labels.
    """
    verdict = {"rubric": task.rubric_hash, "pass": None, "criteria": [], "error": None, "p": None}
    state = {"question": task.question, "reference": task.reference.strip(), "answer": answer}
    questions = {f"c{i}": Noul(prompts.JEV_JUDGE.format(criterion=c)) for i, c in enumerate(task.rubric, 1)}
    try:
        answers = await decider.decide(state, questions)
    except JevError as e:
        verdict["error"] = str(e)
        return verdict
    ordered = [answers[f"c{i}"] for i in range(1, len(task.rubric) + 1)]
    verdict["criteria"] = [{"evidence": f"p = {a.p:.2f}", "met": bool(a.value), "p": a.p} for a in ordered]
    verdict["pass"] = all(a.value for a in ordered)
    verdict["p"] = min(a.p for a in ordered)
    return verdict


async def judge_online(model: BaseChatModel, question: str, answer: str) -> OnlineJudgement:
    check = structured(model, OnlineJudgement)
    body = f"Question:\n{question}\n\n{wrap_answer(answer)}"
    return await check([SystemMessage(prompts.ONLINE_JUDGE), HumanMessage(body)])
