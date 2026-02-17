import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class Subtopic(BaseModel):
    """Description of the subtopic."""

    title: str = Field(..., description="Technical title of research subtopic")
    description: str = Field(
        ..., description="1-2 sentence description of the subtopic "
    )


class SubtopicList(BaseModel):
    """List of subtopics for a given topic."""

    subtopics: list[Subtopic]


class ResearchPaper(BaseModel):
    """Represents a research paper."""

    title: str = Field(
        default="Untitled paper", description="Title of the research paper"
    )
    authors: str = Field(
        default="Unknown author",
        description="Comma-separated author names of the paper",
    )
    abstract: str = Field(default="", description="Abstract content")
    url: str = Field(default="#", description="ArXiv URL")
    published: str = Field(
        default="Unknown date", description="Publication date of the paper"
    )


class ResearchResult(BaseModel):
    """Represents the result of a research query."""

    subtopic: Subtopic
    papers: list[ResearchPaper]
    research_gap: str


class GraphState(TypedDict):
    """Represents the state of the graph."""

    topic: str
    subtopics: list[Subtopic]
    approval: Literal["pending", "approved", "rejected"]
    research_results: Annotated[list[ResearchResult], operator.add]
