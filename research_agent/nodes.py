import uuid
from typing import Any

import arxiv
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from research_agent.const import ApprovalState
from research_agent.schemas import (
    GraphState,
    ResearchGap,
    ResearchPaper,
    ResearchResult,
    Subtopic,
    SubtopicList,
)

load_dotenv()

llm = ChatOpenAI(temperature=0)


def fetch_arxiv_papers(query: str, max_results: int = 3) -> list[ResearchPaper]:
    """Fetch papers from arXiv based on a title."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in client.results(search):
        authours = ", ".join(author.name for author in result.authors)
        papers.append(
            ResearchPaper(
                title=result.title,
                authors=authours,
                abstract=result.summary,
                url=result.entry_id,
                published=str(result.published.date()),
            )
        )
    return papers


def generate_subtopics(state: GraphState) -> GraphState:
    """Node function for generating subtopics."""
    topic = state["topic"]
    structured_llm_generator = llm.with_structured_output(SubtopicList)
    print(f"---GENERATE SUBTOPICS FOR USER TOPIC: {topic}---")
    try:
        prompt = PromptTemplate(
            template="""
            You are a senior AI researcher specializing in LLM architectures.
            Given topic: {topic}.
            Provide exactly 3 narrow, technical subtopics.\n
            """,
            input_variables=["topic"],
        )  # TODO: move to prompts file
        chain = prompt | structured_llm_generator
        response: SubtopicList = chain.invoke({"topic": topic})
        return {
            "subtopics": response.subtopics,
            "approval": ApprovalState.PENDING.value,
        }
    except Exception as e:
        print(f"Subtopic generation error: {e}. User topic input: {topic}.")
        return {
            "subtopics": [
                Subtopic(
                    title="Technical Subtopic 1", description="Fallback description"
                ),
                Subtopic(
                    title="Technical Subtopic 2", description="Fallback description"
                ),
                Subtopic(
                    title="Technical Subtopic 3", description="Fallback description"
                ),
            ],
            "approval": ApprovalState.PENDING.value,
        }


def ask_approval(state: GraphState) -> GraphState:
    try:
        print("Generated research subtopics:")
        for i, subtopic in enumerate(state["subtopics"]):
            print(f"[{i+1}] {subtopic.title}: {subtopic.description}")

        decision = input("Approve subtopics? (yes/no/edit): ").strip().lower()

        if decision.startswith("y"):
            return {"approval": ApprovalState.APPROVED.value}
        elif decision.startswith("e"):
            print("Enter revised topics (one per line, empty line for finish):")
            new_subtopics = []
            for n_sub in range(3):
                title = input(f"Subtopic {n_sub+1} title: ").strip()
                description = input(f"Subtopic {n_sub+1} description: ").strip()
                if title and description:
                    new_subtopics.append(Subtopic(title=title, description=description))
            return {
                "subtopics": new_subtopics,
                "approval": ApprovalState.APPROVED.value,
            }
        else:
            # TODO: implement logic to Truly replace rejected subtopics
            return {"approval": ApprovalState.REJECTED.value}

    except Exception as e:
        print(f"Approval error: {str(e)}")
        return {"approval": ApprovalState.APPROVED.value}


def analyze_papers(papers: list[ResearchPaper], subtopic_title: str) -> str:
    structured_llm_gap = llm.with_structured_output(ResearchGap)
    try:
        papers_info = "\n\n".join(
            [
                f"Paper {ind+1}: {paper.title}\nURL: {paper.url}\nAbstract: {paper.abstract[:500]}...\nPublished: {paper.published}"
                for ind, paper in enumerate(papers)
            ]
        )
        gap_prompt = PromptTemplate(
            template="""
                Analyze these three research papers on {subtopic}:\n\n
                {papers_info}\n\n
                Identify ONE specific research gep considering:\n
                - What limitations do these papers share?\n
                - What opportunities do they collectively miss?\n
                - What technical challenges remain unaddressed?\n
                Provide a concise gap description based on all three papers.\n
                Include references to specific papers where appropriate.\n
            """,
            input_variables=["subtopic", "papers_info"],
        )

        gap_chain = gap_prompt | structured_llm_gap
        response = gap_chain.invoke(
            {"subtopic": subtopic_title, "papers_info": papers_info}
        )
        return response.gap

    except Exception as e:
        print(f"Gap analysis error: {str(e)}")
        return "Error in gap analysis"


def conduct_research(state: GraphState) -> GraphState:
    try:
        research_results: list[ResearchResult] = []

        for subtopic in state["subtopics"]:
            print(f"--- RESEARCHING: {subtopic.title}... ---")

            try:
                papers = fetch_arxiv_papers(subtopic.title, max_results=3)
                print(f"Found {len(papers)} papers for {subtopic.title}.")
            except Exception as e:
                print(f"Paper retrieval error: {str(e)}")
                papers = [ResearchPaper(title="Error: Paper not found")]

            gap = analyze_papers(papers, subtopic.title)  # TODO: move to separate node?

            research_results.append(
                ResearchResult(subtopic=subtopic, papers=papers, research_gap=gap)
            )
            return {"research_results": research_results}
    except Exception as e:
        print(f"Critical research error: {str(e)}")
        return {"research_results": []}


def compile_report(state: GraphState) -> dict[str, Any]:
    try:
        if not state.get("research_results"):
            return {"report": "# Research Report\n\nNo results generated"}

        report_id = str(uuid.uuid4())
        report = f"# Research Report {report_id}: {state['topic']}\n\n"

        for research_result in state["research_results"]:
            report += f"## {research_result.subtopic.title}\n"
            report += f"*{research_result.subtopic.description}*\n\n"
            report += "### Key Papers\n"

            for ind, paper in enumerate(research_result.papers):
                report += f"#### Paper {ind+1}: {paper.title}\n"
                report += f"- URL: {paper.url}\n"
                report += f"- Authors: {paper.authors}\n"
                report += f"- Abstract: {paper.abstract[:500]}...\n"
                report += f"- Published: {paper.published}\n\n"

            report += f"### Identified Research Gap:\n"
            report += f"{research_result.research_gap}\n\n"

        with open(f"report_{report_id}.md", "w") as f:
            f.write(report)

        print(f"Research report saved as report_{report_id}.md")
        return {"report": report}
    except Exception as e:
        print(f"Report generation error: {str(e)}")
        return {"report": "# Error: Report generation failed."}
