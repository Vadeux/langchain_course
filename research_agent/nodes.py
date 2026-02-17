import re

import arxiv
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from research_agent.schemas import (
    GraphState,
    ResearchPaper,
    ResearchResult,
    Subtopic,
    SubtopicList,
)

load_dotenv()

llm = ChatOpenAI(temperature=0)


def fetch_arxiv_papers(query: str, max_results: int = 3) -> list[ResearchPaper]:
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
    structured_llm_generator = llm.with_structured_output(SubtopicList)
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
        out: SubtopicList = chain.invoke({"topic": state["topic"]})
        return {
            "subtopics": out.subtopics,
            "approval": "pending",
        }  # TODO: use enum in approval
    except Exception as e:
        print(f"Subtopic generation error: {e}")
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
            "approval": "pending",  # TODO: use enum in approval
        }


def ask_approval(state: GraphState) -> GraphState:
    try:
        print("Generated research subtopics:")
        for i, subtopic in enumerate(state["subtopics"]):
            print(f"[{i+1}] {subtopic.title}: {subtopic.description}")

        decision = input("Approve subtopics? (yes/no/edit): ").strip().lower()

        if decision.startswith("y"):
            return {"approval": "approved"}  # TODO: use enum in approval
        elif decision.startswith("e"):
            pass  # TODO: implement edit option (enter subtopics manually)
        else:
            return {"approval": "rejected"}  # TODO: use enum in approval

    except Exception as e:
        print(f"Approval error: {str(e)}")
        return {"approval": "approved"}  # TODO: use enum in approval


def analyze_papers(papers: list[ResearchPaper], subtopics: str) -> str:
    try:
        papers_info = "\n\n".join(
            [
                f"Paper {ind+1}: {paper.title}\nURL: {paper.url}\nAbstract: {paper.abstract[:500]}...\nPublished: {paper.published}"
                for ind, paper in enumerate(papers)
            ]
        )
        gap_prompt = PromptTemplate(
            template="""
                Analyze these three research papers on {subtopic}:\n\n\
                {papers_info}\n\n
                Identify ONE specific research gep considering:\n
                - What limitations do these papers share?\n
                - What opportunities do they collectively miss?\n
                - What technical challenges remain unaddressed?\n
                Provide a concise gap description based on all three papers.\n
                Include references to specific papers where appropriate.\n
                Format: [concise description].
            """,
            input_variables=["subtopic", "papers_info"],
        )

        gap_chain = gap_prompt | llm
        gap = gap_chain.invoke({"subtopic": subtopics, "papers_info": papers_info})

        gap = re.sub(
            r"^(Gap:?\s*)+", "", gap.content, flags=re.IGNORECASE
        ).strip()  # TODO: reimplement using structured output
        return gap

    except Exception as e:
        print(f"Gap analysis error: {str(e)}")
        return "Error in gap analysis"


def conduct_research(state: GraphState) -> GraphState:
    try:
        research_results = []

        for subtopic in state["subtopics"]:
            print(f"Researching: {subtopic.title}...")

            try:
                papers = fetch_arxiv_papers(subtopic.title, max_results=3)
                print(f"Found {len(papers)} papers for {subtopic.title}.")
            except Exception as e:
                print(f"Paper retrieval error: {str(e)}")
                papers = [ResearchPaper(title="Error: Paper not found")]

            gap = analyze_papers(papers, subtopic.title)

            research_results.append(
                ResearchResult(subtopic=subtopic, papers=papers, research_gap=gap)
            )
            return {"research_results": research_results}
    except Exception as e:
        print(f"Critical research error: {str(e)}")
        return {"research_results": []}


def compile_report(state: GraphState) -> GraphState:
    try:
        if not state.get("research_results"):
            return {"report": "# Research Report\n\nNo results generated"}

        report = f"# Research Report: {state['topic']}\n\n"

        for result in state["research_results"]:
            report += f"## {result.subtopic.title}\n"
            report += f"*{result.subtopic.description}*\n\n"
            report += "### Key Papers\n"

            for ind, paper in enumerate(result.papers):
                report += f"#### Paper {ind+1}: {paper.title}\n"
                report += f"- URL: {paper.url}\n"
                report += f"- Authors: {paper.authors}\n"
                report += f"- Abstract: {paper.abstract[:500]}...\n"
                report += f"- Published: {paper.published}\n\n"

            report += f"### Identified Research Gap: {result.research_gap}\n"
            report += f"{result.research_gap}\n\n"

        with open("report.md", "w") as f:
            f.write(report)

        print("Research report saved as report.md")
        return {"report": report}
    except Exception as e:
        print(f"Report generation error: {str(e)}")
        return {"report": "# Error: Report generation failed."}
