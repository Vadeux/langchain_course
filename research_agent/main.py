import time

from research_agent.graph import research_agent
from research_agent.schemas import GraphState

if __name__ == "__main__":
    print("Academic Research Agent System")
    print("--------------------------------")
    topic = input("Enter your research topic: ").strip()

    init_state = GraphState(
        topic=topic,
        subtopics=[],
        approval="pending",
        research_results=[],
    )

    start_time = time.time()
    final_state = research_agent.invoke(init_state)

    print("\n" + "=" * 50)
    print(f"final Report Summary for: {topic}")
    print("\n" + "=" * 50)

    if not final_state.get("research_results"):
        print("No results generated.")
    else:
        for ind, result in enumerate(final_state["research_results"]):
            print(f"\nResearch Area {ind+1}: {result.subtopic.title}")
            print(f" Gap: {result.research_gap}")

        print("Full report saved ti report.md file")

    duration = time.time() - start_time
    print(f"\nTotal time taken: {duration:.2f} seconds")
