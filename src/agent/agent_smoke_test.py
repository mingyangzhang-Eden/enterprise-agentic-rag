from advanced_rag import AdvancedRAG

from agent.agent import RetrievalAgent
from agent.decision_maker import DecisionMaker
from agent.tools import AgentTools
from generation import Generator

QUESTION = (
    "What storage setup and TTL were used to keep long stop-and-go "
    "chat sessions cheap without replaying the whole conversation history?"
)


def main() -> None:
    print("Loading V1 components...")

    rag = AdvancedRAG()

    chunks = rag.candidate_generator.dense_retriever.chunks

    tools = AgentTools(
        candidate_generator=rag.candidate_generator,
        reranker=rag.reranker,
        chunks=chunks,
        evidence_top_k=5,
    )

    decision_maker = DecisionMaker(
        max_steps=4,
    )

    generator = Generator()

    agent = RetrievalAgent(
        tools=tools,
        decision_maker=decision_maker,
        generator=generator,
        max_steps=4,
    )

    print("\nStarting Agent...")
    print(f"Question: {QUESTION}")

    state = agent.run(
        question=QUESTION,
    )

    print("\n" + "=" * 80)
    print("FINAL AGENT STATE")
    print("=" * 80)

    summary = state.summary()

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("ACTION HISTORY")
    print("=" * 80)

    for index, action in enumerate(
        state.action_history,
        start=1,
    ):
        print(f"{index}. {action}")

    print("\n" + "=" * 80)
    print("DECISION / JUDGMENT HISTORY")
    print("=" * 80)

    for judgment in state.judge_history:
        print(f"Step: {judgment['step']}")

        print(f"Strategy: {judgment['strategy']}")

        print(f"Sufficient: {judgment['sufficient']}")

        print(f"Reason: {judgment['reason']}")

        print(f"Missing: {judgment['missing_information']}")

        print("-" * 80)

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    if state.final_answer:
        print(state.final_answer)
    else:
        print("No final answer was generated.")


if __name__ == "__main__":
    main()
