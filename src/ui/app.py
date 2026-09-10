import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="E",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 50% -8%,
            rgba(120, 119, 198, 0.08),
            transparent 30rem
        ),
        #fcfcfd;
}

.block-container {
    max-width: 820px;
    padding-top: 3.8rem;
    padding-bottom: 8rem;
}

h1 {
    color: #202124 !important;
    letter-spacing: -0.045em !important;
    font-weight: 650 !important;
    line-height: 1.1 !important;
}

h2,
h3,
h4 {
    color: #2f3035 !important;
    letter-spacing: -0.02em !important;
}

p {
    color: #55565f;
}

[data-testid="stCaptionContainer"] {
    color: #93949d;
}

[data-testid="stHorizontalBlock"] {
    gap: 0.8rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #ececf0 !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, 0.72);
    box-shadow:
        0 1px 2px rgba(0, 0, 0, 0.015),
        0 10px 30px rgba(0, 0, 0, 0.02);
}

[data-testid="stChatMessage"] {
    background: transparent;
    padding-top: 0.65rem;
    padding-bottom: 0.65rem;
}

[data-testid="stChatMessageContent"] {
    font-size: 0.98rem;
    line-height: 1.75;
}

[data-testid="stChatInput"] {
    max-width: 790px;
    margin: auto;
}

[data-testid="stChatInput"] > div {
    border: 1px solid #e5e5e9 !important;
    border-radius: 22px !important;
    background: #f7f7f8 !important;
    box-shadow:
        0 1px 2px rgba(0, 0, 0, 0.025),
        0 8px 28px rgba(0, 0, 0, 0.035);
}

[data-testid="stChatInput"] textarea {
    font-size: 0.96rem !important;
}

[data-testid="stMetric"] {
    background: rgba(248, 248, 249, 0.9);
    border: 1px solid #ededf0;
    border-radius: 14px;
    padding: 0.65rem 0.75rem;
}

[data-testid="stMetricLabel"] {
    color: #96969e;
    font-size: 0.72rem;
}

[data-testid="stMetricValue"] {
    color: #37373d;
    font-size: 1rem;
}

[data-testid="stExpander"] {
    border: 1px solid #ececf0 !important;
    border-radius: 14px !important;
    background: rgba(255, 255, 255, 0.8);
}

@media (max-width: 700px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 2.5rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def backend_is_online() -> bool:
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3,
        )
        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


def ask_backend(question: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={
            "question": question,
        },
        timeout=120,
    )

    if response.status_code != 200:
        try:
            detail = response.json().get(
                "detail",
                "Unknown API error.",
            )

        except ValueError:
            detail = response.text

        raise RuntimeError(detail)

    return response.json()


def render_header() -> None:
    online = backend_is_online()

    brand_col, status_col = st.columns([4, 1])

    with brand_col:
        st.markdown("**Enterprise Agentic RAG**")

    with status_col:
        if online:
            st.caption("🟢 V3.2 Online")
        else:
            st.caption("⚪ Backend Offline")

    st.write("")

    st.title("Enterprise Knowledge Assistant")

    st.caption(
        "Search across enterprise knowledge with "
        "source-aware retrieval, agentic reasoning, "
        "and evidence-grounded generation."
    )


def render_welcome() -> None:
    st.write("")
    st.write("")

    st.caption("SUGGESTED QUESTION")

    st.markdown("**What caused the signing mismatch?**")

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(
            border=True,
        ):
            st.markdown("**Multi-document retrieval**")

            st.caption(
                "Searches across multiple enterprise "
                "documents instead of committing to "
                "one source."
            )

    with col2:
        with st.container(
            border=True,
        ):
            st.markdown("**Agentic reasoning**")

            st.caption(
                "Decides when to search, recover "
                "missing evidence, or generate an "
                "answer."
            )

    with col3:
        with st.container(
            border=True,
        ):
            st.markdown("**Evidence-aware generation**")

            st.caption(
                "Preserves useful evidence while "
                "reducing irrelevant retrieval noise."
            )


def render_execution_summary(
    result: dict,
) -> None:
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Latency",
            f"{result.get('latency_seconds', 0):.1f}s",
        )

    with col2:
        st.metric(
            "Reasoning",
            f"{result.get('decision_steps', 0)} steps",
        )

    with col3:
        st.metric(
            "Tool usage",
            f"{result.get('tool_actions', 0)} calls",
        )

    if result.get(
        "evidence_sufficient",
        False,
    ):
        st.caption("✓ Evidence judged sufficient")

    else:
        st.caption(
            "Answer generated from the best available "
            "evidence after bounded recovery."
        )


def render_sources(
    sources: list[dict],
) -> None:
    if not sources:
        return

    document_ids = []

    for source in sources:
        document_id = source.get("document_id")

        if document_id and document_id not in document_ids:
            document_ids.append(document_id)

    title = f"Sources · {len(document_ids)} documents " f"· {len(sources)} chunks"

    with st.expander(
        title,
        expanded=False,
    ):
        for index, source in enumerate(
            sources,
            start=1,
        ):
            document_id = source.get(
                "document_id",
                "Unknown document",
            )

            chunk_index = source.get("chunk_index")

            evidence_source = source.get("evidence_source")

            short_document_id = (
                document_id[:24] + "…" if len(document_id) > 24 else document_id
            )

            st.markdown(f"**{index}. {short_document_id}**")

            metadata = []

            if chunk_index is not None:
                metadata.append(f"Chunk {chunk_index}")

            if evidence_source:
                metadata.append(str(evidence_source))

            if not metadata:
                metadata.append("Retrieved evidence")

            st.caption(" · ".join(metadata))

            if index != len(sources):
                st.divider()


def render_request_details(
    result: dict,
) -> None:
    request_id = result.get("request_id")

    if not request_id:
        return

    with st.expander(
        "Execution details",
        expanded=False,
    ):
        st.caption("Request ID")

        st.code(
            request_id,
            language=None,
        )


if "messages" not in st.session_state:
    st.session_state.messages = []


render_header()


if not st.session_state.messages:
    render_welcome()


for message in st.session_state.messages:
    role = message["role"]

    with st.chat_message(role):
        st.markdown(message["content"])

        if role == "assistant":
            result = message.get("result")

            if result:
                render_execution_summary(result)

                render_sources(
                    result.get(
                        "sources",
                        [],
                    )
                )

                render_request_details(result)


question = st.chat_input("Ask anything across your enterprise knowledge...")


if question:
    clean_question = question.strip()

    if clean_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": clean_question,
            }
        )

        with st.chat_message("user"):
            st.markdown(clean_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching enterprise knowledge..."):
                try:
                    result = ask_backend(clean_question)

                    answer = result.get(
                        "answer",
                        "No answer was returned.",
                    )

                    st.markdown(answer)

                    render_execution_summary(result)

                    render_sources(
                        result.get(
                            "sources",
                            [],
                        )
                    )

                    render_request_details(result)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "result": result,
                        }
                    )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "The backend API is offline. "
                        "Start the FastAPI Docker container "
                        "and try again."
                    )

                except requests.exceptions.Timeout:
                    st.error(
                        "The request timed out before " "the RAG pipeline completed."
                    )

                except requests.exceptions.RequestException as error:
                    st.error(f"Backend request failed: {error}")

                except RuntimeError as error:
                    st.error(f"RAG service error: {error}")
