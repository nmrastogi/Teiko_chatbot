import os
import sqlite3
import subprocess
import sys

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "teiko.db")
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a clinical data analyst assistant for Bob Loblaw's immunotherapy trial at Loblaw Bio.

## Database schema (teiko.db — SQLite)

```sql
subjects(subject_id, project, condition, age, sex, treatment, response)
samples(sample_id, subject_id, sample_type, time_from_treatment_start)
cell_counts(id, sample_id, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte)
```

- `subjects.response`: "yes" = responder, "no" = non-responder
- `subjects.condition`: e.g. "melanoma", "carcinoma"
- `subjects.treatment`: e.g. "miraclib"
- `samples.sample_type`: e.g. "PBMC"
- `samples.time_from_treatment_start`: integer days (0 = baseline)
- Cell count columns: b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte

## Instructions
- Use the `query_database` tool to answer every data question — do not guess or invent numbers.
- For frequency/percentage questions: total_count = b_cell + cd8_t_cell + cd4_t_cell + nk_cell + monocyte per sample; percentage = (population_count / total_count) * 100.
- For statistical comparisons, query the data then describe what you observe (medians, ranges). If asked for a formal test, note that you can describe the numerical pattern but cannot run scipy from here.
- Always show SQL you used when the user might want to verify it.
- Be concise and precise. Cite actual numbers from query results.
"""

TOOLS = [
    {
        "name": "query_database",
        "description": "Execute a read-only SELECT query against teiko.db and return results as a table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SQLite SELECT statement.",
                }
            },
            "required": ["sql"],
        },
    }
]


@st.cache_resource(show_spinner="Setting up database...")
def bootstrap() -> None:
    if not os.path.exists(DB_PATH):
        subprocess.run([sys.executable, "load_data.py"], check=True)


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = None
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not set. Add it to .env or Streamlit secrets.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def run_query(sql: str) -> pd.DataFrame | str:
    sql = sql.strip()
    if not sql.lower().lstrip().startswith("select"):
        return "Only SELECT queries are allowed."
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception as e:
        return f"Query error: {e}"


def chat(client: anthropic.Anthropic, messages: list[dict]) -> str:
    api_messages = messages.copy()

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=api_messages,
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "query_database":
                    result = run_query(block.input["sql"])
                    content = result.to_string(index=False) if isinstance(result, pd.DataFrame) else result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    })
            api_messages = api_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]
        else:
            return "".join(
                block.text for block in response.content if hasattr(block, "text")
            )


def main() -> None:
    st.set_page_config(
        page_title="Teiko Clinical Trial Assistant",
        page_icon="🧬",
        layout="wide",
    )

    bootstrap()
    client = get_client()

    with st.sidebar:
        st.title("🧬 Teiko Clinical Trial")
        st.markdown("**Drug:** miraclib  \n**Indications:** melanoma, carcinoma")
        st.markdown("---")
        st.markdown("**Example questions:**")
        st.markdown("- What is the frequency of each cell type in sample00000?")
        st.markdown("- Compare CD4 T cell frequencies between responders and non-responders in melanoma PBMC samples")
        st.markdown("- How many melanoma PBMC subjects are at baseline on miraclib?")
        st.markdown("- What is the average B cell count for male melanoma responders at time=0?")
        st.markdown("---")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    st.title("Clinical Trial Data Assistant")
    st.caption("Ask anything about the immune cell data — I query the database live.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        api_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        with st.chat_message("assistant"):
            with st.spinner("Querying..."):
                reply = chat(client, api_msgs)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
