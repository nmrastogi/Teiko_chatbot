import io
import os
import sqlite3
import subprocess
import sys

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from scipy import stats

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
- Use `query_database` to fetch data, `generate_plot` to visualize it, and `compute_statistics` for formal tests.
- For frequency/percentage: total_count = b_cell + cd8_t_cell + cd4_t_cell + nk_cell + monocyte; percentage = (count / total_count) * 100.
- When generating plots, first query the data to understand its shape, then call generate_plot with the same SQL.
- Always be concise. Cite actual numbers from results.
"""

TOOLS = [
    {
        "name": "query_database",
        "description": "Execute a read-only SELECT query against teiko.db and return results as text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A valid SQLite SELECT statement."}
            },
            "required": ["sql"],
        },
    },
    {
        "name": "generate_plot",
        "description": (
            "Run a SQL query and render the results as a chart displayed to the user. "
            "Supported chart types: boxplot, bar, histogram, scatter, line."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql":        {"type": "string", "description": "SELECT query returning the data to plot."},
                "chart_type": {"type": "string", "enum": ["boxplot", "bar", "histogram", "scatter", "line"]},
                "x":          {"type": "string", "description": "Column name for the x-axis."},
                "y":          {"type": "string", "description": "Column name for the y-axis."},
                "hue":        {"type": "string", "description": "Column name for color grouping (optional)."},
                "title":      {"type": "string", "description": "Chart title."},
            },
            "required": ["sql", "chart_type", "x", "y", "title"],
        },
    },
    {
        "name": "compute_statistics",
        "description": (
            "Run a formal statistical test comparing two groups. "
            "Each SQL query must return a single numeric column of values for that group."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql_group1":   {"type": "string", "description": "SELECT query returning numeric values for group 1."},
                "sql_group2":   {"type": "string", "description": "SELECT query returning numeric values for group 2."},
                "test":         {"type": "string", "enum": ["mannwhitneyu", "ttest_ind"], "description": "Statistical test to run."},
                "group1_label": {"type": "string", "description": "Human-readable label for group 1."},
                "group2_label": {"type": "string", "description": "Human-readable label for group 2."},
            },
            "required": ["sql_group1", "sql_group2", "test", "group1_label", "group2_label"],
        },
    },
]


# ── Bootstrap ─────────────────────────────────────────────────────────────────

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


# ── Tool implementations ──────────────────────────────────────────────────────

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


def run_generate_plot(inp: dict) -> tuple[str, bytes | None]:
    df = run_query(inp["sql"])
    if isinstance(df, str):
        return df, None

    chart_type = inp["chart_type"]
    x, y, title = inp["x"], inp["y"], inp["title"]
    hue = inp.get("hue")

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if chart_type == "boxplot":
            sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
        elif chart_type == "bar":
            sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
        elif chart_type == "histogram":
            sns.histplot(data=df, x=x, hue=hue, ax=ax)
        elif chart_type == "scatter":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
        elif chart_type == "line":
            sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)

        ax.set_title(title)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        return f"Chart '{title}' generated successfully.", buf.read()
    except Exception as e:
        return f"Plot error: {e}", None
    finally:
        plt.close(fig)


def run_compute_statistics(inp: dict) -> str:
    g1 = run_query(inp["sql_group1"])
    g2 = run_query(inp["sql_group2"])

    if isinstance(g1, str):
        return f"Group 1 query error: {g1}"
    if isinstance(g2, str):
        return f"Group 2 query error: {g2}"

    try:
        vals1 = g1.iloc[:, 0].dropna().astype(float)
        vals2 = g2.iloc[:, 0].dropna().astype(float)
    except Exception as e:
        return f"Could not extract numeric values: {e}"

    label1, label2 = inp["group1_label"], inp["group2_label"]
    test = inp["test"]

    try:
        if test == "mannwhitneyu":
            stat, p = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            test_name = "Mann-Whitney U"
        else:
            stat, p = stats.ttest_ind(vals1, vals2)
            test_name = "Independent t-test"
    except Exception as e:
        return f"Test error: {e}"

    sig = "significant" if p < 0.05 else "not significant"
    return (
        f"{test_name} results:\n"
        f"  {label1}: n={len(vals1)}, median={vals1.median():.4f}, mean={vals1.mean():.4f}\n"
        f"  {label2}: n={len(vals2)}, median={vals2.median():.4f}, mean={vals2.mean():.4f}\n"
        f"  statistic={stat:.4f}, p-value={p:.6f} → {sig} at α=0.05"
    )


# ── Chat loop ─────────────────────────────────────────────────────────────────

def chat(client: anthropic.Anthropic, messages: list[dict]) -> tuple[str, list[bytes]]:
    figures: list[bytes] = []
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
                if block.type != "tool_use":
                    continue

                if block.name == "query_database":
                    result = run_query(block.input["sql"])
                    content = result.to_string(index=False) if isinstance(result, pd.DataFrame) else result

                elif block.name == "generate_plot":
                    content, fig_bytes = run_generate_plot(block.input)
                    if fig_bytes:
                        figures.append(fig_bytes)

                elif block.name == "compute_statistics":
                    content = run_compute_statistics(block.input)

                else:
                    content = "Unknown tool."

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
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return text, figures


# ── Streamlit UI ──────────────────────────────────────────────────────────────

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
        st.markdown("- Plot CD4 T cell % for responders vs non-responders in melanoma PBMC")
        st.markdown("- Run a Mann-Whitney test on B cell frequencies between responders and non-responders")
        st.markdown("- Show a bar chart of average monocyte % by project")
        st.markdown("---")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    st.title("Clinical Trial Data Assistant")
    st.caption("Ask anything about the immune cell data — I can query the database, run stats, and generate charts.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for fig_bytes in msg.get("figures", []):
                st.image(fig_bytes)

    if prompt := st.chat_input("Ask a question about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "figures": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        api_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, figures = chat(client, api_msgs)
            st.markdown(reply)
            for fig_bytes in figures:
                st.image(fig_bytes)

        st.session_state.messages.append({"role": "assistant", "content": reply, "figures": figures})


if __name__ == "__main__":
    main()
