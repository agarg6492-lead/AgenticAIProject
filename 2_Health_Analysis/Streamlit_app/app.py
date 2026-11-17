import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = "gemini-3.7-flash"

st.set_page_config(page_title="Blood Report Analyzer", page_icon="🩸", layout="wide")
st.title("🩸 Blood Report Analyzer")
st.caption(
    "Extract every test value from a blood report and classify it as HIGH, LOW or NORMAL "
    "using the reference ranges printed in the report."
)


@st.cache_resource
def get_llm(model_name: str):
    return ChatGoogleGenerativeAI(model=model_name)


def build_prompt(blood_report: str) -> str:
    return f"""
You are medical data extraction assistant.

From the blood report below. Extract ALL the test values and classify each one as HIGH, LOW and NORMAL 
based on the reference ranges provided in the report.

Format your response as:
-Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood_Report:
{blood_report}
"""


LINE_RE = re.compile(
    r"^\s*[-*•]?\s*(?P<test>.+?):\s*(?P<value>.+?)\s*\|\s*Status:\s*(?P<status>HIGH|LOW|NORMAL)"
    r"\s*\|\s*Reference:\s*(?P<ref>.+?)\s*$",
    re.IGNORECASE,
)


def parse_results(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        m = LINE_RE.match(line.replace("**", ""))
        if m:
            rows.append(
                {
                    "Test": m.group("test").strip(),
                    "Value": m.group("value").strip(),
                    "Status": m.group("status").upper(),
                    "Reference": m.group("ref").strip(),
                }
            )
    return pd.DataFrame(rows, columns=["Test", "Value", "Status", "Reference"])


def color_status(val: str) -> str:
    colors = {
        "HIGH": "background-color:#fde2e2;color:#b42318;font-weight:600",
        "LOW": "background-color:#fff3cd;color:#8a5a00;font-weight:600",
        "NORMAL": "background-color:#dcfce7;color:#166534;font-weight:600",
    }
    return colors.get(val, "")


with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Gemini model", value=MODEL_NAME)
    st.markdown("Set `GOOGLE_API_KEY` in your `.env` file before running.")
    st.divider()
    st.warning(
        "AI-generated output can contain mistakes. This is not medical advice — "
        "please consult a doctor to interpret your results."
    )

tab_upload, tab_paste = st.tabs(["Upload file", "Paste text"])

blood_report = ""
with tab_upload:
    uploaded = st.file_uploader("Upload blood report (.txt)", type=["txt"])
    if uploaded is not None:
        blood_report = uploaded.read().decode("utf-8", errors="ignore")

with tab_paste:
    pasted = st.text_area("Or paste the report text here", height=250)
    if pasted.strip() and not blood_report:
        blood_report = pasted

if blood_report:
    with st.expander("Report preview", expanded=False):
        st.text(blood_report[:2000] + ("..." if len(blood_report) > 2000 else ""))

if st.button("Analyze report", type="primary", disabled=not blood_report.strip()):
    with st.spinner("Extracting values..."):
        try:
            llm = get_llm(model_name)
            response = llm.invoke(build_prompt(blood_report))
            st.session_state["extracted"] = response.text
        except Exception as e:
            st.session_state.pop("extracted", None)
            st.error(f"Something went wrong: {e}")

extracted = st.session_state.get("extracted")
if extracted:
    df = parse_results(extracted)

    if df.empty:
        st.subheader("Results")
        st.markdown(extracted)
    else:
        counts = df["Status"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total tests", len(df))
        c2.metric("High", int(counts.get("HIGH", 0)))
        c3.metric("Low", int(counts.get("LOW", 0)))
        c4.metric("Normal", int(counts.get("NORMAL", 0)))

        status_filter = st.multiselect(
            "Filter by status",
            ["HIGH", "LOW", "NORMAL"],
            default=["HIGH", "LOW", "NORMAL"],
        )
        filtered = df[df["Status"].isin(status_filter)]

        st.dataframe(
            filtered.style.map(color_status, subset=["Status"]),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="blood_report_results.csv",
            mime="text/csv",
        )

    with st.expander("Raw model output"):
        st.text(extracted)