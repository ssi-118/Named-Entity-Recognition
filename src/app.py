import html
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="NER Entity Extractor",
    page_icon="🔎",
    layout="wide"
)

MODEL_NAME = "soha118/distilbert-conll2003-ner"


LABEL_COLORS = {
    "PER": "#ffdd99",
    "ORG": "#b7e4ff",
    "LOC": "#c8f7c5",
    "MISC": "#e5ccff"
}

LABEL_NAMES = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Miscellaneous"
}

@st.cache_resource
def load_model():
    return pipeline(
        "token-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        aggregation_strategy="simple"
    )

ner_pipeline = load_model()

def build_highlighted_text(text, entities):
    pieces = []
    last_idx = 0

    sorted_entities = sorted(entities, key=lambda item: item["start"])

    for ent in sorted_entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["entity_group"]
        color = LABEL_COLORS.get(label, "#eeeeee")

        pieces.append(html.escape(text[last_idx:start]))

        entity_text = html.escape(text[start:end])
        pieces.append(
            f"<mark style='background:{color}; padding:3px 6px; "
            f"border-radius:4px; margin:0 2px;'>"
            f"{entity_text} <strong>{label}</strong>"
            f"</mark>"
        )

        last_idx = end

    pieces.append(html.escape(text[last_idx:]))

    return "".join(pieces)

st.title("Named Entity Recognition System")

st.write(
    "This app uses a DistilBERT model fine-tuned on the CoNLL-2003 dataset "
    "to identify people, organizations, locations, and miscellaneous named entities."
)

sample_text = (
    "Barack Obama was born in Hawaii. He worked with Microsoft and later visited London "
    "for a technology conference."
)

text = st.text_area(
    "Enter text",
    value=sample_text,
    height=180
)

run_button = st.button("Extract Entities", use_container_width=True)

if run_button:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        entities = ner_pipeline(text)

        st.subheader("Highlighted Text")

        if entities:
            highlighted_html = build_highlighted_text(text, entities)
            st.markdown(highlighted_html, unsafe_allow_html=True)
        else:
            st.info("No named entities found.")

        st.subheader("Detected Entities")

        if entities:
            rows = []

            for ent in entities:
                label = ent["entity_group"]

                rows.append({
                    "Entity": ent["word"],
                    "Label": label,
                    "Meaning": LABEL_NAMES.get(label, label),
                    "Confidence": round(float(ent["score"]), 4),
                    "Start": ent["start"],
                    "End": ent["end"]
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            st.subheader("Entity Counts")
            count_df = df["Label"].value_counts().reset_index()
            count_df.columns = ["Label", "Count"]
            st.bar_chart(count_df, x="Label", y="Count")
        else:
            st.write("No entities to display.")

st.sidebar.title("Project Details")
st.sidebar.write("Model: DistilBERT")
st.sidebar.write("Dataset: CoNLL-2003")
st.sidebar.write("Task: Named Entity Recognition")

st.sidebar.markdown("### Entity Labels")
st.sidebar.write("PER: Person")
st.sidebar.write("ORG: Organization")
st.sidebar.write("LOC: Location")
st.sidebar.write("MISC: Miscellaneous")
