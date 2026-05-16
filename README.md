# Named Entity Recognition System

A Streamlit web application that identifies named entities in text using a DistilBERT model fine-tuned on the CoNLL-2003 dataset.

The app detects entities such as people, organizations, locations, and miscellaneous named entities, then displays them with highlights, confidence scores, and entity counts.

## Dataset

The model is trained on the CoNLL-2003 Named Entity Recognition dataset.

Entity labels used:

```text
PER  - Person
ORG  - Organization
LOC  - Location
MISC - Miscellaneous
```

## Model

The project uses:

```text
DistilBERT
```

DistilBERT is fine-tuned for token classification. It reads input text and predicts the entity label for each token.

## Features

- Named entity extraction from user text
- Highlighted entity output
- Entity table with confidence scores
- Entity count visualization
- Streamlit-based web interface

## Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- Pandas
- DistilBERT

## Project Structure

```text
Named-Entity-Recognition/
│
├── src/
│   └── app.py
│
├── assets/
│   └── NER.ipynb
│
├── docs/
│   └── NER_IP_OP.docx
│
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

The trained model is loaded from Hugging Face Hub instead of being stored directly in the repository.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run src/app.py
```

## Example Input

```text
Barack Obama was born in Hawaii. He worked with Microsoft and later visited London for a technology conference.
```

## Example Output

```text
Barack Obama - PER
Hawaii - LOC
Microsoft - ORG
London - LOC
```

## Output Display

The app shows:

- Highlighted text
- Detected entity table
- Confidence score for each entity
- Entity count bar chart

## Live Demo

```
https://named-entity-recognition-qqyf9mvuljfir7es4oysey.streamlit.app/
```
