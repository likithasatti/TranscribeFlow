# from transformers import pipeline

# summarizer = pipeline("summarization", model="t5-small")

# def summarize_text(text):
#     summary = summarizer(text, max_length=60, min_length=25, do_sample=False)
#     return summary[0]['summary_text'] 

# from transformers import pipeline
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# def summarize_text(text):
#     summary = summarizer(
#         text,
#         max_length=100,
#         min_length=40,
#         do_sample=False
#     )
#     return summary[0]['summary_text']

# summarizer = pipeline("summarization", model="t5-small")

# def summarize_text(text):

#     # Add prefix (important for T5)
#     text = "summarize: " + text

#     summary = summarizer(
#         text,
#         max_length=120,
#         min_length=40,
#         do_sample=False
#     )

#     return summary[0]['summary_text']
from transformers import pipeline

# Load FLAN-T5 Small model (Good for 8GB RAM)
summarizer = pipeline(
    "summarization",
    model="google/flan-t5-small"
)

def summarize_text(text):
    """
    Takes transcript text as input
    Returns summarized text
    """

    # Handle empty text
    if not text or len(text.strip()) == 0:
        return "No text available to summarize."

    # Generate summary
    summary = summarizer(
        text,
        max_length=120,     # Maximum summary length
        min_length=40,      # Minimum summary length
        do_sample=False     # Deterministic output
    )

    return summary[0]['summary_text']