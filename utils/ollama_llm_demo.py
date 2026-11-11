"""Supporting API and display helpers for the introductory Ollama notebook.

The notebook keeps the prompt and Ollama generation settings visible. This
module handles HTTP details, response parsing, and chart/table formatting.
"""

import json
import math
import re
import urllib.error
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def _request(host, path, payload=None, timeout=15):
    """Send a JSON request to a local Ollama server."""
    url = host.rstrip("/") + path
    if payload is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def list_models(host):
    """Return names of models available to the Ollama server."""
    return [item["name"] for item in _request(host, "/api/tags").get("models", [])]


def generate(host, payload):
    """Request scored tokens in one complete response for the notebook demo."""
    # These settings are fixed by the exercise, not exposed as learner choices.
    payload = {**payload, "stream": False, "logprobs": True}
    try:
        return _request(host, "/api/generate", payload=payload, timeout=300)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "Ollama rejected this generation request. Check that the server version "
            "supports logprobs/top_logprobs, then update the Ollama module if needed.\n"
            + detail
        ) from error


def show_tokenization_comparison(text, encoding_name="cl100k_base"):
    """Compare a simple word tokenizer with a real tiktoken BPE encoding.

    Word-token IDs are assigned in order of first appearance in this example;
    BPE token IDs come from the selected tiktoken encoding vocabulary.
    """
    try:
        import tiktoken
    except ImportError as error:
        raise ImportError(
            "This exercise uses tiktoken for real BPE tokenization. Install it in "
            "the notebook environment with `python -m pip install tiktoken`."
        ) from error

    # This transparent regex tokenizer groups words and separates punctuation.
    word_pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    word_vocabulary = {}
    word_rows = []
    for piece in word_pieces:
        if piece not in word_vocabulary:
            word_vocabulary[piece] = len(word_vocabulary)
        word_rows.append({"Token": repr(piece), "Token ID": word_vocabulary[piece]})

    encoding = tiktoken.get_encoding(encoding_name)
    bpe_ids = encoding.encode(text)
    bpe_rows = [
        {
            "Token": repr(encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")),
            "Token ID": token_id,
        }
        for token_id in bpe_ids
    ]

    print(f"Input: {text!r}")
    word_table = pd.DataFrame(word_rows).drop_duplicates().reset_index(drop=True)
    bpe_table = pd.DataFrame(bpe_rows).drop_duplicates().reset_index(drop=True)

    print(f"Word-level token/ID pairs — {len(word_table)} unique (IDs assigned for this example):")
    display(word_table.style.hide(axis="index"))
    print(f"BPE token/ID pairs using {encoding_name} — {len(bpe_table)} unique:")
    display(bpe_table.style.hide(axis="index"))
    return {"word_level": word_table, "bpe": bpe_table}

def show_next_token_scores(result):
    """Display the top next-token candidates and their probabilities."""
    logprobs = result.get("logprobs") or []
    if not logprobs:
        print("This Ollama build did not return log probabilities for this model.")
        return None

    first_step = logprobs[0]
    alternatives = first_step.get("top_logprobs") or []
    if not alternatives:
        print("No top-token alternatives were returned; check server/model logprobs support.")
        return None

    selected = first_step.get("token", result.get("response", ""))
    rows = []
    for item in alternatives:
        token = item.get("token", "")
        logprob = float(item["logprob"])
        rows.append({
            "Token": repr(token),
            "Probability (%)": 100 * math.exp(logprob),
            "Log probability": logprob,
            "Selected": token == selected,
        })

    table = pd.DataFrame(rows).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
    display(table.style.format({"Probability (%)": "{:.2f}", "Log probability": "{:.3f}"}).hide(axis="index"))

    colors = ["#E07A5F" if value else "#81A9C3" for value in table["Selected"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(table["Token"], table["Probability (%)"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Model probability for next token (%)")
    ax.set_title("Top next-token candidates for the supplied prefix")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    plt.show()
    return table
