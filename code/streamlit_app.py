import os
import re
import streamlit as st
from io import StringIO
from st_keyup import st_keyup
from annotated_text import annotated_text
from autocomplete import (
    tokenize_data,
    get_words_with_nplus_frequency,
    replace_oov_words_by_unk,
    compute_n_grams,
    get_suggestions,
    get_interpolated_suggestions,
)

st.set_page_config(page_title="Autocomplete", layout="centered")


def read_file():
    file = st.session_state["file_uploader"]
    if file:
        stringio = StringIO(file.getvalue().decode("utf-8"))
        data = stringio.read()
        tokenized_data = tokenize_data(data)
        vocabulary = get_words_with_nplus_frequency(tokenized_data, min_freq=2)
        train_data_replaced = replace_oov_words_by_unk(tokenized_data, vocabulary)
        n_gram_counts_list = compute_n_grams(train_data_replaced)
    else:
        vocabulary, n_gram_counts_list = get_model()
    st.session_state["vocabulary"] = vocabulary
    st.session_state["n_gram_counts_list"] = n_gram_counts_list
    st.session_state["n_gram_counts_list_size"] = len(st.session_state["n_gram_counts_list"])


@st.cache_resource
def get_model():
    with open(os.path.join("data", "en_US.twitter.txt"), "r", encoding="utf-8") as f:
        data = f.read()

    tokenized_data = tokenize_data(data)
    vocabulary = get_words_with_nplus_frequency(tokenized_data, min_freq=2)
    train_data_replaced = replace_oov_words_by_unk(tokenized_data, vocabulary)
    n_gram_counts_list = compute_n_grams(train_data_replaced)

    return vocabulary, n_gram_counts_list


def autocomplete(text, vocabulary, n_gram_counts_list):
    if len(text.strip()) > 0:
        typing = text[-1] not in ["\t", " "]

        tokenized_sentences = tokenize_data(text)
        tokenized_sentence = tokenized_sentences[-1]  # only work with the last sentence

        start_with = None
        if typing:
            start_with = tokenized_sentence[-1]
            tokenized_sentence = tokenized_sentence[:-1]

        tokenized_sentence = replace_oov_words_by_unk([tokenized_sentence], vocabulary)[0]

        suggestions = get_suggestions(tokenized_sentence, n_gram_counts_list, vocabulary, k=1.0, start_with=start_with, top_k=5)

        interpolated_suggestions = get_interpolated_suggestions(
            tokenized_sentence, n_gram_counts_list, [0.1, 0.2, 0.3, 0.4], vocabulary, k=1.0, start_with=start_with, top_k=5
        )

        return suggestions, interpolated_suggestions
    else:
        return None, None


def display_annotated_suggestion(suggestions):
    def escape_markdown(text: str) -> str:
        escape_chars = r"\_*[]()~`>#+-=|{}.!"
        return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

    if len(suggestions) > 0:
        att = []
        for word, prob in suggestions[:-1]:
            att.append((escape_markdown(word), f"{prob:.3f}"))
            att.append(" ")
        att.append((escape_markdown(suggestions[-1][0]), f"{suggestions[-1][1]:.3f}"))

        annotated_text(*att)


def main():
    if len(st.session_state) == 0:
        st.session_state["input"] = ""

        # default model
        vocabulary, n_gram_counts_list = get_model()
        st.session_state["vocabulary"] = vocabulary
        st.session_state["n_gram_counts_list"] = n_gram_counts_list
        st.session_state["n_gram_counts_list_size"] = len(st.session_state["n_gram_counts_list"])

    st.title("N-gram Autocomplete")

    st.subheader("File upload")
    st.markdown("Upload a file to calculate n-grams and probabilities. If no file is loaded, a default data will be used.")
    st.file_uploader("Upload corpus file", type=["txt"], key="file_uploader", on_change=read_file, label_visibility="collapsed")

    st.subheader("Input")
    st_keyup("Input", key="input", label_visibility="collapsed")

    suggestions, interpolated_suggestions = autocomplete(
        st.session_state["input"], st.session_state["vocabulary"], st.session_state["n_gram_counts_list"]
    )

    st.markdown("---")

    if suggestions is not None:
        for i in range(5 - 1):
            st.subheader(f"Suggested autocompletion Ngram where N = {i+1}")
            display_annotated_suggestion(suggestions[i])

    st.markdown("---")

    if interpolated_suggestions is not None:
        st.subheader("Suggested interpolated autocompletion")
        display_annotated_suggestion(interpolated_suggestions)

    st.markdown("---")

    st.write("Source code [here](https://github.com/viniciusarruda/autocomplete).")


if __name__ == "__main__":
    main()
