import nltk


def filter_probabilities(
    probabilities: list[tuple[str, float]],
    start_with: str | None = None,
    unknown_token: str | None = "<unk>",
    end_token: str | None = "<e>",
    threshold: float | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    filtered_probabilities = []
    for word, prob in probabilities:
        if unknown_token is not None and word == unknown_token:
            continue

        if end_token is not None and word == end_token:
            continue

        if threshold is not None and prob < threshold:
            continue

        if start_with is not None and (not word.startswith(start_with) or len(word) == len(start_with)):
            continue

        filtered_probabilities.append((word, prob))

    filtered_probabilities.sort(key=lambda t: t[1], reverse=True)

    if top_k is not None:
        filtered_probabilities = filtered_probabilities[:top_k]

    return filtered_probabilities


def tokenize_data(data):
    sentences = data.split("\n")
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]

    tokenized_sentences = [nltk.word_tokenize(s.lower()) for s in sentences]

    return tokenized_sentences


def get_words_with_nplus_frequency(tokenized_sentences, min_freq):
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] = word_counts.get(token, 0) + 1

    closed_vocab = [word for word, cnt in word_counts.items() if cnt >= min_freq]

    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace words not in the given vocabulary with '<unk>' token.
    """
    vocabulary = set(vocabulary)

    replaced_tokenized_sentences = []
    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)

        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences


def count_n_grams(data, n, start_token="<s>", end_token="<e>"):
    n_grams = {}

    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]

        sentence = tuple(sentence)

        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i : i + n]

            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1

    return n_grams


def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    previous_n_gram = tuple(previous_n_gram)

    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + vocabulary_size * k

    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    numerator = n_plus1_gram_count + k

    probability = numerator / denominator

    return probability


def estimate_probabilities(
    previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token="<e>", unknown_token="<unk>", k=1.0
):
    previous_n_gram = tuple(previous_n_gram)

    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + [end_token, unknown_token]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    for word in vocabulary:
        probabilities[word] = estimate_probability(
            word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k
        )

    return probabilities


def get_interpolated_suggestions(
    previous_tokens,
    n_gram_counts_list,
    lambdas,
    vocabulary,
    k=1.0,
    start_with=None,
    top_k=5,
):
    # get ngram probabilities
    n_gram_probabilities = []
    for i in range(len(lambdas)):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i + 1]

        # get previous n-gram
        n = len(list(n_gram_counts.keys())[0])
        s_previous_tokens = n * ["<s>"] + previous_tokens
        previous_n_gram = s_previous_tokens[-n:]

        # estimate probabilities
        probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)

        n_gram_probabilities.append(probabilities)

    # interpolate
    probabilities = {}
    for i, (_, probs) in enumerate(zip(lambdas, n_gram_probabilities)):
        for word, prob in probs.items():
            if word not in probabilities:
                probabilities[word] = [0.0] * len(lambdas)

            probabilities[word][i] = prob

    for word, probs in probabilities.items():
        probabilities[word] = sum([prob * lambd for prob, lambd in zip(probs, lambdas)])

    # filter
    probabilities = list(probabilities.items())
    probabilities = filter_probabilities(probabilities, start_with, threshold=k / (len(vocabulary) + 2), top_k=top_k)

    return probabilities


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None, top_k=5):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts - 1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i + 1]

        # get previous n-gram
        n = len(list(n_gram_counts.keys())[0])
        previous_tokens = n * ["<s>"] + previous_tokens
        previous_n_gram = previous_tokens[-n:]

        # estimate probabilities
        probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)

        # filter
        probabilities = list(probabilities.items())
        probabilities = filter_probabilities(probabilities, start_with, threshold=k / (len(vocabulary) + 2), top_k=top_k)

        suggestions.append(probabilities)
    return suggestions


def compute_n_grams(sentences: list[list[str]]):
    unigram_counts = count_n_grams(sentences, 1)
    bigram_counts = count_n_grams(sentences, 2)
    trigram_counts = count_n_grams(sentences, 3)
    quadgram_counts = count_n_grams(sentences, 4)
    qintgram_counts = count_n_grams(sentences, 5)

    n_gram_counts_list = [unigram_counts, bigram_counts, trigram_counts, quadgram_counts, qintgram_counts]

    return n_gram_counts_list
