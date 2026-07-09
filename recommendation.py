from keybert import KeyBERT


def get_recommendations(split_docs):

    kw_model = KeyBERT()

    full_text = " ".join(
        [doc.page_content for doc in split_docs[:20]]
    )

    keywords = kw_model.extract_keywords(
        full_text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5
    )

    return [item[0] for item in keywords]