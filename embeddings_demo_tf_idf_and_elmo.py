import tensorflow as tf
import tensorflow_hub as hub
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tf.compat.v1.disable_eager_execution()


def preprocess_text(text: str) -> str:
    # Tokenization: Split the text into individual words
    tokens = text.split()

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Removing punctuation and special characters
    tokens = [re.sub(r"[^a-zA-Z0-9]", "", token) for token in tokens]

    # Removing empty tokens (if any, after removing special characters)
    tokens = [token for token in tokens if token]

    # Remove stop words
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }

    tokens = [token for token in tokens if token not in stop_words]

    return " ".join(tokens)


def get_weighted_average_embedding(embedding_model, tfidf_scores, idx):
    """
    Get the weighted average embedding for a sentence.

    Args:
    - tfidf_scores (dict): Dictionary containing TF-IDF scores for each term.
    - tfidf_matrix (csr_matrix): TF-IDF matrix for all sentences.
    - idx (int): Index of the sentence in the TF-IDF matrix.

    Returns:
    - numpy array: Weighted average embedding for the sentence.
    """
    # Compute the weighted average embedding
    weighted_embeddings = []
    for word, score_arr in tfidf_scores.items():
        score = score_arr[idx][0]
        embedding = embedding_model(
            [word],
            signature="default",
            as_dict=True,
        )[
            "elmo"
        ][0]
        weighted_embedding = score * embedding
        weighted_embeddings.append(weighted_embedding)

    # Average the embeddings
    average_embedding = sum(weighted_embeddings) / len(weighted_embeddings)

    return average_embedding


def compute_similarities(sentence_pairs, weighted_average_embeddings, all_sentences):
    # Calculate cosine similarities between user queries and restaurant descriptions
    similarities = []
    for i, (query, description) in enumerate(sentence_pairs):
        query_idx = all_sentences.index(query)
        description_idx = all_sentences.index(description)

        # Reshape the embeddings to 2D arrays
        query_embedding = weighted_average_embeddings[query_idx].reshape(1, -1)
        description_embedding = weighted_average_embeddings[description_idx].reshape(
            1, -1
        )

        similarity = cosine_similarity(query_embedding, description_embedding)[0][0]
        similarities.append(similarity)

    return similarities


elmo = hub.Module(
    "https://kaggle.com/models/google/elmo/frameworks/TensorFlow1/variations/elmo/versions/3",
    trainable=False,
)

restaurant_1_description = preprocess_text(
    "We slice and dice the food and make it hot and spicy."
)
restaurant_2_description = preprocess_text(
    "Your sweet tooth will be satisfied with our desserts among our general food menu."
)
restaurant_3_description = preprocess_text("We serve piquant food.")


user_query_1 = preprocess_text("spicy items")
user_query_2 = preprocess_text("sweet dish")

sentence_pairs = [
    (user_query_1, restaurant_1_description),
    (user_query_1, restaurant_2_description),
    (user_query_1, restaurant_3_description),
    (user_query_2, restaurant_1_description),
    (user_query_2, restaurant_2_description),
    (user_query_2, restaurant_3_description),
]

all_sentences = [
    restaurant_1_description,
    restaurant_2_description,
    restaurant_3_description,
    user_query_1,
    user_query_2,
]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the TF-IDF vectorizer on the restaurant descriptions
tfidf_vectorizer.fit(
    [restaurant_1_description, restaurant_2_description, restaurant_3_description]
)

# Transform all sentences using the fitted TF-IDF vectorizer
tfidf_matrix_all = tfidf_vectorizer.transform(all_sentences)

# Fetch the TF-IDF scores for terms in each sentence
tfidf_scores = {
    word: tfidf_matrix_all.getcol(idx).toarray()
    for word, idx in tfidf_vectorizer.vocabulary_.items()
}

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())

    # Compute the weighted average embeddings for all sentences
    weighted_average_embeddings = [
        sess.run(get_weighted_average_embedding(elmo, tfidf_scores, i))
        for i in range(tfidf_matrix_all.shape[0])
    ]

    # Compute cosine similarities between user queries and restaurant descriptions
    similarities = compute_similarities(
        sentence_pairs, weighted_average_embeddings, all_sentences
    )

    for i, (query, description) in enumerate(sentence_pairs):
        print(
            f"Query: {query}\nDescription: {description}\nSimilarity: {similarities[i]}\n"
        )
        print("\n")


"""
Query: spicy items
Description: we slice dice food make hot spicy
Similarity: 0.7129544019699097

Query: spicy items
Description: your sweet tooth satisfied our desserts among our general food menu
Similarity: 0.5356979370117188

Query: spicy items
Description: we serve piquant food
Similarity: 0.6104246377944946

Query: sweet dish
Description: we slice dice food make hot spicy
Similarity: 0.6239169836044312

Query: sweet dish
Description: your sweet tooth satisfied our desserts among our general food menu
Similarity: 0.610281765460968

Query: sweet dish
Description: we serve piquant food
Similarity: 0.545783281326294

"""