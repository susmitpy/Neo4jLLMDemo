import tensorflow as tf
import tensorflow_hub as hub
import re

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


def euclidean_distance(v1, v2):
    """Compute Euclidean distance between two vectors."""
    return tf.norm(v1 - v2)


def get_embeddings(sentences: list[str], embedding_model: tf.Module) -> list:
    """Compute embeddings for a list of sentences using the provided embedding model."""
    embeddings_tensor = embedding_model(sentences, signature="default", as_dict=True)[
        "elmo"
    ]
    return embeddings_tensor


def compute_distances(query_embeddings: list, description_embeddings: list) -> list:
    """Compute Euclidean distances between pairs of query and description embeddings."""
    distances = []
    for query_embedding, description_embedding in zip(
        query_embeddings, description_embeddings
    ):
        distance = euclidean_distance(query_embedding, description_embedding)
        distances.append(distance)
    return distances


def get_distance(pair_of_sentences: tuple[str, str], embedding_model: tf.Module):
    embeddings_tensor = embedding_model(
        [pair_of_sentences[0], pair_of_sentences[1]],
        signature="default",
        as_dict=True,
    )["elmo"]

    distance = euclidean_distance(embeddings_tensor[0], embeddings_tensor[1])

    return distance


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


def compute_distances_for_query(query_embedding, all_description_embeddings):
    """Compute distances between a query and a list of descriptions."""
    distances = []
    for description_embedding in all_description_embeddings:
        distance = euclidean_distance(query_embedding, description_embedding)
        distances.append(distance)
    return distances


# Compute embeddings for all sentences together
all_texts = [
    user_query_1,
    user_query_2,
    restaurant_1_description,
    restaurant_2_description,
    restaurant_3_description,
]
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())

    all_embeddings = sess.run(
        elmo(all_texts, signature="default", as_dict=True)["elmo"]
    )

    # Separate the embeddings
    query_embeddings = all_embeddings[:2]  # Assuming you have 2 user queries
    description_embeddings = all_embeddings[2:]

    # Compute distances for each query against all descriptions
    for i, query_embedding in enumerate(query_embeddings):
        distances = sess.run(
            compute_distances_for_query(query_embedding, description_embeddings)
        )
        for j, distance in enumerate(distances):
            print(
                f"Query: {all_texts[i]}\nDescription: {all_texts[2 + j]}\nDistance: {distance}\n"
            )
            print("\n")

"""
Query: spicy items
Description: we slice dice food make hot spicy
Distance: 39.636348724365234

Query: spicy items
Description: your sweet tooth satisfied our desserts among our general food menu
Distance: 49.983680725097656

Query: spicy items
Description: we serve piquant food
Distance: 32.227256774902344

Query: sweet dish
Description: we slice dice food make hot spicy
Distance: 39.6467399597168

Query: sweet dish
Description: your sweet tooth satisfied our desserts among our general food menu
Distance: 49.735565185546875

Query: sweet dish
Description: we serve piquant food
Distance: 32.353485107421875
"""