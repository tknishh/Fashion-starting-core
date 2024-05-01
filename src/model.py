# model.py
import numpy as np
import openai
from scipy.spatial.distance import cosine

class LaceRecommender:
    def __init__(self, laces_data, openai_api_key):
        self.laces = laces_data
        self.api = OpenAI_API(api_key=openai_api_key)
        self.embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        """ Generate embeddings for all lace descriptions using OpenAI's GPT-4. """
        embeddings = {}
        for lace_name, lace_info in self.laces.items():
            description = lace_info['description']
            if description:
                embeddings[lace_name] = self.api.get_embedding(description)
        return embeddings

    def recommend_lace(self, user_query):
        """ Recommend a lace based on textual similarity to the user's query. """
        query_embedding = self.api.get_embedding(user_query)
        best_lace = None
        min_distance = float('inf')

        for lace_name, desc_embedding in self.embeddings.items():
            distance = cosine(query_embedding, desc_embedding)
            if distance < min_distance:
                min_distance = distance
                best_lace = lace_name

        return self.laces[best_lace]

class OpenAI_API:
    """ Class for handling interactions with OpenAI API. """
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def get_embedding(self, text):
        """ Fetch an embedding for the given text using OpenAI's GPT-4 model. """
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002",  # You might choose a different model based on your subscription
            )
            # Extract the embedding vector from the response
            embedding_vector = response['data']['embedding']
            return np.array(embedding_vector)
        except Exception as e:
            print(f"An error occurred: {e}")
            return np.zeros(768)  # Return a zero vector in case of an error

def get_recommendation(user_input, laces_data, api_key):
    """ Function to get lace recommendation for a given user input. """
    recommender = LaceRecommender(laces_data, api_key)
    return recommender.recommend_lace(user_input)
