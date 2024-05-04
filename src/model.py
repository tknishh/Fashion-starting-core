import numpy as np
import openai
from scipy.spatial.distance import cosine

class LaceRecommender:
    def __init__(self, laces_data, openai_api_key):
        self.laces = laces_data
        self.api = OpenAI_API(api_key=openai_api_key)
        self.embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        embeddings = {}
        for lace_name, lace_info in self.laces.items():
            description = lace_info.get('description', None)
            if description is not None:
                embeddings[lace_name] = self.api.get_embedding(description)
            else:
                embeddings[lace_name] = np.zeros(768)  # default to zero embedding if no description
        return embeddings

    def recommend_lace(self, user_query):
        if user_query:
            query_embedding = self.api.get_embedding(user_query)
            best_lace = None
            min_distance = float('inf')

            for lace_name, desc_embedding in self.embeddings.items():
                if desc_embedding is not None:
                    distance = cosine(query_embedding, desc_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        best_lace = lace_name

            if best_lace:
                return self.laces.get(best_lace, None)
        return None

class OpenAI_API:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def get_embedding(self, text):
        try:
            response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            if 'data' in response and response['data'] and 'embedding' in response['data'][0]:
                return np.array(response['data'][0]['embedding'])
        except KeyError as e:
            print(f"KeyError accessing response data: {e}")
        return np.zeros(768)  # Return zero vector in case of error or missing data

def get_recommendation(user_input, laces_data, api_key):
    recommender = LaceRecommender(laces_data, api_key)
    recommended_lace = recommender.recommend_lace(user_input)
    return [recommended_lace] if recommended_lace else []