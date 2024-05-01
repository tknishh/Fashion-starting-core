import streamlit as st
from PIL import Image
from utils import load_lace_data
from model import get_recommendation
import os
import dotenv

dotenv.load_dotenv()
# Set up your OpenAI API key here. It's better to use environment variables or secure vaults in production.
API_KEY = os.environ.get('OPENAI_API_KEY')

def main():
    st.title('Lace Recommender System')

    # Load lace data
    laces = load_lace_data('data/laces', 'data/descriptions.txt')

    st.sidebar.title("User Input")
    user_input = st.sidebar.text_input("Enter your preferences or describe the event:")
    
    # Upload image for context (optional)
    uploaded_image = st.sidebar.file_uploader("Optionally, upload an image for better context", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        st.sidebar.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    if st.sidebar.button('Recommend Lace'):
        if user_input:  # Ensure there is some input to process
            try:
                recommended_lace = get_recommendation(user_input, laces, API_KEY)
                st.success("Recommendation successful!")
                st.write("Recommended Lace:", recommended_lace['name'])
                st.image(Image.open(recommended_lace['image_path']), caption=recommended_lace['name'])
                st.write("Description:", recommended_lace['description'])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter some preferences or upload an image to get recommendations.")

if __name__ == "__main__":
    main()
