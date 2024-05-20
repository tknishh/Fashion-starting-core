import streamlit as st
from PIL import Image
from utils import load_lace_data, extract_colors
from model import get_recommendation, get_colors
import os
import dotenv

dotenv.load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

def main():
    st.title('Lace Recommender System & Color Matcher')

    # Sidebar for Lace Recommender
    st.sidebar.title("Lace Recommender")
    st.sidebar.header("User Input")
    user_input = st.sidebar.text_input("Enter your preferences or describe the event:")

    uploaded_image = st.sidebar.file_uploader("Optionally, upload an image for better context", type=['jpg', 'jpeg', 'png'], key="lace_image")
    if uploaded_image is not None:
        st.sidebar.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    if st.sidebar.button('Recommend Lace', key="lace_button"):
        if user_input or uploaded_image:
            try:
                # Load lace data
                laces = load_lace_data('data/laces', 'data/descriptions.txt')
                recommended_laces = get_recommendation(user_input, laces, API_KEY)
                if recommended_laces:
                    recommended_lace = recommended_laces[0]
                    st.success("Recommendation successful!")
                    st.image(Image.open(recommended_lace['image_path']), caption=recommended_lace['name'])
                    st.write("Description:", recommended_lace['description'])
                else:
                    st.error("No recommendations found.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)} - {type(e).__name__}")
        else:
            st.error("Please enter some preferences or upload an image to get recommendations.")

    # Sidebar for Color Matcher
    st.sidebar.title("Color Matcher")
    st.sidebar.header("Image Upload")
    color_image = st.sidebar.file_uploader("Upload an image to find the best matching colors", type=['jpg', 'jpeg', 'png'], key="color_image")
    if color_image is not None:
        st.sidebar.image(color_image, caption='Uploaded Image', use_column_width=True)

    if st.sidebar.button('Find Colors', key="color_button"):
        if color_image:
            try:
                colors = get_colors(color_image, API_KEY)
                if colors:
                    st.success("Colors identified successfully!")
                    st.write("Here are the colors that suit your image the best:")
                    for color in colors:
                        color_box = f'<div style="width: 50px; height: 50px; background-color: {color["code"]}; display: inline-block; margin: 5px;"></div>'
                        st.markdown(color_box, unsafe_allow_html=True)
                        st.write(f'Color: {color["name"]}, Code: {color["code"]}')
                else:
                    st.error("No colors identified.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)} - {type(e).__name__}")
        else:
            st.error("Please upload an image to find the best matching colors.")

if __name__ == "__main__":
    main()