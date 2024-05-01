# utils.py
import os
from PIL import Image
import json

def load_lace_descriptions(description_path):
    """ Load lace descriptions from a text or JSON file. """
    lace_descriptions = {}
    if description_path.endswith('.txt'):
        with open(description_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    lace_descriptions[parts[0].strip()] = parts[1].strip()
    elif description_path.endswith('.json'):
        with open(description_path, 'r') as file:
            lace_descriptions = json.load(file)
    else:
        raise ValueError("Unsupported file format for descriptions.")
    return lace_descriptions

def load_lace_data(directory, description_path):
    """ Load lace images and descriptions into a dictionary. """
    descriptions = load_lace_descriptions(description_path)
    laces = {}
    supported_image_formats = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_image_formats):
            file_path = os.path.join(directory, filename)
            lace_name = filename.split('.')[0]
            laces[lace_name] = {
                'image_path': file_path,
                'name': descriptions.get(lace_name, 'No description available'),
                'description': descriptions.get(lace_name)
            }
    return laces

def validate_image_path(image_path):
    """ Check if the image file exists and is accessible. """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")
    if not os.path.isfile(image_path):
        raise ValueError(f"Path {image_path} is not a file.")

def get_image(image_path):
    """ Open an image from a path and handle errors. """
    validate_image_path(image_path)
    try:
        with Image.open(image_path) as img:
            return img
    except IOError:
        raise IOError("Unable to open the image. The file may be corrupted or unsupported.")
