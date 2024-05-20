import os
import cv2
from sklearn.cluster import KMeans
from PIL import Image

def load_lace_descriptions(description_path):
    lace_descriptions = {}
    try:
        with open(description_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()  # Normalize key
                    value = parts[1].strip()
                    lace_descriptions[key] = value
    except FileNotFoundError as e:
        print(f"Description file not found: {description_path} - {e}")
    except Exception as e:
        print(f"Failed to load descriptions from {description_path}: {e}")
    return lace_descriptions

def load_lace_data(directory, description_path):
    laces = {}
    descriptions = load_lace_descriptions(description_path)
    if not descriptions:
        print("No descriptions were loaded, check the description file format and content.")

    supported_image_formats = ('.jpg', '.jpeg', '.png')
    try:
        for filename in os.listdir(directory):
            base_name, ext = os.path.splitext(filename)
            if ext.lower() in supported_image_formats:
                normalized_name = base_name.lower()  # Normalize filename
                file_path = os.path.join(directory, filename)
                description = descriptions.get(normalized_name, 'No description available')
                laces[normalized_name] = {
                    'image_path': file_path,
                    'name': base_name,
                    'description': description
                }
    except FileNotFoundError as e:
        print(f"Directory not found: {directory} - {e}")
    except Exception as e:
        print(f"Failed to load images from {directory}: {e}")
    return laces


def validate_image_path(image_path):
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")
    
def extract_colors(image_path, num_colors=3):
    """
    Extract dominant colors from an image.
    
    Parameters:
    image_path (str): The path to the image file.
    num_colors (int): The number of colors to extract.
    
    Returns:
    list: A list of color codes.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    clt = KMeans(n_clusters=num_colors)
    clt.fit(image)
    
    colors = clt.cluster_centers_
    colors = colors.astype(int)
    
    color_codes = ['#%02x%02x%02x' % tuple(color) for color in colors]
    return color_codes

def get_image(image_path):
    validate_image_path(image_path)
    try:
        with Image.open(image_path) as img:
            return img
    except IOError as e:
        raise IOError(f"Unable to open the image. The file may be corrupted or unsupported: {e}")
