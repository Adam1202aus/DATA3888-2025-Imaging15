import tensorflow as tf
import numpy as np
import os
from PIL import Image
import sys
import platform

# MODEL_PATH - This will now be set based on OS detection
MODEL_PATH_WINDOWS = r""
MODEL_PATH_LINUX = ""

TARGET_IMAGE_SIZE = (224, 224)


try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
    print("Using EfficientNet preprocess_input.")
except ImportError:
    print("Warning: Could not import a specific preprocess_input. Using generic scaling.")
    def preprocess_input(x):
        return x / 255.0


CLASS_LABELS = None

CURRENT_OS = platform.system()
MODEL_PATH = ""

if CURRENT_OS == "Windows":
    MODEL_PATH = MODEL_PATH_WINDOWS
    print(f"Running on Windows. Model path: {MODEL_PATH}")
elif CURRENT_OS == "Linux":
    MODEL_PATH = MODEL_PATH_LINUX
    print(f"Running on Linux (likely WSL). Model path: {MODEL_PATH}")
else:
    print(f"Unrecognized OS: {CURRENT_OS}. Please set MODEL_PATH manually.")
    MODEL_PATH = MODEL_PATH_WINDOWS


try:
    from skimage.color import rgb2gray, rgb2hed, rgb2ycbcr, rgb2lab
    from skimage.exposure import rescale_intensity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    if TARGET_COLORSPACE != "original":
        print(f"Warning: scikit-image is not installed, but TARGET_COLORSPACE is '{TARGET_COLORSPACE}'. "
              "Colorspace conversion will be skipped. Predictions might be inaccurate.")
        TARGET_COLORSPACE = "original"


def convert_colorspace_and_scale(image_np_rgb, target_colorspace_str):
    if not SKIMAGE_AVAILABLE and target_colorspace_str != "original":
        print(f"Info: Skipping colorspace conversion to {target_colorspace_str} as scikit-image is not available.")
        current_image_normalized_01 = image_np_rgb.astype(np.float32)
        if np.max(current_image_normalized_01) > 1.01:
             current_image_normalized_01 = current_image_normalized_01 / 255.0
        processed_image_intermediate = current_image_normalized_01
    elif target_colorspace_str == "original":
        current_image_normalized_01 = image_np_rgb.astype(np.float32)
        if np.max(current_image_normalized_01) > 1.01:
             current_image_normalized_01 = current_image_normalized_01 / 255.0
        processed_image_intermediate = current_image_normalized_01
    else:
        current_image_float = image_np_rgb.astype(np.float32)
        if np.max(current_image_float) > 1.01:
            current_image_normalized_01 = current_image_float / 255.0
        else:
            current_image_normalized_01 = current_image_float

        if target_colorspace_str == "grayscale":
            gray = rgb2gray(current_image_normalized_01)
            processed_image_intermediate = np.stack((gray,) * 3, axis=-1)
        elif target_colorspace_str == "hed":
            processed_image_intermediate = rgb2hed(current_image_normalized_01)
        elif target_colorspace_str == "ycbcr":
            processed_image_intermediate = rgb2ycbcr(current_image_normalized_01)
        elif target_colorspace_str == "cielab":
            processed_image_intermediate = rgb2lab(current_image_normalized_01)
        else:
            print(f"Warning: Unknown target_colorspace '{target_colorspace_str}'. Using original RGB.")
            processed_image_intermediate = current_image_normalized_01

    final_output_image_channels = []
    if processed_image_intermediate.ndim == 2:
        rescaled_channel = rescale_intensity(processed_image_intermediate, out_range=(0, 255))
        final_output_image = np.dstack((rescaled_channel, rescaled_channel, rescaled_channel))
    elif processed_image_intermediate.ndim == 3 and processed_image_intermediate.shape[2] == 3:
        for i in range(processed_image_intermediate.shape[2]):
            channel = processed_image_intermediate[:, :, i]
            if np.min(channel) < -0.1 or np.max(channel) > 1.1 and not (np.min(channel) >=0 and np.max(channel) <=255) :
                 rescaled_channel = rescale_intensity(channel, out_range=(0, 255))
            elif np.max(channel) <= 1.01 and np.min(channel) >= -0.01:
                 rescaled_channel = channel * 255.0
            else:
                rescaled_channel = channel
            final_output_image_channels.append(rescaled_channel)
        final_output_image = np.dstack(final_output_image_channels)
    elif processed_image_intermediate.ndim == 3 and processed_image_intermediate.shape[2] == 1:
        rescaled_channel = rescale_intensity(processed_image_intermediate[:, :, 0], out_range=(0, 255))
        final_output_image = np.dstack((rescaled_channel, rescaled_channel, rescaled_channel))
    else:
        print(f"Warning: Unexpected image shape {processed_image_intermediate.shape} after color conversion. Using as is.")
        final_output_image = processed_image_intermediate

    return final_output_image.astype(np.float32)


def normalize_path(path_str, current_os):
    if path_str.startswith('"') and path_str.endswith('"'):
        path_str = path_str[1:-1]
    elif path_str.startswith("'") and path_str.endswith("'"):
        path_str = path_str[1:-1]

    if current_os == "Linux":
        if path_str.startswith("C:\\") or path_str.startswith("c:\\"):
            path_str = "/mnt/c/" + path_str[3:].replace("\\", "/")
        elif path_str.startswith("D:\\") or path_str.startswith("d:\\"):
            path_str = "/mnt/d/" + path_str[3:].replace("\\", "/")
    elif current_os == "Windows":
        if path_str.startswith("/mnt/c/"):
            path_str = "C:\\" + path_str[7:].replace("/", "\\")
        elif path_str.startswith("/mnt/d/"):
            path_str = "D:\\" + path_str[7:].replace("/", "\\")

    return path_str


def preprocess_image(image_path, colorspace):
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(TARGET_IMAGE_SIZE)
        img_array_rgb = np.array(img_resized)
        img_array_processed_colorspace = convert_colorspace_and_scale(img_array_rgb, colorspace)
        img_array_preprocessed = preprocess_input(img_array_processed_colorspace)
        img_batch = np.expand_dims(img_array_preprocessed, axis=0)
        return img_batch
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_image(model, image_path_input, colorspace):
    processed_img = preprocess_image(image_path_input, colorspace)
    if processed_img is None:
        return

    try:
        predictions = model.predict(processed_img, verbose=0)
        predicted_index = np.argmax(predictions[0])

        if CLASS_LABELS:
            if 0 <= predicted_index < len(CLASS_LABELS):
                predicted_label = CLASS_LABELS[predicted_index]
            else:
                predicted_label = f"Unknown Index {predicted_index}"
        else:
            predicted_label = f"Index {predicted_index}"

        confidence = np.max(predictions[0])
        print(f"Prediction for {os.path.basename(image_path_input)}: {predicted_label} (Confidence: {confidence:.4f})")
        return (predicted_index, confidence)
    except Exception as e:
        print(f"Error during prediction for {image_path_input}: {e}")

def model_predict(path_to_model, img_path, color_space):
    MODEL_PATH = path_to_model
    colorspace = color_space
    if not MODEL_PATH:
        print("FATAL ERROR: MODEL_PATH is not set. Exiting.")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load model. {e}")
        sys.exit(1)

    user_input_raw = img_path

    user_input_normalized = normalize_path(user_input_raw, CURRENT_OS)

    if os.path.isfile(user_input_normalized):
        if user_input_normalized.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
            result = predict_image(model, user_input_normalized, colorspace)
            return result
        else:
            print(f"'{os.path.basename(user_input_normalized)}' is a file, but not a recognized image type. Skipping.")
    # elif os.path.isdir(user_input_normalized):
    #     print(f"\nProcessing images in folder: {user_input_normalized}")
    #     image_files_found = 0
    #     for item_name in os.listdir(user_input_normalized):
    #         item_path = os.path.join(user_input_normalized, item_name)
    #         if os.path.isfile(item_path) and item_path.lower().endswith(
    #                 ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
    #             image_files_found +=1
    #             predict_image(model, item_path)
    #     if image_files_found == 0:
    #         print("No image files found in the specified folder.")
    #     print(f"Finished processing folder: {user_input_normalized}\n")
    else:
        print(f"Error: Path '{user_input_raw}' (normalized to '{user_input_normalized}') is not a valid file or directory. Please try again.")
