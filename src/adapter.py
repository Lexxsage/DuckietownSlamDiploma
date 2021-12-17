import cv2
import os

loadedImages = []
current_position = 0


def load_all_images_from_folder(folder):
    for filename in folder:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            loadedImages.append(img)
    return loadedImages


def clear_loaded_images():
    return loadedImages.clear()


def get_all_images():
    return loadedImages


def get_next_image():
    global current_position
    image = loadedImages[1]
    print(current_position)
    current_position += 1
    print(current_position)
    return image


def get_image_from_current_position(position):
    global current_position
    image = loadedImages[position]
    current_position = position + 1
    return image


def clear_current_position():
    global current_position
    current_position = 1

