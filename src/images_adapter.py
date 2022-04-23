import cv2
import os

def ImagesAdapter():

    loadedImages = []
    current_position = 0

    def load_all_images_from_folder(folder):
        global loadedImages
        for filename in folder:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                loadedImages.append(img)

    def get_all_images():
        global loadedImages
        return loadedImages


    def clear_loaded_images():
        global loadedImages
        return loadedImages.clear()

    def get_next_image(position = current_position):
        global loadedImages
        image = loadedImages[position]
        if (position == current_position):
            current_position += 1
        return image

    def clear_current_position():
        global current_position
        current_position = 0

