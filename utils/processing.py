import numpy as np
from PIL import Image
import cv2

import numpy as np
from PIL import Image

def load_and_preprocess_image(image, target_size=(224, 224)):

    image = image.convert("RGB")

    image_np = np.array(image)

    input_image_resize = cv2.resize(image_np, (224, 224))

    input_image_scaled = input_image_resize / 255

    image_reshape = np.reshape(input_image_scaled, [1, 224, 224, 3])

    return image_reshape


def get_prediction_label(prediction):

    if prediction[0][0] > 0.5:    
        return 1, prediction[0][0]
    
    else:
        return 0, 1.0 - prediction[0][0]
