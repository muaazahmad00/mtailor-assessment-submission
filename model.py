import onnxruntime
import numpy as np
import cv2

class OnnxModel:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)

    def predict(self, image):
        image = np.expand_dims(image, axis=0)
        inputs = {'input': image.astype('float32')}
        outputs = self.session.run(None, inputs)
        return np.argmax(outputs)

class ImagePreprocessor:
    def __init__(self, size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.size = size
        self.mean = mean
        self.std = std

    def preprocess(self, image):
        # Convert to RGB format if needed
        if image.shape[2] == 3 and image.shape[2] != 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to (224, 224) with bilinear interpolation
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)

        # Convert to float and divide by 255
        image = image.astype(np.float32) / 255.0

        # Normalize using mean and standard deviation values for each channel
        image -= self.mean
        image /= self.std

        # Convert from (H, W, C) to (C, H, W) format
        image = np.transpose(image, (2, 0, 1))

        return image


