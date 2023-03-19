import onnxruntime
from torchvision import transforms
from torch import Tensor
import onnx

import numpy as np
from PIL import Image
import os

# # Load the ONNX model
# onnx_model_path = os.path.abspath('model.onnx')
# ort_session = onnxruntime.InferenceSession(onnx_model_path)

# # Define the class names
# class_names = ['n01440764_tench', 'n01667114_mud_turtle']

# # Load the test images
# image_paths = [os.path.abspath('images/n01440764_tench.JPEG'), os.path.abspath('images/n01667114_mud_turtle.JPEG')]
# images = [Image.open(path) for path in image_paths]

# # Preprocess the images
# inputs = []
# def preprocess_numpy(img):
#     resize = transforms.Resize((224, 224))   #must same as here
#     crop = transforms.CenterCrop((224, 224))
#     to_tensor = transforms.ToTensor()
#     normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     img = resize(img)
#     img = crop(img)
#     img = to_tensor(img)
#     img = normalize(img)
#     return img

# for image in images:
#     image = preprocess_numpy(image) 
#     inputs.append(image)

# # Run the ONNX model
# outputs = ort_session.run(None, {'input': inputs})

# # Check the outputs
# for i, output in enumerate(outputs):
#     class_id = np.argmax(output)
#     class_name = class_names[class_id]
#     expected_id = i * 35
#     expected_name = class_names[expected_id]
#     if class_id != expected_id or class_name != expected_name:
#         print(f"Test failed for image {image_paths[i]}")
#         print(f"Expected class: {expected_id} ({expected_name}), but got {class_id} ({class_name})")
#     else:
#         print(f"Test passed for image {image_paths[i]}")

##################################################################

# model = onnx.load("model.onnx")

# # Get the name of the input node
# input_name = model.graph.input[0].name

# # Create a session with the ONNX model
# ort_session = onnxruntime.InferenceSession('model.onnx')

# # Load the test images
# image_paths = [".\\images\\n01440764_tench.jpeg", ".\\images\\n01667114_mud_turtle.JPEG"]
# images = [Image.open(path).convert('RGB') for path in image_paths]

# # Preprocess the test images
# preprocessed_images = []
# for image in images:
#     image = image.resize((224, 224), resample=Image.BILINEAR)
#     image = np.array(image).astype(np.float32) / 255.0
#     image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
#     image = image.transpose((2, 0, 1))
#     image = image.astype(np.float32)
#     preprocessed_images.append(image)

# # Run inference on the preprocessed images
# for i, image in enumerate(preprocessed_images):
#     # Create an input dictionary
#     input_dict = {input_name: image}

#     # Run the ONNX session to get the output
#     outputs = ort_session.run(None, input_dict)

#     # Get the class ID from the output
#     class_id = np.argmax(outputs)

#     # Print the result
#     print("Test image", i+1, "belongs to class ID", class_id)


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper

# model_dir ="./mnist"
model="model.onnx"
# path=sys.argv[1]
# path = ".\\images\\n01667114_mud_turtle.JPEG"

path = ".\\images\\n01440764_tench.jpeg"
 
#Preprocess the image
img = cv2.imread(path)
img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
img.resize((1, 3, 224, 224))


# image = image.resize((224, 224), resample=Image.BILINEAR)
# image = np.array(image).astype(np.float32) / 255.0
# image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
# image = image.transpose((2, 0, 1))
# image = image.astype(np.float32)



data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
#print(input_name)
#print(output_name)
 
result = session.run([output_name], {input_name: data})
prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
print(prediction)