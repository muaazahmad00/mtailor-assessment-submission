import torch
import torchvision
import onnx
import os

# Load the PyTorch model
model = torchvision.models.resnet18()
model.load_state_dict(torch.load(os.path.abspath('resnet18-f37072fd.pth')))

# Create sample input data
input_data = torch.randn(1, 3, 224, 224)

# Convert the PyTorch model to ONNX format
torch.onnx.export(model, input_data, 'model.onnx', verbose=True)


# import torch
# import torchvision.transforms as transforms

# # Define the PyTorch model
# model = torchvision.models.resnet18()
# model.load_state_dict(torch.load(os.path.abspath('resnet18-f37072fd.pth')))

# # Define the pre-processing transformations
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# preprocess = transforms.Compose([transforms.Resize(256),
#                                  transforms.CenterCrop(224),
#                                  transforms.ToTensor(),
#                                  normalize])

# # Define the input tensor shape
# input_shape = (1, 3, 224, 224)

# # Define the dynamic axes for the input tensor
# dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'}}

# # Export the PyTorch model to ONNX with pre-processing steps
# torch.onnx.export(model, torch.randn(*input_shape), 'model.onnx',
#                   input_names=['input'], output_names=['output'],
#                   dynamic_axes=dynamic_axes, opset_version=11,
#                   do_constant_folding=True,
#                   example_outputs=torch.randn(1, 1000))

#---------------------------------------------------------------------------------

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import onnx
# import onnxruntime
# import os
# from PIL import Image

# # define the preprocessing function
# def preprocess(image_path):
#     # load the image using Pillow library
#     img = Image.open(image_path)
#     # convert to RGB if needed
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     # resize to 224x224 using bilinear interpolation
#     img = img.resize((224, 224), resample=Image.BILINEAR)
#     # convert image to tensor
#     img = transforms.ToTensor()(img)
#     # divide by 255
#     img /= 255.0
#     # normalize using mean and standard deviation
#     mean = torch.tensor([0.485, 0.456, 0.406])
#     std = torch.tensor([0.229, 0.224, 0.225])
#     img = transforms.Normalize(mean, std)(img)
#     # add batch dimension
#     img = img.unsqueeze(0)
#     # return the preprocessed image tensor
#     return img

# # load the PyTorch model
# model = torchvision.models.resnet18()
# model = model.load_state_dict(torch.load(os.path.abspath('resnet18-f37072fd.pth')))

# # convert the model to ONNX format with preprocessing steps
# input_shape = (1, 3, 224, 224)
# dummy_input = preprocess(os.path.abspath('images/n01440764_tench.jpeg'))
# onnx_model = torch.onnx.export(model, dummy_input, 'model.onnx', input_names=['input'], output_names=['output'])

# test the converted ONNX model
# ort_session = onnxruntime.InferenceSession('model.onnx')
# input_name = ort_session.get_inputs()[0].name
# output_name = ort_session.get_outputs()[0].name
# img1 = preprocess('n01440764_tench.JPEG')
# img2 = preprocess('n01667114_mud_turtle.JPEG')
# outputs = ort_session.run([output_name], {input_name: img1.numpy()})
# print('Image 1: Class ID:', outputs[0].argmax(), 'Class Name:', classes[outputs[0].argmax()])
# outputs = ort_session.run([output_name], {input_name: img2.numpy()})
# print('Image 2: Class ID:', outputs[0].argmax(), 'Class Name:', classes[outputs[0].argmax()])

