import os
from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets
import torch.nn as nn


app = Flask(__name__)

# Load models
people_model = torch.load('back-project/static/models/ZFNetModel/PeopleModelZFNet.pt')
other_model = torch.load('back-project/static/models/ResNetModel/OtherModelResNet.pt')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_translate')
def video_translate():
    file = request.files['file']
    object_type = request.form['object_type']

    dataset = datasets.ImageFolder(file, transform=transform)
    img = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    if object_type == 'person':
        model = people_model
    else:
        model = other_model
    
    model.eval()

    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in (img):
            outputs = model(images)
            loss = loss_func(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            if predicted.item() == 0:
                result = 'Good'
            else:
                result = 'Bad'
            return jsonify({'result': result})
            break

if __name__ == '__main__':
    app.run()
