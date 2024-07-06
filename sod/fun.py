import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
import numpy as np
import pickle
from model import ZFNet 
import pandas as pd
import cv2

data_dir = '/home/julia/Documents/python/sod/train_resized'

# Создаем пустой список для хранения данных
data = []

# Проходим по папкам "good" и "bad"
for folder in ['P0', 'P1']:
    folder_path = os.path.join(data_dir, folder)
    
    # Проходим по файлам в папке
    for filename in os.listdir(folder_path):
        # Создаем путь к файлу
        file_path = os.path.join(folder_path, filename)
        
        # Открываем изображение и преобразуем его в массив пикселей
        # img = cv2.imread(file_path)
        
        # Добавляем данные в список
        # data.append({
        #     'image': file_path,
        #     'class': 0 if folder == 'P0' else 1
        # })
        print(file_path)

# Создаем DataFrame из списка данных
# df = pd.DataFrame(data)

# Сохраняем DataFrame в CSV-файл
# df.to_csv('dataset.csv', index=False)
