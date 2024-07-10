
from PIL import Image
import os

def sss():
    for i in os.listdir("train/357/1"):
        im = Image.open(f"train/357/1/{i}")
        im = im.resize((30, 40))
        im.save(f"train_1/C1/{i[:-4]}.png")


sss()
