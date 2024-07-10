import cv2
import os
import torch
import torch.nn.functional as F
from cv2 import imread
from config import get_config
from load import load_model
from transform import get_test_augmentation
from PIL import Image
from torchvision import transforms

def predict(img, model, batch_t):
    with torch.no_grad():
        outputs, edge_mask, ds_map = model(batch_t)

    h, w = img.shape[:2]
    output = (
        F.interpolate(outputs[0].unsqueeze(0), size=(h, w), mode="bilinear")[0][0]
        .cpu()
        .numpy()
    )
    return output



from PIL import Image
import os

def sss():
    for i in os.listdir("train/357/1"):
        im = Image.open(f"train/357/1/{i}")
        im = im.resize((60, 80))
        im.save(f"train_2/C1/{i[:-4]}.png")

    for i in os.listdir("train/357/0"):
        im = Image.open(f"train/357/0/{i}")
        im = im.resize((60, 80))
        im.save(f"train_2/C0/{i[:-4]}.png")

    for i in os.listdir("train/163/0"):
        im = Image.open(f"train/163/0/{i}")
        im = im.resize((60, 80))
        im.save(f"train_2/P0/{i[:-4]}.png")

    for i in os.listdir("train/163/1"):
        im = Image.open(f"train/163/1/{i}")
        im = im.resize((60, 80))
        im.save(f"train_2/P1/{i[:-4]}.png")




def clean():
    #вне цикла
    arch = "5"
    cfg = get_config(int(arch))
    transform = get_test_augmentation(cfg.img_size)
    model = load_model(cfg, device="cpu")
    count = 0


def router():
    #вне цикла
    arch = "5"
    cfg = get_config(int(arch))
    transform = get_test_augmentation(cfg.img_size)
    model = load_model(cfg, device="cpu")
    count = 0
    img = imread("train/163/1/6824438.jpg")
    img_t = transform(image=img)["image"]
    batch_t = torch.unsqueeze(img_t, 0)

    output = predict(img, model, batch_t)

    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = (output) * 255
    cv2.imwrite("new.png", rgba)


router()









