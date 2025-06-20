import os
import sys
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd


class MyDataLoader(Dataset):
    def __init__(self, noisy_image_path, clean_image_path, image_height, image_width):
        """
        IMAGES_PATHS: list of images paths ['./data_Image/Image/0001_NI000_slice000.png', 
        './data_Image/Image/0001_NI000_slice001.png']

        MASKS_PATHS: list of masks paths ['./data_Image/Mask/0001_MA000_slice000.png',
        './data_Image/Mask/0001_MA000_slice001.png']
        """
        self.noisy_image_path = noisy_image_path
        self.clean_image_path = clean_image_path

        self.image_names = [f for f in os.listdir(
            noisy_image_path) if f.endswith(('.png'))]
        self.image_names = [f for f in self.image_names if os.path.exists(
            os.path.join(clean_image_path, f))]

        self.image_height = image_height
        self.image_width = image_width
        self.transformations = transforms.Compose([transforms.PILToTensor()])

    def transform(self, noisy_image, clean_image):

        # noisy_image = noisy_image.convert('RGBA')
        # clean_image = clean_image.convert('L')

        noisy_image = self.transformations(noisy_image)
        clean_image = self.transformations(clean_image)

        noisy_image, clean_image = noisy_image.type(
            torch.FloatTensor), clean_image.type(torch.FloatTensor)

        return noisy_image, clean_image

    def adjust_dimensions(self, noisy_image, clean_image):
        # image resize to the shape
        new_resolution = (self.image_width, self.image_height)
        resized_noisy_image = noisy_image.resize(
            new_resolution, Image.Resampling.LANCZOS)
        resized_clean_image = clean_image.resize(
            new_resolution, Image.Resampling.LANCZOS)

        return resized_noisy_image, resized_clean_image

    def __getitem__(self, index):
        img_name = self.image_names[index]

        noisy_path = os.path.join(self.noisy_image_path, img_name)
        clean_path = os.path.join(self.clean_image_path, img_name)

        # print(noisy_path)

        noisy_image = Image.open(noisy_path).convert('RGBA')
        clean_image = Image.open(clean_path).convert('L')

        noisy_image, clean_image = self.adjust_dimensions(
            noisy_image, clean_image)

        noisy_image, clean_image = self.transform(
            noisy_image, clean_image)
        return noisy_image, clean_image, noisy_path, clean_path

        # cnt_try = 0
        # # loop in case if there are any corrupted files
        # while cnt_try < 10 and index < self.__len__():
        #     try:
        #         img_name = self.image_names[index]

        #         noisy_path = os.path.join(self.noisy_image_path, img_name)
        #         clean_path = os.path.join(self.clean_image_path, img_name)

        #         print(noisy_path)

        #         noisy_image = Image.open(noisy_path).convert('RGBA')
        #         clean_image = Image.open(clean_path).convert('L')

        #         noisy_image, clean_image = self.adjust_dimensions(
        #             noisy_image, clean_image)

        #         noisy_image, clean_image = self.transform(
        #             noisy_image, clean_image)
        #         return noisy_image, clean_image, self.noisy_path, self.clean_path

        #     except Exception as e:
        #         # if the image is corrupted, load the next image
        #         print("Corrupted file: ",
        #               self.noisy_image_path[index], '  |  ', sys.exc_info()[0])
        #         print(e)
        #         index += 1
        #         cnt_try += 1
        # raise ("Could not resolve Corrupted file: ",
        #        self.noisy_image_path[index], '  |  ', sys.exc_info()[0])

    def __len__(self):
        return len(self.noisy_image_path)


def Load_data(mode, image_width, image_height):
    noisy_folder = "./data_image/noise/"
    clean_folder = "./data_image/clean/"

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = MyDataLoader(noisy_image_path=noisy_folder, clean_image_path=clean_folder,
                           image_height=image_height, image_width=image_width)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Split dataset
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    if mode == 'Train':
        return train_dataset

    if mode == 'Test':
        return test_dataset
