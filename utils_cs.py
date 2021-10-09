from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, functional
import numpy as np


Image.MAX_IMAGE_PIXELS = 933120000

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

## Convert to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


        ## No need to use this ## 
def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        # Resize(400),
        # CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir_compressed, dataset_dir_original, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()

        # self.image_original_filenames   = [join(dataset_dir_original, x) for x in listdir(dataset_dir_original) if is_image_file(x)]
        # self.image_compressed_filenames = [join(dataset_dir_compressed, x) for x in listdir(dataset_dir_compressed) if is_image_file(x)]

        self.image_original_filenames = np.load(dataset_dir_original)
        self.image_compressed_filenames = np.load(dataset_dir_compressed)

        crop_size         =  calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform =  train_hr_transform(crop_size)
        self.lr_transform =  train_lr_transform(crop_size, upscale_factor)
        
        
    def __getitem__(self, index):

        # print(len(self.image_original_filenames))
        # print(self.image_compressed_filenames[index])
        
        hr_image = self.hr_transform(Image.fromarray(np.uint8(self.image_original_filenames[index])))
        lr_image = self.hr_transform(Image.fromarray(np.uint8(self.image_compressed_filenames[index])))

        # hr_image = self.hr_transform(Image.open(self.image_original_filenames[index]))
        # lr_image = self.hr_transform(Image.open(self.image_compressed_filenames[index]))   ## Here both the low_res image and high_res image has same scaling 
        
        return lr_image, hr_image

    def __len__(self):
        return 40000


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir_compressed, dataset_dir_original, crop_size, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        
        self.upscale_factor = upscale_factor

        # self.image_filenames            = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # self.image_original_filenames   = [join(dataset_dir_original, x) for x in listdir(dataset_dir_original) if is_image_file(x)]
        # self.image_compressed_filenames = [join(dataset_dir_compressed, x) for x in listdir(dataset_dir_compressed) if is_image_file(x)]

        self.image_original_filenames = np.load(dataset_dir_original)
        self.image_compressed_filenames = np.load(dataset_dir_compressed)

        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = Image.fromarray(np.uint8(self.image_original_filenames[index]))
        # print(self.image_original_filenames[index])
        # print(self.image_compressed_filenames[index])
        # hr_image = RandomCrop(self.crop_size)(hr_image)

        lr_image = Image.fromarray(np.uint8(self.image_compressed_filenames[index]))
        lr_scale = Resize(self.crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(self.crop_size, interpolation=Image.BICUBIC)
        lr_image_int = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image_int)
        return ToTensor()(lr_image), ToTensor()(hr_image)

    def __len__(self):
        return 200


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
