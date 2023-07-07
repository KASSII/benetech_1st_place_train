import torchvision
from torchvision import transforms
from PIL import Image
import albumentations as A

def get_train_transforms():
    return A.Compose(
        [
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5),            
            A.OneOf([
                A.HueSaturationValue(p=1),
                A.RGBShift(p=1),
                A.ChannelShuffle(p=0.8),
                A.ToGray(p=0.8),
            ],p=0.5),
            A.OneOf([
                A.InvertImg(p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.1), p=1),
                A.CLAHE(p=1),
            ],p=0.1),
            A.Normalize(p=1),
        ], 
        p=1.0, 
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Normalize(p=1),
        ], 
        p=1.0, 
    )
    