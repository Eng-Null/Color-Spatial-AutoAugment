import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import copy
from scipy import ndimage

class CS_AutoAugment(object):
    def __init__(self, numcolor=1, numspatial=1):
        self.color = numcolor
        self.spatial = numspatial
        self.applied_policies = []
        self.current_policies = None
        self.policies = [
            {
                'name': 'color',
                'sub_policies': [
                    {'factor': color, 'value': (0.4, 3)},
                    {'factor': color, 'value': (0.7, 7)},
                    {'factor': color, 'value': (0.9, 9)},
                    {'factor': color, 'value': (0.2, 8)},
                    {'factor': color, 'value': (0.7, 0)},
                    {'factor': brightness, 'value': (0.6, 7)},
                    {'factor': brightness, 'value': (0.7, 9)},
                    {'factor': brightness, 'value': (0.9, 6)},
                    {'factor': brightness, 'value': (0.1, 3)},
                    {'factor': contrast, 'value': (0.2, 6)},
                    {'factor': contrast, 'value': (0.6, 7)},
                    {'factor': auto_contrast, 'value': (0.5, 8)},
                    {'factor': auto_contrast, 'value': (0.4, 8)},
                    {'factor': auto_contrast, 'value': (0.6, 0)},
                    {'factor': auto_contrast, 'value': (0.8, 4)},
                    {'factor': auto_contrast, 'value': (0.9, 3)},
                    {'factor': auto_contrast, 'value': (0.9, 2)},
                    {'factor': auto_contrast, 'value': (0.9, 1)},
                    {'factor': invert, 'value': (0.1, 7)},
                    {'factor': invert, 'value': (0.0, 3)},
                    {'factor': invert, 'value': (0.1, 3)},
                    {'factor': equalize, 'value': (0.9, 2)},
                    {'factor': equalize, 'value': (0.6, 5)},
                    {'factor': equalize, 'value': (0.5, 1)},
                    {'factor': equalize, 'value': (0.3, 7)},
                    {'factor': equalize, 'value': (0.2, 0)},
                    {'factor': equalize, 'value': (0.6, 4)},
                    {'factor': equalize, 'value': (0.6, 6)},
                    {'factor': equalize, 'value': (0.8, 8)},
                    {'factor': solarize, 'value': (0.5, 2)},
                    {'factor': solarize, 'value': (0.2, 8)},
                    {'factor': solarize, 'value': (0.4, 5)},
                    {'factor': solarize, 'value': (0.8, 3)},
                    {'factor': posterize, 'value': (0.3, 7)},
                    {'factor': sharpness, 'value': (0.8, 1)},
                    {'factor': sharpness, 'value': (0.9, 3)},
                    {'factor': sharpness, 'value': (0.3, 9)},
                    {'factor': sharpness, 'value': (0.6, 5)},
                    {'factor': sharpness, 'value': (0.2, 6)},
                ],
            },
            {
                'name': 'spatial',
                'sub_policies': [
                    {'factor': shear_y, 'value': (0.5, 8)},
                    {'factor': shear_y, 'value': (0.2, 7)},
                    {'factor': translate_x, 'value': (0.3, 9)},
                    {'factor': translate_x, 'value': (0.5, 8)},
                    {'factor': translate_y, 'value': (0.7, 9)},
                    {'factor': translate_y, 'value': (0.4, 3)},
                    {'factor': translate_y, 'value': (0.9, 9)},
                    {'factor': rotate, 'value': (0.7, 2)},

                ],
            },
        ]
    
    def apply_augment(self):
        for i in range(self.color):
            policy = self.get_random_policy('color')
            if policy != None:
                self.apply_policy(policy)
        for i in range(self.spatial):
            policy = self.get_random_policy('spatial')
            if policy != None:
                self.apply_policy(policy)

    def get_random_policy(self, policy_type):
        available_policies = [p for p in self.current_policies if p['name'] == policy_type]
        if not available_policies:
            return None
        policy = random.choice(available_policies)
        
        return policy

    def apply_policy(self, policy):
        sub_policy = random.choice(policy['sub_policies'])
        factor = sub_policy['factor']
        value_range = sub_policy['value']
        value = random.uniform(*value_range)
        self.image = factor(self.image, value)
        policy['sub_policies'] = [p for p in policy['sub_policies'] if p['factor'] != factor]
    
    def __call__(self, img):
        self.current_policies = copy.deepcopy(self.policies)
        self.image = img
        self.apply_augment()
        return self.image
    

def transform_matrix_offset_center(matrix, x, y):   
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5

    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])

    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix

def shear_x(img, magnitude):
    
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)
    s = random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])
    transform_matrix = np.array([[1, s, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img

def shear_y(img, magnitude):
    
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)
    s = random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])
    transform_matrix = np.array([[1, 0, 0],
                                 [s, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)   
    img = Image.fromarray(img)
    return img

def translate_x(img, magnitude):

    img = np.array(img)
    magnitudes = np.linspace(-150/331, 150/331, 11)
    s = img.shape[1]*random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])
    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1, s],
                                 [0, 0, 1]])
    
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img

def translate_y(img, magnitude):
   
    img = np.array(img)
    magnitudes = np.linspace(-150/331, 150/331, 11)
    s = img.shape[0]*random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])
    transform_matrix = np.array([[1, 0, s],
                                 [0, 1, 0],
                                 [0, 0, 1]])

    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img

def rotate(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-30, 30, 11)
    theta = np.deg2rad(random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)]))
    
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    
    img = np.stack([ndimage.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)

    img = Image.fromarray(img)
    return img

def auto_contrast(img, magnitude):
    img = ImageOps.autocontrast(img)
    return img

def invert(img, magnitude):
    img = ImageOps.invert(img)
    return img

def equalize(img, magnitude):
    img = ImageOps.equalize(img)
    return img

def solarize(img, magnitude):
    magnitudes = np.linspace(0, 256, 11)
    threshold = random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])
    img = ImageOps.solarize(img, threshold)
    return img

def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)
    posterize_mag = int(round(random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])))
    img = ImageOps.posterize(img, posterize_mag)
    return img

def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    contrast_mag = random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])
    img = ImageEnhance.Contrast(img).enhance(contrast_mag)
    return img

def color(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    color_mag = random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])
    img = ImageEnhance.Color(img).enhance(color_mag)
    return img

def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    brightness_mag = random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])
    img = ImageEnhance.Brightness(img).enhance(brightness_mag)
    return img

def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    sharpness_mag = random.uniform(magnitudes[int(magnitude)], magnitudes[int(magnitude+1)])
    img = ImageEnhance.Sharpness(img).enhance(sharpness_mag)
    return img
