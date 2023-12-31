import random
import numpy as np
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps


class AutoAugment(object):
    def __init__(self):
        self.policies = [
            ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
            ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
            ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
            ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
            ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
            ['Color', 0.4, 3, 'Brightness', 0.6, 7],
            ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
            ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
            ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
            ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
            ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
            ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
            ['Brightness', 0.9, 6, 'Color', 0.2, 8],
            ['Solarize', 0.5, 2, 'Invert', 0.0, 3],
            ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
            ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
            ['Color', 0.9, 9, 'Equalize', 0.6, 6],
            ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
            ['Brightness', 0.1, 3, 'Color', 0.7, 0],
            ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
            ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
            ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
            ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
        ]
    def __call__(self, img):
        img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img

# maps the names of image augmentation operations to lambda functions that take an image and a magnitude parameter as input,
# and return the augmented image.
operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
}

def apply_policy(img, policy):
    """
    Apply a given data augmentation policy to an image.

    Args:
        img (PIL Image or NumPy array): the input image to be transformed.
        policy (tuple): a tuple containing the parameters of the data augmentation policy,
            as generated by the `Policy` class.

    Returns:
        img (PIL Image or NumPy array): the transformed image.

    The `policy` parameter is a tuple with the following structure:
    (operation_1, p_1, magnitude_1, operation_2, p_2, magnitude_2)
    where:
    - operation_1 and operation_2 are strings representing the names of two data augmentation
      operations, chosen from the `operations` dictionary.
    - p_1 and p_2 are the probabilities of applying operation_1 and operation_2, respectively.
    - magnitude_1 and magnitude_2 are the magnitudes of the corresponding operations.

    The function applies the first operation with probability p_1 and magnitude magnitude_1,
    and the second operation with probability p_2 and magnitude magnitude_2. Each operation
    is applied by calling the corresponding function from the `operations` dictionary.

    """
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img

def transform_matrix_offset_center(matrix, x, y):
    """
    Apply an offset transformation to a given matrix to center it at the origin.

    Args:
        matrix (np.ndarray): Transformation matrix to apply offset to.
        x (int): Width of the input image.
        y (int): Height of the input image.

    Returns:
        np.ndarray: Offset transformation matrix.

    """
    # Calculate the center point of the image     
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5

    # Define offset and reset matrices to center the image at the origin
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])

    # Apply the offset transformation to the input matrix
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix

def shear_x(img, magnitude):
    """
    Applies a shear transformation in the x-direction to the input image.

    Args:
        img (PIL.Image): The input image.
        magnitude (int): The magnitude level of the shear. Must be an integer between 0 and 10.

    Returns:
        PIL.Image: The transformed image.
    """
    # Convert input image to numpy array
    img = np.array(img)

    # Define magnitudes for shear transformation
    magnitudes = np.linspace(-0.3, 0.3, 11)

    # Create a transformation matrix for shear in the x-direction with a random magnitude
    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    # Adjust the transformation matrix to center the image
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])

    # Extract affine transformation matrix and offset from the complete transformation matrix
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]

    # Apply the affine transformation to each channel of the input image
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    
    # Convert the resulting numpy array back to an Image object and return it
    img = Image.fromarray(img)
    return img

def shear_y(img, magnitude):
    """
    Applies vertical shearing to the input image.

    Args:
        img (PIL.Image.Image): The input image.
        magnitude (int): The magnitude level of the shearing transformation to be applied.

    Returns:
        PIL.Image.Image: The sheared image.
    """
    # Convert the input image to a NumPy array
    img = np.array(img)

    # Define the possible magnitudes for the transformation
    magnitudes = np.linspace(-0.3, 0.3, 11)

    # Generate a random transformation matrix with a vertical shearing component based on the magnitude
    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 1, 0],
                                 [0, 0, 1]])
    
    # Adjust the transformation matrix so that it is centered on the image
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])

    # Extract the affine matrix and offset from the transformation matrix
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]

    # Apply the affine transformation to each color channel of the image using NumPy
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    
    # Convert the NumPy array back to a PIL Image object
    img = Image.fromarray(img)

    # Return the sheared image
    return img

def translate_x(img, magnitude):
    """
    Applies a horizontal translation to an input image by a random amount within a specified range of magnitudes.
    The function first converts the input image to a numpy array, then applies an affine transformation matrix
    to shift the image horizontally. The magnitude parameter controls the amount of translation applied.

    Args:
        img (PIL.Image.Image): Input image to apply horizontal translation.
        magnitude (int): Integer index of the magnitude value to use from a predefined set of translation magnitudes.

    Returns:
        PIL.Image.Image: The adjusted image with horizontal translation applied.
    """
    # Convert input image to numpy array
    img = np.array(img)
    
    # Define range of magnitudes to use for translation
    magnitudes = np.linspace(-150/331, 150/331, 11)
    
    # Define affine transformation matrix for horizontal translation
    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1, img.shape[1]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 0, 1]])
    
    # Adjust the transformation matrix to ensure that the translation is applied from the center of the image
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    
    # Extract affine matrix and offset values from the adjusted transformation matrix
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    
    # Apply the affine transformation to each color channel of the input image
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    
    # Convert the resulting numpy array back to a PIL image
    img = Image.fromarray(img)
    return img

def translate_y(img, magnitude):
    """
    Translates an input image vertically by a random amount within a specified range. The amount of translation is
    controlled by the `magnitude` parameter, which indexes into a list of pre-defined magnitudes.

    Args:
        img (PIL.Image.Image): Input image to translate.
        magnitude (int): Index of the magnitude value to use for the random translation. Must be between 0 and 9.

    Returns:
        PIL.Image.Image: The translated image.
    """
    # Convert the input image to a numpy array for manipulation
    img = np.array(img)

    # Define a range of magnitudes for vertical translation
    magnitudes = np.linspace(-150/331, 150/331, 11)

    # Define a 3x3 transformation matrix to perform the translation
    transform_matrix = np.array([[1, 0, img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 1, 0],
                                 [0, 0, 1]])

    # Offset the transformation matrix to ensure that the translated image is centered
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])

    # Extract the 2x2 affine transformation matrix and translation offset from the full 3x3 matrix
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]

    # Apply the affine transformation to each color channel of the input image separately
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)

    # Convert the translated numpy array back to a PIL image
    img = Image.fromarray(img)
    return img

def rotate(img, magnitude):
    '''
    Rotate an image by a random angle within a specified range.

    Args:
        img (PIL.Image.Image): The input image.
        magnitude (int): An integer in the range [0, 10] indicating the magnitude
            of rotation to apply.

    Returns:
        PIL.Image.Image: The rotated image.
    '''
    # Convert the image to a NumPy array.
    img = np.array(img)

    # Define a range of magnitudes of rotation.
    magnitudes = np.linspace(-30, 30, 11)

    # Choose a random angle within the specified range.
    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    
    # Construct a transformation matrix to apply the rotation.
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    
    # Apply the transformation to the image.
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    
    # Convert the NumPy array back to a PIL image and return it.
    img = Image.fromarray(img)
    return img

def auto_contrast(img, magnitude):
    """
    Automatically adjusts the contrast of an input image by scaling its pixel values to span the full intensity
    spectrum. This enhances the visibility of image details and can improve the overall appearance of the image.

    Args:
        img (PIL.Image.Image): Input image to apply automatic contrast adjustment.
        magnitude (int): Unused parameter for compatibility with other image transformation functions.

    Returns:
        PIL.Image.Image: The adjusted image with improved contrast.
    """
    # Automatically adjust the contrast of the input image
    img = ImageOps.autocontrast(img)
    return img

def invert(img, magnitude):
    """
    Inverts the pixel values in an input image, resulting in a negative of the original image.

    Args:
        img (PIL.Image.Image): Input image to invert.
        magnitude (int): Unused parameter for compatibility with other image transformation functions.

    Returns:
        PIL.Image.Image: The adjusted image with inverted pixel values.
    """
    # Invert the pixel values in the input image
    img = ImageOps.invert(img)
    return img

def equalize(img, magnitude):
    """
    Enhances the contrast of an input image by redistributing its pixel values uniformly across the intensity
    spectrum. This results in a more balanced distribution of brightness and enhances the visibility of image
    details.

    Args:
        img (PIL.Image.Image): Input image to equalize.
        magnitude (int): Unused parameter for compatibility with other image transformation functions.

    Returns:
        PIL.Image.Image: The adjusted image with enhanced contrast.
    """
    # Apply the equalization effect to the input image
    img = ImageOps.equalize(img)
    return img

def solarize(img, magnitude):
    """
    Inverts the pixel values in an input image above a randomly selected threshold, resulting in a solarization effect.

    Args:
        img (PIL.Image.Image): Input image to apply solarization.
        magnitude (int): An integer value between 0 and 9, indicating the magnitude range to use for
            solarization. The range is defined by dividing the values between 0 and 256 into 10 equal intervals,
            and selecting the magnitude value and its next neighbor as the lower and upper bound of the range,
            respectively.

    Returns:
        PIL.Image.Image: The adjusted image with the desired solarization magnitude.
    """
    magnitudes = np.linspace(0, 256, 11)
    # Select a random solarization threshold within the specified range
    threshold = random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])
    # Apply the solarization effect to the input image
    img = ImageOps.solarize(img, threshold)
    return img

def posterize(img, magnitude):
    """
    Adjusts the number of colors in an input image by reducing the number of bits used to represent each
    pixel value, resulting in a poster-like effect. The number of bits is chosen randomly within a specified
    range.

    Args:
        img (PIL.Image.Image): Input image to adjust posterization.
        magnitude (int): An integer value between 0 and 9, indicating the magnitude range to use for
            posterization. The range is defined by dividing the values between 4 and 8 into 10 equal intervals,
            and selecting the magnitude value and its next neighbor as the lower and upper bound of the range,
            respectively.

    Returns:
        PIL.Image.Image: The adjusted image with the desired posterization magnitude.
    """
    magnitudes = np.linspace(4, 8, 11)
    # Select a random posterization magnitude within the specified range
    posterize_mag = int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])))
    # Apply the posterization effect to the input image
    img = ImageOps.posterize(img, posterize_mag)
    return img

def contrast(img, magnitude):
    """
    Adjusts the contrast of an input image by a random amount within a specified range.

    Args:
        img (PIL.Image.Image): Input image to adjust contrast.
        magnitude (int): An integer value between 0 and 9, indicating the magnitude range
            to use for adjusting contrast. The range is defined by dividing the values
            between 0.1 and 1.9 into 10 equal intervals, and selecting the magnitude
            value and its next neighbor as the lower and upper bound of the range, respectively.

    Returns:
        PIL.Image.Image: The adjusted image with the desired contrast magnitude.
    """
    magnitudes = np.linspace(0.1, 1.9, 11)
    # Select a random contrast magnitude within the specified range
    contrast_mag = random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])
    # Apply the contrast enhancement to the input image
    img = ImageEnhance.Contrast(img).enhance(contrast_mag)
    return img

def color(img, magnitude):
    """
    Adjusts the color saturation of an input image by a random amount within a specified range.

    Args:
        img (PIL.Image.Image): Input image to adjust color saturation.
        magnitude (int): An integer value between 0 and 9, indicating the magnitude range
            to use for adjusting color saturation. The range is defined by dividing the values
            between 0.1 and 1.9 into 10 equal intervals, and selecting the magnitude
            value and its next neighbor as the lower and upper bound of the range, respectively.

    Returns:
        PIL.Image.Image: The adjusted image with the desired color saturation magnitude.
    """
    magnitudes = np.linspace(0.1, 1.9, 11)
    # Select a random color saturation magnitude within the specified range
    color_mag = random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])
    # Apply the color saturation enhancement to the input image
    img = ImageEnhance.Color(img).enhance(color_mag)
    return img

def brightness(img, magnitude):
    """
    Adjusts the brightness of an input image by a random amount within a specified range.

    Args:
        img (PIL.Image.Image): Input image to adjust brightness.
        magnitude (int): An integer value between 0 and 9, indicating the magnitude range
            to use for adjusting brightness. The range is defined by dividing the values
            between 0.1 and 1.9 into 10 equal intervals, and selecting the magnitude
            value and its next neighbor as the lower and upper bound of the range, respectively.

    Returns:
        PIL.Image.Image: The adjusted image with the desired brightness magnitude.
    """
    magnitudes = np.linspace(0.1, 1.9, 11)
    # Select a random brightness magnitude within the specified range
    brightness_mag = random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])
    # Apply the brightness enhancement to the input image
    img = ImageEnhance.Brightness(img).enhance(brightness_mag)
    return img

def sharpness(img, magnitude):
    """
    Adjusts the sharpness of an input image by a random amount within a specified range.

    Args:
        img (PIL.Image.Image): Input image to adjust sharpness.
        magnitude (int): An integer value between 0 and 9, indicating the magnitude range
            to use for adjusting sharpness. The range is defined by dividing the values
            between 0.1 and 1.9 into 10 equal intervals, and selecting the magnitude
            value and its next neighbor as the lower and upper bound of the range, respectively.

    Returns:
        PIL.Image.Image: The adjusted image with the desired sharpness magnitude.
    """
    magnitudes = np.linspace(0.1, 1.9, 11)
    # Select a random sharpness magnitude within the specified range
    sharpness_mag = random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])
    # Apply the sharpness enhancement to the input image
    img = ImageEnhance.Sharpness(img).enhance(sharpness_mag)
    return img

def cutout(org_img, magnitude=None):
    """
    Applies a cutout operation to an input image by masking out a random rectangular
    region within the image.

    Args:
        org_img (PIL.Image.Image or ndarray): Input image to apply cutout operation.
        magnitude (int, optional): An integer value between 0 and 9, indicating the magnitude
            range to use for defining the size of the cutout region. If None, a default size of 16
            pixels is used.

    Returns:
        PIL.Image.Image: The adjusted image with the cutout region applied.
    """
    img = np.array(img)
    magnitudes = np.linspace(0, 60/331, 11)

    img = np.copy(org_img)
    mask_val = img.mean()

    # Define the size of the cutout region based on the magnitude value
    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])))

    # Define the location of the cutout region randomly within the image
    top = np.random.randint(0 - mask_size//2, img.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size//2, img.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    # Make sure the cutout region is within the bounds of the image
    if top < 0:
        top = 0
    if left < 0:
        left = 0

    # Apply the cutout operation by filling the selected region with the mean value of the image
    img[top:bottom, left:right, :].fill(mask_val)

    # Convert the resulting numpy array back to a PIL image and return it
    img = Image.fromarray(img)
    return img

class Cutout(object):
    """
    Cutout is a data augmentation technique that involves masking out a random rectangular
    region within an image.

    Args:
        length (int, optional): The length of each side of the square cutout region. Default is 16.

    Returns:
        PIL.Image.Image: The adjusted image with the cutout region applied.
    """
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        """
        Applies the cutout operation to the input image by masking out a random rectangular
        region within the image.

        Args:
            img (PIL.Image.Image): Input image to apply cutout operation.

        Returns:
            PIL.Image.Image: The adjusted image with the cutout region applied.
        """
        img = np.array(img)

        mask_val = img.mean()

        # Define the location of the cutout region randomly within the image
        top = np.random.randint(0 - self.length//2, img.shape[0] - self.length)
        left = np.random.randint(0 - self.length//2, img.shape[1] - self.length)
        bottom = top + self.length
        right = left + self.length

        # Make sure the cutout region is within the bounds of the image
        top = 0 if top < 0 else top
        left = 0 if left < 0 else top

        # Apply the cutout operation by masking out this region of the image
        img[top:bottom, left:right, :] = mask_val
        
        # Convert the resulting numpy array back to a PIL image and return it
        img = Image.fromarray(img)

        return img
