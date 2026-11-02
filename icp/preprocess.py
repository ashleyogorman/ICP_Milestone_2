"""

Step 1: Image Enhancement Step
Contract: enhance_fn(image: np.ndarray) -> np.ndarray:

The first step of the pipeline will be to use the cv2 library from python to enchance the base image if need be. The documentation to the library can be found here: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html.


Step 2: Channel Selection Step
Contract: channel_fn(image: np.ndarray) -> np.ndarray:

The next step is choose to extract a specific color channel from the image array. One color or a combination of colors or the original image may be chosen. As before the cv2 library is used.

"""
import cv2
import numpy as np

""" Step 1 """

# Baseline: no enhancement applied
def enhance_none(image: np.ndarray) -> np.ndarray:
    return image

""" Contrast """

# Ashley
def enhance_contrast_increase_25(image: np.ndarray) -> np.ndarray:
    contrast = 1.25
    brightness = 0
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

# Ashley
def enhance_contrast_increase_50(image: np.ndarray) -> np.ndarray:
    contrast = 1.5
    brightness = 0
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

# Ashley
def enhance_contrast_increase_75(image: np.ndarray) -> np.ndarray:
    contrast = 1.75
    brightness = 0
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

""" Brightness """

# Ashley
def enhance_brightness_decrease_50(image: np.ndarray) -> np.ndarray:
    contrast = 1.0
    brightness = -50
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

# Ashley
def enhance_brightness_increase_50(image: np.ndarray) -> np.ndarray:
    contrast = 1.0
    brightness = 50
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

# Ashley
def enhance_brightness_increase_100(image: np.ndarray) -> np.ndarray:
    contrast = 1.0
    brightness = 100
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

""" Laplacian """

# Ashley
def enhance_laplacian_simple(image: np.ndarray) -> np.ndarray:
    return cv2.Laplacian(image, -1)

# Ashley
def enhance_laplacian_precise(image: np.ndarray) -> np.ndarray:
    laplacian_64f = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_uint8 = cv2.convertScaleAbs(laplacian_64f)
    return laplacian_uint8

""" Saturation """

# Alex
def enhance_saturation_increase_50(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.convertScaleAbs(s, alpha=1.5, beta=0)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

""" Sharpen """

# Alex
def enhance_sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

""" Deskew """

# Alex
def deskew_rotation(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    card = cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))
    contours, _ = cv2.findContours(card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    _, _, angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    if angle > 45:
        angle -= 90
    height, width = image.shape[:2]
    return cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)


def enhance_deskew(image: np.ndarray) -> np.ndarray:
    rotation = deskew_rotation(image)
    if rotation is None:
        return image
    height, width = image.shape[:2]
    return cv2.warpAffine(image, rotation, (width, height), borderMode=cv2.BORDER_REPLICATE)


def unrotate_boxes(result: list[dict], rotation: np.ndarray) -> list[dict]:
    inverse = cv2.invertAffineTransform(rotation)
    restored = []
    for r in result:
        x1, y1, x2, y2 = r["box"]
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=float) @ inverse.T
        box = [corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()]
        restored.append({**r, "box": [round(float(v), 1) for v in box]})
    return restored

""" Step 2 """

# Baseline: no channel applied
def channel_none(image: np.ndarray) -> np.ndarray:
    return image

""" Single Colors """

# Ashley
def channel_blue(image: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(image)
    return b

# Ashley
def channel_green(image: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(image)
    return g

# Ashley
def channel_red(image: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(image)
    return r

""" Multicolors """

# Alex
def channel_red_green(image: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(image[:, :, 0], 0.5, image[:, :, 1], 0.5, 0)

# Alex
def channel_green_blue(image: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(image[:, :, 1], 0.5, image[:, :, 2], 0.5, 0)

# Alex
def channel_red_blue(image: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(image[:, :, 0], 0.5, image[:, :, 2], 0.5, 0)

""" Grayscale """

# Alex
def channel_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

""" Options """

enhancement_options = {
    "none": enhance_none,
    
    "+.25 contrast": enhance_contrast_increase_25,
    "+.50 contrast": enhance_contrast_increase_50,
    "+.75 contrast": enhance_contrast_increase_75,
    
    "-50 brightness": enhance_brightness_decrease_50,
    "+50 brightness": enhance_brightness_increase_50,
    "+100 brightness": enhance_brightness_increase_100,
    
    "simple laplacian": enhance_laplacian_simple,
    "precise laplacian": enhance_laplacian_precise,
    
    "+50% saturation": enhance_saturation_increase_50,
    "sharpen": enhance_sharpen,
    "deskew": enhance_deskew,
}

channel_options = {
    "none": channel_none,

    "blue": channel_blue,
    "green": channel_green,
    "red": channel_red,
    
    "red-green": channel_red_green,
    "green-blue": channel_green_blue,
    "red-blue": channel_red_blue,

    "grayscale": channel_grayscale,
}

"""Returns the processed image and the deskew rotation if any"""
def preprocess(image, enhancement, channel):
      rotation = deskew_rotation(image) if enhancement == "deskew" else None
      x = enhancement_options[enhancement](image)
      return channel_options[channel](x), rotation