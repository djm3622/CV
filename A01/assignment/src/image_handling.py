import numpy as np
import cv2 as cv  # noqa: F401


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with uint8 values in the range [0, 255] and
    return a copy of the image with data type float32 and values in the range [0, 1]
    """
    return image.astype(np.float32) / 255.0


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with float32 values in the range [0, 1] and
    return a copy of the image with uint8 values in the range [0, 255]. Values outside the range
    should be clipped (i.e. a float of 1.1 should be converted to a uint8 of 255, and a float of
    -0.1 should be converted to a uint8 of 0).
    """
    return (np.clip(np.round(image, 2), 0, 1) * 255.0).astype(np.uint8)


def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image cropped to the
    given rectangle. Any part of the rectangle that falls outside the image should be considered
    black (i.e. 0 intensity in all channels).
    """
    cropped = np.zeros((h, w, image.shape[2]), dtype=image.dtype)

    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, image.shape[1])
    y1 = min(y + h, image.shape[0])

    if x0 >= x1 or y0 >= y1:
        return cropped

    dx0 = max(-x, 0)
    dy0 = max(-y, 0)
    dx1 = dx0 + (x1 - x0)
    dy1 = dy0 + (y1 - y0)

    cropped[dy0:dy1, dx0:dx1, :] = image[y0:y1, x0:x1, :]
    return cropped


def scale_by_half_using_numpy(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image taking every
    other pixel in each row and column. For example, if the original image has shape (H, W, 3),
    the returned image should have shape (H // 2, W // 2, 3).
    """
    return image[::2, ::2, :].copy()


def scale_by_half_using_cv(image: np.ndarray) -> np.ndarray:
    """Using cv.resize, take an image and return a copy of the image scaled down by a factor of 2,
    mimicking the behavior of scale_by_half_using_numpy_slicing. Pay attention to the
    'interpolation' argument of cv.resize (see the OpenCV documentation for details).
    """
    return cv.resize(
        image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv.INTER_NEAREST
    )


def horizontal_mirror_image(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image flipped
    horizontally (i.e. a mirror image). The behavior should match cv.flip(image, 1).
    """
    return image[:, ::-1, :].copy()


def rotate_counterclockwise_90(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image rotated
    counterclockwise by 90 degrees. The behavior should match
    cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE).
    """
    return image.transpose(1, 0, 2)[::-1, :, :].copy()


def swap_b_r(image: np.ndarray) -> np.ndarray:
    """Given an OpenCV image in BGR channel format, return a copy of the image with the blue and red
    channels swapped. You may use any numpy or opencv functions you like.
    """
    return image[:, :, [2, 1, 0]].copy()


def blues(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the blue
    channel
    """
    zeros = np.zeros_like(image)
    blues = image[:, :, 0]

    zeros[:, :, 0] = blues
    return zeros


def greens(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the green
    channel
    """
    zeros = np.zeros_like(image)
    greens = image[:, :, 1]

    zeros[:, :, 1] = greens
    return zeros


def reds(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the red
    channel
    """
    zeros = np.zeros_like(image)
    reds = image[:, :, 2]

    zeros[:, :, 2] = reds
    return zeros


def scale_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    """Take an OpenCV image in BGR channel format. Convert to HSV and multiply the saturation
    channel by the given scale factor, then convert back to BGR.
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)

    return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)


def grayscale(image: np.ndarray) -> np.ndarray:
    """Using numpy, reproduce the OpenCV function cv.cvtColor(image, cv.COLOR_BGR2GRAY) to convert
    the given image to grayscale. The returned image should still be in BGR channel format.
    """
    gray = image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299
    return np.stack((gray, gray, gray), axis=-1).astype(image.dtype)


def tile_bgr(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a 2x2 tiled copy of the image, with the
    original image in the top-left, the blue channel in the top-right, the green channel in the
    bottom-left, and the red channel in the bottom-right. If the original image has shape (H, W, 3),
    the returned image has shape (2 * H, 2 * W, 3).
    """
    blue = blues(image)
    green = greens(image)
    red = reds(image)

    top = np.concatenate((image, blue), axis=1)
    bottom = np.concatenate((green, red), axis=1)

    return np.concatenate((top, bottom), axis=0)
