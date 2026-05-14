import numpy as np
import cv2 as cv
from pathlib import Path
from image_handling import *

if __name__ == "__main__":
    # Resolve the test image relative to this file so the script works from any cwd.
    image_path = Path(__file__).resolve().parents[1] / "data" / "bouquet.png"

    # load image
    image = cv.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # display original image
    cv.imshow("Original Image", image)
    print(f"Original: {image.shape}, dtype: {image.dtype}")
    cv.waitKey(0)

    # function 1
    f1 = uint8_to_float(image)
    cv.imshow("uint8_to_float", f1)
    print(f"F1: {f1.shape}, dtype: {f1.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("uint8_to_float")

    # function 2, depended on the output of function 1
    f2 = float_to_uint8(f1)
    cv.imshow("float_to_uint8", f2)
    print(f"F2: {f2.shape}, dtype: {f2.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("float_to_uint8")

    # function 3
    f3 = crop(image, -10, -20, 2000, 2000)
    cv.imshow("crop", f3)
    print(f"F3: {f3.shape}, dtype: {f3.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("crop")

    # function 4
    f4 = scale_by_half_using_numpy(image)
    cv.imshow("scale_by_half_using_numpy", f4)
    print(f"F4: {f4.shape}, dtype: {f4.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("scale_by_half_using_numpy")

    # function 5
    f5 = scale_by_half_using_cv(image)
    cv.imshow("scale_by_half_using_cv", f5)
    print(f"F5: {f5.shape}, dtype: {f5.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("scale_by_half_using_cv")

    # PICKUP HERE

    # function 6
    f6 = horizontal_mirror_image(image)
    cv.imshow("horizontal_mirror_image", f6)
    print(f"F6: {f6.shape}, dtype: {f6.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("horizontal_mirror_image")

    # function 7
    f7 = rotate_counterclockwise_90(image)
    cv.imshow("rotate_counterclockwise_90", f7)
    print(f"F7: {f7.shape}, dtype: {f7.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("rotate_counterclockwise_90")

    # function 8
    f8 = swap_b_r(image)
    cv.imshow("swap_b_r", f8)
    print(f"F8: {f8.shape}, dtype: {f8.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("swap_b_r")

    # function 9
    f9 = blues(image)
    cv.imshow("blues", f9)
    print(f"F9: {f9.shape}, dtype: {f9.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("blues")

    # function 10
    f10 = greens(image)
    cv.imshow("greens", f10)
    print(f"F10: {f10.shape}, dtype: {f10.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("greens")

    # function 11
    f11 = reds(image)
    cv.imshow("reds", f11)
    print(f"F11: {f11.shape}, dtype: {f11.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("reds")

    # function 12
    f12 = scale_saturation(image, 2.0)
    cv.imshow("scale_saturation", f12)
    print(f"F12: {f12.shape}, dtype: {f12.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("scale_saturation")

    # function 13
    f13 = grayscale(image)
    cv.imshow("grayscale", f13)
    print(f"F13: {f13.shape}, dtype: {f13.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("grayscale")

    # function 14
    f14 = tile_bgr(image)
    cv.imshow("tile_bgr", f14)
    print(f"F14: {f14.shape}, dtype: {f14.dtype}")
    cv.waitKey(0)
    cv.destroyWindow("tile_bgr")

    # cleanup windows
    cv.destroyAllWindows()
