import numpy as np
import cv2 as cv
from image_handling import *

if __name__ == "__main__":
    # get relative path to test image in data/
    image_path = "A01/assignment/data/bouquet.png"

    # load image
    image = cv.imread(image_path)

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

    # cleanup windows
    cv.destroyAllWindows()
