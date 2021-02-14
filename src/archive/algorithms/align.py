# Based on examples by Satya Mallick (https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/)
import cv2
import numpy as np
import time


def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def align_signals(signal1, signal2):
    """Aligns two signal arrays using a similarity measure
    called Enhanced Correlation Coefficient (ECC).

    Parameters
    ----------
    signal1 : ndarray, dtype : uint16 or float
        Signal array
    signal2 : ndarray, dtype : uint16 or float
        Signal array, will be aligned to signal1

    Returns
    -------
    signal2_aligned : ndarray
        Aligned version of signal2
    """


def align_stacks(stack1, stack2):
    """Aligns two stacks of images using the gradient representation of the image
    and a similarity measure called Enhanced Correlation Coefficient (ECC).
    TODO try Feature-Based approach https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/, https://github.com/spmallick/learnopencv/blob/c8e3ae2d2b0423f5c6d21c6189ee8ff3192c0555/ImageAlignment-FeatureBased/align.py

    Parameters
    ----------
    stack1 : ndarray, dtype : uint16
        Image stack with shape (x, y, t)
    stack2 : ndarray, dtype : uint16
        Image stack with shape (x, y, t), will be aligned to stack_dual

    Returns
    -------
    stack2_aligned : ndarray
        Aligned version of stack2
    """
    # Read unit16 grayscale images from the image stacks
    im1 = np.float32(stack1[..., 0])
    im2 = np.float32(stack2[..., 0])
    data_type = im1.dtype

    # Find the width and height of the image
    im_size = im1.shape
    width, height = im_size[0], im_size[1]
    print('im1 min, max: ', np.nanmin(im1), ' , ', np.nanmax(im1))
    print('im2 min, max: ', np.nanmin(im2), ' , ', np.nanmax(im2))
    # Find the number of frames in the stacks (should be identical)
    frames = stack1.shape[2]

    # Allocate space for aligned image
    im2_aligned = np.zeros((width, height), dtype=np.uint16, order='F')
    # Define motion model
    warp_mode = cv2.MOTION_TRANSLATION
    # Define 2x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Specify the number of iterations
    number_of_iterations = 5000
    # Specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    start = time.time()
    # Warp the second stack image to the first
    # Run the ECC algorithm, the results are stored in warp_matrix
    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(im1), get_gradient(im2),
                                             warp_matrix, warp_mode, criteria)
    # Use Affine warp when the transformation is not a Homography
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (height, width),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    print('im2_aligned min, max: ', np.nanmin(im2_aligned), ' , ', np.nanmax(im2_aligned))
    # Convert aligned image back to uint16
    im2_aligned = np.uint16(im2_aligned)
    print('im2_aligned min, max: ', np.nanmin(im2_aligned), ' , ', np.nanmax(im2_aligned))

    # Align and save every stack2 frame using the same process
    stack2_aligned = np.zeros((width, height, frames), dtype=np.uint16, order='F')
    for i in range(frames):
        # Find the old frame
        stack2_frame = np.float32(stack2[..., i])
        stack2_frame_aligned = cv2.warpAffine(stack2_frame, warp_matrix, (height, width),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # Convert aligned frame back to uint16
        stack2_frame_aligned = np.uint16(stack2_frame_aligned)
        # Save the aligned frame in the new stack
        stack2_aligned[..., i] = stack2_frame_aligned

    # # ECC Method
    # # Image registration using first frame
    # # Read the images to be aligned
    # im1 = np.float32(stack_dual[0, ...])
    # im2 = np.float32(stack2[0, ...])
    # # # Convert images to grayscale
    # # im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # # im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # # Find size of image1
    # sz = im1.shape
    # im1_min, im1_max = np.nanmin(im1), np.nanmax(im1)
    # im2_min, im2_max = np.nanmin(im2), np.nanmax(im2)
    # print('im1 min, max: ', im1_min, ' , ', im1_max)
    # print('im2 min, max: ', im2_min, ' , ', im2_max)
    #
    # # Define the motion model
    # warp_mode = cv2.MOTION_TRANSLATION
    # # Define 2x3 matrices and initialize the matrix to identity
    # warp_matrix = np.eye(2, 3, dtype=np.float32)
    # # Specify the number of iterations.
    # number_of_iterations = 5000
    # # Specify the threshold of the increment
    # # in the correlation coefficient between two iterations
    # termination_eps = 1e-10
    # # Define termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    #
    # # Run the ECC algorithm. The results are stored in warp_matrix.
    # (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)
    #
    # # Use warpAffine for Translation, Euclidean and Affine
    # im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    #
    # cv2.imshow("Aligned Image 2", im2_aligned)
    # cv2.waitKey(0)
    end = time.time()
    print('Alignment time (s): ', end - start)

    return stack2_aligned
