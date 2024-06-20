'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###
    segments = skimage.segmentation.felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)

    # print(segments)

    # Expand the segmentation mask to 3 channels
    segments_colored = np.expand_dims(segments, axis=-1)
    # print(segments_colored)
    # Merge the segmentation mask as the 4th channel
    im_orig = np.concatenate((im_orig, segments_colored), axis=-1)
    # print(im_orig)

    return im_orig

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###

    hist_r1 = np.array(r1["hist_c"])
    hist_r2 = np.array(r2["hist_c"])

    # # Normalize histograms using L1 norm
    # hist_r1 = np.sum(hist_r1)
    # hist_r2 = np.sum(hist_r2)

    # Calculate the sum of histogram intersection of color
    intersection = np.minimum(hist_r1, hist_r2).sum()

    return intersection


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """

    ### YOUR CODE HERE ###
    # Assuming 'hist_t' in r1 and r2 represents the texture histogram
    hist_t1 = np.array(r1["hist_t"])
    hist_t2 = np.array(r2["hist_t"])

    # # # Normalize the texture histograms within the similarity function
    # epsilon = 1e-10
    # hist_t1 /= np.sum(hist_t1) + epsilon
    # hist_t2 /= np.sum(hist_t2) + epsilon

    # Calculate histogram intersection
    intersection_sum = np.minimum(hist_t1, hist_t2).sum()

    return intersection_sum


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """

    ### YOUR CODE HERE ###
    size_ij = r1["size"] + r2["size"]
    size_similarity = 1.0 - size_ij / imsize

    return size_similarity


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """

    ### YOUR CODE HERE ###

    bb_ij = (np.maximum(r1["max_x"], r2["max_x"]) - np.minimum(r1["min_x"], r2["min_x"])) * \
                (np.maximum(r1["max_y"], r2["max_y"]) - np.minimum(r1["min_y"], r2["min_y"]))

    fill_similarity = 1.0 - (bb_ij - r1["size"] - r2["size"]) / imsize
    # print(fill_similarity.shape)

    return fill_similarity

def calc_sim(r1, r2, imsize):

    sum = (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))
    return sum

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])

    ### YOUR CODE HERE ###
    for colour in range(3):

        c = img[:,colour]
        hist = np.hstack([hist, np.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    hist= hist / len(img.flatten())
    # print(hist)
    return hist


def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    grad = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###
    for colour_channel in range(2):
        grad[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)
    return grad

def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###
    for colour_channel in range(2):
        # Mask by the color channel
        fd = img[:, colour_channel]

        # Calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = np.hstack([hist, np.histogram(fd, BINS, (0.0, 1.0))[0]])

        # L1 Normalize
    hist = hist / len(img.flatten())

    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''

    R = {}
    ### YOUR CODE HERE ###
    # Get HSV image
    HSV = skimage.color.rgb2hsv(img[:, :, :3])
    # Pass 1: Count pixel positions
    for y, row in enumerate(img):
        for x, (r, g, b, label) in enumerate(row):
            # Initialize a new region
            if label not in R:
                R[label] = {
                    "min_x": float('inf'), "min_y": float('inf'),
                    "max_x": 0, "max_y": 0, "labels": [label]}

            # Bounding box
            R[label]["min_x"] = min(R[label]["min_x"], x)
            R[label]["min_y"] = min(R[label]["min_y"], y)
            R[label]["max_x"] = max(R[label]["max_x"], x)
            R[label]["max_y"] = max(R[label]["max_y"], y)

    # Pass 2: Calculate texture gradient
    tex_grad = calc_texture_gradient(img)

    # Pass 3: Calculate color and texture histograms of each region
    for l, region in R.items():
        # Color histogram
        masked_pixels = HSV[:, :, :][img[:, :, 3] == l]
        region["size"] = len(masked_pixels) / 4
        region["hist_c"] = calc_colour_hist(masked_pixels)

        # Texture histogram
        region["hist_t"] = calc_texture_hist(tex_grad[:, :][img[:, :, 3] == l])

    return R

def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Convert regions dictionary to a list for easier iteration
    region_list = list(regions.items())
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###

    # Iterate through pairs of regions and check for intersection
    for current_index, region_a in enumerate(region_list[:-1]):
        for region_b in region_list[current_index + 1:]:
            if intersect(region_a[1], region_b[1]):
                neighbours.append((region_a, region_b))


    return neighbours

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    ### YOUR CODE HERE


    # Calculate new bounding box coordinates
    min_x = min(r1["min_x"], r2["min_x"])
    min_y = min(r1["min_y"], r2["min_y"])
    max_x = max(r1["max_x"], r2["max_x"])
    max_y = max(r1["max_y"], r2["max_y"])

    # Merge color histograms by weighted average
    hist_c = (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size

    # Merge texture histograms by weighted average
    hist_t = (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size

    # Merge labels
    labels = r1["labels"] + r2["labels"]

    # Create the merged region dictionary
    rt = {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "size": new_size,
        "hist_c": hist_c,
        "hist_t": hist_t,
        "labels": labels
    }

    return rt



def selective_search(image_orig, scale=1.0, sigma=1, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        # Mark similarities for regions to be removed
        removed = [k for k, v in S.items() if (i in k) or (j in k)]


        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for k in removed:
            del S[k]

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###
        # Calculate similarity set with the new region using list comprehension
        S.update({(t, n): calc_sim(R[t], R[n], imsize) for k in removed if k != (i, j) for n in k if
                  n != i and n != j})

    # Task 8: Generating the final regions from R
    regions = [
        {
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']
            ),
            'size': r['size'],
            'labels': r['labels']
        }
        for k, r in R.items()
    ]

    ### YOUR CODE HERE ###


    return image, regions


# if __name__ == '__main__':
#     if __name__ == '__main__':
#         # Load image and perform selective search
#         image_path = '/Users/pankajrathi/Projcv/exercise5/data/arthist/adoration1.jpg'
#         image = skimage.io.imread(image_path)
#         pank =generate_segments(image,scale =1,sigma= 0.8,min_size=50)
#         print(pank.shape)
#
#         print(image.shape)





