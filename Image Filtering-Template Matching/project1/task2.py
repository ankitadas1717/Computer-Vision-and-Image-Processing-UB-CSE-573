
"""
Template Matching
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with template matching techniques, i.e., normalized cross correlation (NCC).

Please complete all the functions that are labelled with '# TODO'. When implementing those functions, comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'
and the functions you implement in 'task1.py' are of great help.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""

import argparse
import json
import os

import utils
from task1 import *


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args


def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:
    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) / (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5
    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """
    # Initialization
    sum_patch_element = 0
    sum_template_element = 0
    sum_patch_minus_mean_sq = 0
    sum_template_minus_mean_sq = 0
    patch_minus_mean = copy.deepcopy(patch)
    patch_minus_mean_sq = copy.deepcopy(patch)
    template_minus_mean = copy.deepcopy(template)
    template_minus_mean_sq = copy.deepcopy(template)
    ncc = 0

    # mean calculation for patch
    for i in range(len(patch)):
        for j in range(len(patch[0])):
            sum_patch_element += patch[i][j]
    patch_mean = sum_patch_element/(len(patch)*len(patch[0]))  #patch_mean

    #mean calculation for template
    for k in range(len(template)):
        for l in range(len(template[0])):
            sum_template_element += template[k][l]
    template_mean = sum_template_element / (len(template) *len(template[0]))  #template_mean

    #Calculation for numerator and denominator
    for x in range(len(patch)):
        for y in range(len(patch[0])):
            patch_minus_mean[x][y] = patch[x][y]-patch_mean                    # (patch - patch mean)
            patch_minus_mean_sq[x][y] = (patch[x][y]-patch_mean)**2            # (patch - patch mean)^2
            sum_patch_minus_mean_sq += patch_minus_mean_sq[x][y]               # sum[ (patch - patch mean)^2]
            template_minus_mean[x][y] = template[x][y] - template_mean         # (template - template mean)
            template_minus_mean_sq[x][y] = (template[x][y]-template_mean)**2   # (template - template mean)^2
            sum_template_minus_mean_sq += template_minus_mean_sq[x][y]         # sum[(template - template mean)^2]


    mult_patch_template = utils.elementwise_mul(patch_minus_mean, template_minus_mean)   # (patch - patch mean) * (template - template mean)
    ncc_nummerator = 0
    for g in range(len(mult_patch_template)):
        for h in range(len(mult_patch_template[0])):
            ncc_nummerator = ncc_nummerator + mult_patch_template[g][h]                  # Sum [(patch - patch mean) * (template - template mean)]

    ncc_denominator = 0
    ncc_denominator = (sum_patch_minus_mean_sq * sum_template_minus_mean_sq)**0.5        # Sum (patch - patch mean)^2 * Sum (template - template mean)^]
    if ncc_denominator != 0:
        ncc = ncc_nummerator/ncc_denominator                                             # NCC value calculation
    return float(ncc)
    # raise NotImplementedError



def match(img, template):
    """Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    """
    # TODO: implement this function.
    #Initialization
    max_ncc = 0
    x_pos = 0
    y_pos = 0

    for a in range(len(img)):
        for b in range(len(img[0])):
            patch = utils.crop(img, a, a + len(template), b, b + len(template))  # Cropping the patch
            ncc_value = norm_xcorr2d(patch, template)
            if ncc_value >= max_ncc:                                             # Calculating max ncc value
                max_ncc = ncc_value
                x_pos = a
                y_pos = b

    return x_pos,y_pos,max_ncc                                                   # Returning x position, y position and NCC value

    # raise NotImplementedError
    # raise NotImplementedError


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = read_image(args.template_path)

    x, y, max_value = match(img, template)
    # The correct results are: x: 17, y: 129, max_value: 0.994
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":    main()
