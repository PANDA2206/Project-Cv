'''
@author: Prathmesh R Madhu.
For educational purposes only
'''

# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import os
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from selective_search import selective_search



def process_images(folder_path):
    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, image_file)
            image = skimage.io.imread(image_path)

            # perform selective search
            image_label, regions = selective_search(
                                    image,
                                    scale=500,
                                    min_size=20
                                )

            candidates = set()
            for r in regions:
                # excluding same rectangle (with different segments)
                if r['rect'] in candidates:
                    continue

                # excluding regions smaller than 2000 pixels
                if r['size'] < 2000:
                    continue

                # excluding distorted rects
                x, y, w, h = r['rect']
                if w/h > 1.2 or h/w > 1.2:
                    continue

                candidates.add(r['rect'])

            # Draw rectangles on the original image
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax.imshow(image)
            for x, y, w, h in candidates:
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=1
                )
                ax.add_patch(rect)
            plt.axis('off')

            # saving the image
            results_folder = 'results/'
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            result_image_path = os.path.join(results_folder, image_file)
            fig.savefig(result_image_path)
            plt.close()


def main():
    process_images('/Users/pankajrathi/Projcv/exercise5/data/arthist')
    process_images('/Users/pankajrathi/Projcv/exercise5/data/chrisarch')
    process_images('/Users/pankajrathi/Projcv/exercise5/data/classarch')
    
    # # loading a test image from '../data' folder
    # image_path = ['/Users/pankajrathi/Projcv/exercise5/data/chrisarch/ca-annun3.jpg']
    # image = skimage.io.imread(image_path)
    # # print (image.shape)
    #
    # # perform selective search
    # image_label, regions = selective_search(
    #                         image,
    #                         scale=500,
    #                         min_size=20
    #                     )
    #
    # candidates = set()
    # for r in regions:
    #     # excluding same rectangle (with different segments)
    #     if r['rect'] in candidates:
    #         continue
    #
    #     # excluding regions smaller than 2000 pixels
    #     # you can experiment using different values for the same
    #     if r['size'] < 2000:
    #         continue
    #
    #     # excluding distorted rects
    #     x, y, w, h = r['rect']
    #     if w/h > 1.2 or h/w > 1.2:
    #         continue
    #
    #     candidates.add(r['rect'])
    #
    # # Draw rectangles on the original image
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # ax.imshow(image)
    # for x, y, w, h in candidates:
    #     # print (x, y, w, h, r['size'])
    #     rect = mpatches.Rectangle(
    #         (x, y), w, h, fill=False, edgecolor='red', linewidth=1
    #     )
    #     ax.add_patch(rect)
    # plt.axis('off')
    # # saving the image
    # if not os.path.isdir('results/'):
    #     os.makedirs('results/')
    # fig.savefig('results/'+image_path.split('/')[-1])
    # plt.show()


if __name__ == '__main__':
    main()