from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    folder = "/data/chaoyi_he/HSI/HSI_RGB/weights/rgb_recover_visual/predict_results"
    files = [f for f in os.listdir(folder) if f.endswith('.mat')]
    
    for f in files:
        image = loadmat(os.path.join(folder, f))['pred_rgb']
        #save the image as a .png file
        fig = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(os.path.join(folder, f.replace('.mat', '.png')))
        plt.close(fig=fig)