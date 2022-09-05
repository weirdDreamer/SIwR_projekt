import sys
import cv2
import numpy as np
import os

if __name__ == "__main__":
    # pobranie pierwszego parametru (ścieżki do katalogu z plikami)
    # z parametów uruchomieniowych
    path = sys.argv[1]

    pic_names = os.listdir(path + '/frames')
    print(pic_names)
