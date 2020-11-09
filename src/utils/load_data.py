import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset


from .args import args



# ----- Data to Image Transformer -----

def data2img(arr, font_size=50, resolution=(256, 256), font=cv2.FONT_HERSHEY_SIMPLEX):
    """ Structured Tabular Data to Image with cv2

        NOTE currently supports only iris and wine dataset
    """
    x, y = resolution
    n_colums, n_features = 2, len(arr)
    n_lines = n_features % n_colums + int(n_features / n_colums)
    frame = np.ones((*resolution, 3), np.uint8)*0

    k = 0

    for i in range(n_colums):
        for j in range(n_lines):
            try:
                cv2.putText(
                    frame, str(arr[k]), (30 + i * (x // n_colums), 5 + (j + 1) * (y // (n_lines + 1))),
                    fontFace=font, fontScale=1, color=(255, 255, 255), thickness=2)
                k += 1
            except IndexError:
                break
    return np.array(frame, np.uint8)


# ----- Dataset -----

class CustomTensorDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        x = self.data[0][index]
        img = data2img(x)
        if self.transform:
            x = self.transform(img)

        if self.data[1] is not None:
            y = self.data[1][index]
            return x, y
        else:
            return x

if __name__ == "__main__":
    pass

