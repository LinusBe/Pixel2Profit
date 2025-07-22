# src/data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional, Callable

class ImageDataset(Dataset):
    """A PyTorch Dataset for loading image data from .npz files.

    This class handles the lazy loading of images stored as compressed NumPy
    arrays. It takes a list of file paths and corresponding labels, and
    returns a single data point (image tensor, label tensor) when indexed.

    Parameters
    ----------
    image_paths : List[str]
        A list of file paths to the .npz image files.
    labels : np.ndarray
        A NumPy array of labels corresponding to each image.
    transforms : Optional[Callable]
        Optional transform to be applied on a sample. Defaults to None.
    """
    def __init__(self, image_paths: List[str], labels: np.ndarray, transforms: Optional[Callable] = None):
        """Initializes the ImageDataset.

        Parameters
        ----------
        image_paths : List[str]
            List of file paths to the .npz image files.
        labels : np.ndarray
            NumPy array of corresponding labels.
        transforms : Optional[Callable], optional
            Optional transform to apply to each image, by default None.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns
        -------
        int
            The number of samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a sample from the dataset at the specified index.

        Loads an image from its .npz file path, converts the image and its
        corresponding label into PyTorch tensors, normalizes the image pixel
        values to [0, 1], and applies any specified transforms. The image
        tensor dimensions are permuted to [C, H, W] format.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image tensor and the label tensor.
        """
        image_path = self.image_paths[idx]
        
        # ==================================================================
        # ÄNDERUNG: Lade aus einer .npz Datei
        # ==================================================================
        with np.load(image_path) as data:
            image = data['image'] # Zugriff über den Schlüssel 'image'
        # ==================================================================

        label = self.labels[idx]

        # Konvertiere zu PyTorch Tensoren
        # Wichtig: Permute ändert die Reihenfolge von (Height, Width, Channels) zu (Channels, Height, Width)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, label