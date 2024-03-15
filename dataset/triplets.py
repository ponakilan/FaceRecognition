import os
import random

import pandas as pd
from torch.utils.data import Dataset
from models.mtcnn import MTCNN
from PIL import Image


def get_files_list(root_dir):
    """
    Generates a nested list of file paths for files inside the sub folders of a given directory,
    where each sublist corresponds to a single directory.

    Args:
        root_dir (str): Root directory containing sub folders with files.

    Returns:
        list: Nested list of file paths of the files present in the sub_folders of the root directory.

    Example:
        # Getting the list of files from a root directory \n
        root_directory = "/path/to/root_directory" \n
        files_list = get_files_list(root_directory) \n
        print(files_list)\n

        Output:
        [['/path/to/root_directory/sub_folder1/file1', '/path/to/root_directory/sub_folder1/file2', ...], ...]
    """
    list_root_dir = os.listdir(root_dir)

    # Getting the complete path of the child directories
    child_dirs = []
    for child_dirs_t in list_root_dir:
        child_dirs.append(os.path.join(root_dir, child_dirs_t))

    # Creating a nested list containing all the files with complete paths
    files = []
    for directory in child_dirs:
        cur_files = []
        for file in os.listdir(directory):
            cur_files.append(os.path.join(directory, file))
        files.append(cur_files)

    return files


def generate_dataset(root_dir):
    """
    Generates triplets of images (anchor, positive, negative) from the images present in the given directory
    for training FaceNet with a triplet loss function.

    Args:
        root_dir (str): Root directory containing sub folders of images.

    Returns:
        pandas.DataFrame: Dataframe containing triplets of file paths (anchor, positive, negative).

    Example:
        # Generating a dataset of triplets from a root directory
        root_directory = "/path/to/root_directory" \n
        dataset = generate_dataset(root_directory) \n
        print(dataset.head())
    """
    files = get_files_list(root_dir)

    anchor_files, positive_files, negative_files = [], [], []
    while len(files) > 0:

        # Stop if there are only two folders and one of them is empty
        if len(files) == 2 and len(files[0]) == 0 or len(files[1]) == 0:
            break

        # Remove any empty lists in files
        while [] in files:
            files.remove([])

        # Select random folders for anchor_positive and negative
        anchor_pos_dir = random.choice(files)
        while len(anchor_pos_dir) < 2:
            anchor_pos_dir = random.choice(files)
        neg_dir = random.choice(files)
        while anchor_pos_dir == neg_dir:
            neg_dir = random.choice(files)

        # Select images for anchor, positive and negative
        anchor = random.choice(anchor_pos_dir)
        positive = random.choice(anchor_pos_dir)
        while anchor == positive:
            positive = random.choice(anchor_pos_dir)
        negative = random.choice(neg_dir)

        # Remove the selected files from the list
        anchor_pos_dir.remove(anchor)
        anchor_pos_dir.remove(positive)
        neg_dir.remove(negative)

        # Append all the files to the respective lists
        anchor_files.append(anchor)
        positive_files.append(positive)
        negative_files.append(negative)

    # Create a dataframe with the generated lists
    face_triplets_df = pd.DataFrame({
        "anchor": anchor_files,
        "positive": positive_files,
        "negative": negative_files
    })

    return face_triplets_df


class TripletFaceDataset(Dataset):
    """
        Custom PyTorch dataset class for loading triplets of images for face recognition tasks.

        Args:
            triplets_dataframe (pandas.DataFrame): Pandas dataframe containing the paths of triplets.
            transform (callable, optional): A function/transform to be applied to each image triplet.
                                           Default is None.

        Methods:
            __len__(): Returns the number of triplets in the dataset.
            get_image(image_path): Reads an image from the specified path and applies the specified
                                   transformation (if any).
            __getitem__(idx): Retrieves and returns the anchor, positive, and negative images for
                              the triplet at the specified index.

        Attributes:
            dataframe (pandas.DataFrame): The input dataframe containing triplet information.
            transform (callable): The transformation to be applied on image retrieval.

        Note:
            This dataset assumes that the input dataframe has columns named 'anchor', 'positive',
            and 'negative' containing file paths for the anchor, positive, and negative images
            respectively.
    """

    def __init__(self, triplets_dataframe, transform):
        self.dataframe = triplets_dataframe
        self.mtcnn = MTCNN(
            weights_path="D:\PycharmProjects\FaceRecognition\models\TrainedWeights",
            transform=transform
        )

    def __len__(self):
        return self.dataframe.shape[0]

    def get_image(self, image_path):
        image = Image.open(image_path)
        faces, face_tensors = self.mtcnn(image)
        return face_tensors[0]

    def __getitem__(self, idx):
        data = self.dataframe.iloc[idx]
        anchor_image = self.get_image(data['anchor'])
        positive_image = self.get_image(data['positive'])
        negative_image = self.get_image(data['negative'])
        return anchor_image, positive_image, negative_image
