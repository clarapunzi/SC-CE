"""
This script generates a toy dataset with five classes. The first two classes are generated using the make_moons function from scikit-learn, while the other three classes are generated using the make_blobs function. The blobs are then scaled to fit in the same range as the moons. The dataset is then normalized to the range (-1, 1). The script also generates a plot of the dataset with the class labels and saves it as a PDF file.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs

def generate_toy(samples = 1000,
                 rs = 42,
                 noise=0.3#0.1
                 ):
    moon_samples = 1000#int(samples * 0.6)
    blob_samples = int(samples * 0.4)
    # Genreate Moons

    moons_X, moons_y = make_moons(n_samples=moon_samples, noise=noise, random_state=rs)
    # Generate blob-shaped data
    blobs_X, blobs_y = make_blobs(n_samples=blob_samples, centers=3, cluster_std=1.6, random_state=rs)
    # scale the blobs in the same range of the moons but with a different factor
    min_moons = moons_X.min(axis=0)
    max_moons = moons_X.max(axis=0)
    # scale the blobs in the same range of the moons but with a different factor
    min_blobs = blobs_X.min(axis=0)
    max_blobs = blobs_X.max(axis=0)
    scale = (max_moons - min_moons) / (max_blobs - min_blobs)
    blobs_X = min_moons + (blobs_X - min_blobs) * scale*1.2
    blobs_X[:,1] = blobs_X[:,1]

    # Combine the datasets
    #X = np.vstack((moons_X, blobs_X))
    #y = np.hstack((moons_y, blobs_y+2))  # Shift blob labels to avoid overlap with moon labels

    # just take the moons
    X,y = moons_X, moons_y
    # normalize X in range(-1,1)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 2 - 1


    cmap = plt.cm.get_cmap('Set2', len(set(y)))
    # Create a dictionary to map class numbers to colors
    class_color_map = {}
    for i in range(len(set(y))):
        class_color_map[i] = cmap(i)

    medoids = np.zeros((len(set(y)),2))
    # draw the points
    plt.scatter(X[:, 0], X[:, 1], c=[class_color_map[_y] for _y in y], cmap='Set1',vmin=0,vmax=4)
    for c in range(len(set(y))):
        cls_idx = np.where(y == c)[0]
        medoids[c] = np.median(X[cls_idx], axis=0)
        # write a text with the class number starting from 0
        plt.text(medoids[c, 0], medoids[c, 1], str(c), fontsize=40, ha='center', va='center')

    # put the background color to orange
    plt.gca().set_facecolor('lightblue')
    # Save the figure
    plt.title("Toy Dataset",fontsize=20)
    plt.savefig("toy_dataset.pdf")
    return X, y
