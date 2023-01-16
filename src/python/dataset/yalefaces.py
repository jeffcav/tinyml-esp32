import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2

SHAPE=(243, 320)

def _get_filepaths(folder_path: str):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

def _to_image(filepath: str, flatten: bool):
    image_pil = Image.open(filepath)
    image = np.array(image_pil, 'uint8')

    if flatten:
        return image.flatten()
    return image

def _to_label(filepath: str):
    # i.e. "/home/yalefaces/subject03.sad.gif" -> "subject03.sad"
    filename = os.path.split(filepath)[1]

    # i.e. "subject03.sad" -> "subject03"
    subject = filename.split(".")[0]

    # i.e. "subject03" -> "03"
    subject_id = subject.replace("subject", "")

    # i.e. "03" -> 2
    return int(subject_id) - 1

def load(folder_path: str, flatten=True):
    files = _get_filepaths(folder_path)

    # with CNNs we might not want to flatten
    # images, use flatten=False in this case
    flatten_array = [flatten] * len(files)

    X = np.array(list(map(_to_image, files, flatten_array)))
    y = np.array(list(map(_to_label, files)))

    # sort by subject
    sorted = np.argsort(y)
    X = X[sorted]
    y = y[sorted]

    return X, y

def plot_subset(faces, labels, subjects=[0,3,8,10], num_samples=11):
    plt.figure()
    _, axes = plt.subplots(4, num_samples, figsize=(26, 10), sharey=True)

    for ax, subject in zip(axes, subjects):
        subject_faces = faces[labels==subject]

        for axy, sample_idx in zip(ax, range(num_samples)):
            face = subject_faces[sample_idx].reshape(SHAPE)

            axy.imshow(face, cmap='gray')
            axy.set_title(f"subject: {subject}", fontsize=10)
            axy.set_xticks([])
            axy.set_yticks([])

    plt.subplots_adjust(left=0.4, bottom=0.1, right=None, top=0.61, wspace=0.2, hspace=0.2)
    plt.show()

def plot_face(faceimg):
    fig, (ax0, ax1, ax2) = plt.subplots(
        nrows=3,
        sharex=False,
        sharey=False,
        figsize=(28,2),
        dpi=800,
        gridspec_kw={"height_ratios": [10,0.5, 20]})

    # 2-D ARRAY
    ax0.set_title("2-D Grayscale")
    ax0.imshow(faceimg.reshape(SHAPE), cmap='gray')
    ax0.set_axis_off()

    # FLATTENED GRAYSCALE
    ax1.imshow(faceimg.reshape((1,-1)), cmap='gray', aspect='auto')
    ax1.set_title("1-D Grayscale")
    ax1.set_axis_off()

    # FLATTENED VALUES
    ax2.plot(faceimg)
    ax2.set_title("1-D Values")
    ax2.set_axis_off()

    plt.subplots_adjust(left=0.4, bottom=0.1, right=None, top=3, wspace=4, hspace=0.2)
    plt.axis('off')
    plt.show()

def plot_pixel_importance(images, labels):
    score, _ = chi2(images, labels)
    score_img = (np.array(score)).reshape(SHAPE)

    plt.figure(figsize=(8, 6))

    colormesh = plt.pcolormesh(score_img)
    plt.colorbar(colormesh)

    # mirror the y-axis because pcolormesh doesn't
    # do this trick automatically as imgshow does.
    plt.gca().invert_yaxis()
    plt.show()

def plot_eigenfaces(eigenfaces):
    plt.figure()
    _, axes = plt.subplots(1, len(eigenfaces), figsize=(18, 6), sharey=True)

    for ax, eigenface in zip(axes, eigenfaces):
        ax.imshow(eigenface.reshape(SHAPE), cmap='gray')

        ax.set_xticks([])
        ax.set_yticks([])  

    plt.show()

def plot_compressed(faces, comp_faces, labels, compressed_shape=(11,15), subjects=[0,1,2]):
    plt.figure()
    _, axes = plt.subplots(len(subjects), 2, figsize=(8, 8), sharey=False)

    for (ax0, ax1), subject in zip(axes, subjects):
        original = faces[labels==subject][0]
        compressed = comp_faces[labels==subject][0]

        ax0.imshow(original.reshape(SHAPE), cmap='gray')
        ax1.imshow(compressed.reshape(compressed_shape), cmap='gray')

        ax0.set_title(f"Subject {subject} (original)", size=12)
        ax1.set_title(f"Subject {subject} (compressed)", size=12)

        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1.set_xticks([])
        ax1.set_yticks([])

    plt.show()
