import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from PIL import Image
from torchvision import transforms

# Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def print_message(*args):
    """ Beautifying message with two lines """

    print("=" * 60)
    for arg in args:
        print(arg)
    print("=" * 60)


def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch.
    Transforming is done with PyTorch transforms library.
    Output tensor is converted into Numpy array.
    """

    pil_image = Image.open(image_path)

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    tensor_image = trans(pil_image)
    return tensor_image.numpy()


def show_image(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)

    return ax.axis('off')


def display_result(image_path, probs, classes, cat_to_name):
    """
    Printing probabilities of the top k classes, and displaying the image in question.
    Classes are converted to flower names.

    :param cat_to_name: Dictionary of flower names mapped to numbers.
    :param image_path input image
    :param classes: Array of numbers corresponding to image names.
    :param probs: Array of probabilities for the classes.
    """

    labels = [cat_to_name[str(x)].upper() for x in classes]

    img = process_image(image_path)

    plt.figure(figsize=(6, 10))
    axs = plt.subplot(2, 1, 1)
    show_image(img, ax=axs)

    plt.subplot(2, 1, 2)
    sb.barplot(x=probs, y=labels, color=sb.color_palette()[0])
    plt.show()


def print_results(probs, classes, cat_to_name):
    """Prints top K flower names and corresponding probabilities."""

    labels = [cat_to_name[str(x)] for x in classes]
    for label, prob in zip(labels, probs):
        print("Label: {}, probability: {}".format(label, str(prob)))
