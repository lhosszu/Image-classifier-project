import copy
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch import optim
from torchvision import datasets, transforms

from utils import print_message, process_image

# Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_INFO = {"vgg16": 25088, "alexnet": 9216, "densenet121": 1024}


def data_loader(data_dir):
    """
    This function takes a directory of images, processes the images, and returns a dictionary of
    Pytorch dataloaders.

    :param data_dir: directory with /train /valid /test subdirectories
    :return: a dictionary of dataloaders with 'train', 'valid' and 'test' keys
             && a dictionary of image datasets with 'train', 'valid' and 'test' keys
    """

    all_data_sets = ('train', 'test', 'valid')
    train_dir, valid_dir, test_dir = data_dir + '/train', data_dir + '/valid', data_dir + '/test'
    directories = {'train': train_dir, 'test': test_dir, 'valid': valid_dir}

    # Data transforms for image pre-processing
    # For validation and testing, only resizing and cropping is done
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(40),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(MEAN, STD)]),

        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(MEAN, STD)]),

        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    }

    image_data_sets = {x: datasets.ImageFolder(directories[x], transform=data_transforms[x]) for x in all_data_sets}
    loader = {x: torch.utils.data.DataLoader(image_data_sets[x], batch_size=32) for x in all_data_sets}
    return image_data_sets, loader


def setup_model(arch, hidden_units, learning_rate, device):
    """
    This function prepares a transfer learning model (with the chosen architecture) to be trained.

    :param arch: name of used transfer learning model
    :param hidden_units: output size of the first hidden layer in the classifier
    :param learning_rate: optimizer LR
    :param device: CPU / CUDA (GPU)
    :return: ready-to-be-trained model
    """

    # Downloading the right model, raising error if it is not available
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError("{} not available, try one of the following: 'vgg16', 'densenet121', 'alexnet'.".format(arch))

    # Freezing gradients
    for param in model.parameters():
        param.requires_grad = False

    # Size of the first fully connected layer in the classifier
    in_size = MODEL_INFO[arch]

    classifier = nn.Sequential(OrderedDict([
        ('fully_connected_1', nn.Linear(in_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fully_connected_2', nn.Linear(hidden_units, 102)),
        ('output_layer', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    criterion = nn.NLLLoss()

    # Switch to GPU computing if possible
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, optimizer, criterion


def train(model, criteria, optimizer, image_datasets, data_loaders, number_of_epochs=3, device='gpu'):
    """This method trains the model, and prints training/validation losses and accuracies."""

    start_time = time.time()

    if torch.cuda.is_available() and device == 'gpu':
        model.to(device)

    training_phases = ('train', 'valid')
    best_state_dict = model.state_dict()
    best_accuracy = 0

    for epoch in range(number_of_epochs):
        print_message('Epoch {}/{}'.format(epoch + 1, number_of_epochs))

        for training_phase in training_phases:
            dataset_size = len(image_datasets[training_phase])
            if training_phase == 'train':
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_correct = 0

            for inputs, labels in data_loaders[training_phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(training_phase == 'train'):
                    outputs = model(inputs)
                    temp, predictions = torch.max(outputs, 1)
                    loss = criteria(outputs, labels)
                    if training_phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(predictions == labels.data)
            epoch_loss = running_loss / dataset_size
            epoch_accuracy = running_correct.double() / dataset_size

            print('Phase: {}, loss: {:.3f}, accuracy: {:.1f} %'.format(training_phase, epoch_loss,
                                                                       epoch_accuracy * 100.0))

            if training_phase == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_state_dict = copy.deepcopy(model.state_dict())

    elapsed_time = time.time() - start_time

    print_message(
        'Training took {:.0f} minutes {:.0f} seconds'.format(elapsed_time // 60, elapsed_time % 60))
    print_message('Best validation accuracy: {:1f}%'.format(best_accuracy * 100.0))

    model.load_state_dict(best_state_dict)
    return model


def test_accuracy(model_to_test, test_loader, device):
    """Testing model, printing accuracy and calculating elapsed time."""

    print_message("Testing model accuracy")
    start_time = time.time()

    model_to_test.eval()
    model_to_test.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model_to_test(images)
            __, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the model: %d %%' % (100 * correct / total))
    elapsed_time = time.time() - start_time
    print_message("Testing accuracy took {:05.2f} minutes".format(elapsed_time / 60))


def save_checkpoint(model_to_save, arch, dataset, learning_rate, no_of_epochs, hidden_layer_size, save_dir,
                    file_name='classifier'):
    """
    This method saves the model and training parameters to 'name.pth' file.
    Default file name is classifier.pth.
    """

    file_path = save_dir + file_name + '.pth'
    model_to_save.class_to_idx = dataset.class_to_idx
    model_to_save.cpu()

    torch.save({'arch': arch,
                'LR': learning_rate,
                'epochs': no_of_epochs,
                'hidden_units': hidden_layer_size,
                'state_dict': model_to_save.state_dict(),
                'class_to_idx': model_to_save.class_to_idx,
                'input_size': (3, 224, 224),
                'output_size': 102}, file_path)


def predict(image_path, model, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.to(device)
    numpy_array = process_image(image_path)

    # Converting numpy array back to torch tensor
    tensor = torch.tensor(numpy_array).float()
    tensor = tensor.unsqueeze_(0)

    with torch.no_grad():
        output = model.forward(tensor.to(device))
    probabilities = F.softmax(output.data, dim=1)
    result = probabilities.topk(topk)

    # Converting softmax results to numpy arrays
    # Returning 1D arrays of probabilities/classes
    probs = result[0].cpu().detach().numpy()
    classes = result[1].cpu().detach().numpy()
    return probs[0], classes[0]


def load_model(checkpoint, device):
    """ Loads checkpoint and returns the model. """

    checkpoint = torch.load(checkpoint)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    learning_rate = checkpoint['LR']
    print("Loading model with architecture: {}".format(arch))

    if hidden_units >= MODEL_INFO[arch]:
        raise ValueError("Invalid number of hidden units, please adjust to this size: " + str(MODEL_INFO[arch]))

    model, criterion, optimizer = setup_model(arch, hidden_units, learning_rate, device)
    model.class_to_idx = checkpoint['class_to_idx']
    in_size = MODEL_INFO[arch]

    classifier = nn.Sequential(OrderedDict([
        ('fully_connected_1', nn.Linear(in_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fully_connected_2', nn.Linear(hidden_units, 102)),
        ('output_layer', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    print("Model loaded successfully")
    return model
