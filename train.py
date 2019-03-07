import modelutils
from inargs import in_args
from utils import print_message
import torch

in_args = in_args('train')

print_message("{:^55}".format("TRAINING STARTED"), "For CL argument info run '$python train.py -h' command")

# Reading parameters from command line parser
data_dir = in_args.data_dir
save_dir = in_args.save_dir
arch = in_args.arch
lr = in_args.learning_rate
hidden_units = in_args.hidden_units
epochs = in_args.epochs
device = 'cuda' if in_args.gpu and torch.cuda.is_available() else 'cpu'

print_message("Parameters used:".upper(), "Image directory: {}".format(data_dir),
              "Saving directory: {}".format(save_dir),
              "Model architecture: {}".format(arch), "Learning rate: {}".format(lr),
              "Hidden units: {}".format(hidden_units), "Epochs: {}".format(epochs),
              "Device: {}".format(device))

image_data_sets, data_loader = modelutils.data_loader(data_dir)

# Setting up model, criterion and optimizer
model, optimizer, criterion = modelutils.setup_model(arch, hidden_units, lr, device)

# Switch to cuda/cpu computing
model.to(device)

# Training model
trained_model = modelutils.train(model, criterion, optimizer, image_data_sets, data_loader, epochs, device)

# Saving trained model parameters
modelutils.save_checkpoint(trained_model, arch, image_data_sets['train'], lr, epochs, hidden_units, save_dir)

print_message("{:^55}".format("TRAINING FINISHED"))
