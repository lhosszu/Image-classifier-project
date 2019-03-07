import json

from inargs import in_args
from modelutils import predict, load_model
from utils import print_message, display_result, print_results
import torch

in_args = in_args('predict')

print_message("{:^55}".format("PREDICTION STARTED"), "For CL argument info run '$python predict.py -h' command")

# Reading parameters from CL parser
input_image = in_args.input
checkpoint = in_args.checkpoint
topk = in_args.topk
category_names = in_args.category_names
bar_chart = in_args.bar_chart
device = 'cuda' if in_args.gpu and torch.cuda.is_available() else 'cpu'

# Reading labels from json file
with open(category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)

# Loading model from checkpoint '.pth' file
loaded_model = load_model(checkpoint, device)

# Doing prediction
probabilities, classes = predict(input_image, loaded_model, device, topk)

# Printing results or displaying bar chart
if bar_chart:
    display_result(input_image, probabilities, classes, cat_to_name)
else:
    print_results(probabilities, classes, cat_to_name)

print_message("{:^55}".format("PREDICTION FINISHED"))
