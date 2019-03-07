import argparse


class GetInputArgs(object):
    """
    Parsing arguments for flower image recognition project.
    Different arguments for training / prediction scripts.
    """

    TRAIN_DESC = "This is a Command Line application for training a model to classify flower images."
    PREDICT_DESC = "This is a Command Line application for loading a trained model, and classifying a flower image."

    def __init__(self, phase='train'):
        if phase == 'train':
            self.description = self.TRAIN_DESC
        else:
            self.description = self.PREDICT_DESC
        self.parser = argparse.ArgumentParser(description=self.description)
        self.phase = phase
        self.add_arguments()

    def add_arguments(self):
        if self.phase == 'train':
            self.parser.add_argument('data_dir', type=str)
            self.parser.add_argument('--save_dir', type=str, default='./', help='Directory for saving checkpoint')
            self.parser.add_argument('--arch', type=str, default='vgg16', help='Architecture of the model.')
            self.parser.add_argument('--learning_rate', type=float, default=0.001)
            self.parser.add_argument('--hidden_units', type=int, default=4096, help='Hidden units for the classifier.')
            self.parser.add_argument('--epochs', type=int, default=10)
            self.parser.add_argument('--gpu', action='store_true', default=False, help='Use cuda or not?')
        elif self.phase == 'predict':
            self.parser.add_argument('input', type=str, help='Input image for flower type prediction.')
            self.parser.add_argument('checkpoint', type=str, help='Path to checkpoint.pth file')
            self.parser.add_argument('--top_k', type=int, default='5', help='Top K predicted classes.')
            self.parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                                     help='Json file mapping file names to category names')
            self.parser.add_argument('--gpu', action='store_true', default=False, help='Use cuda or not?')
            self.parser.add_argument('--bar_chart', action='store_true', default=False,
                                     help='View bar plot with probabilities, or only print results?')
        else:
            raise ValueError("Invalid argument type, use 'train' or 'predict'.")

    def get_input_args(self):
        return self.parser.parse_args()


def in_args(phase='train'):
    args = GetInputArgs(phase)
    return args.get_input_args()
