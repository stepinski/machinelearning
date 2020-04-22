import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # TODO initialize model layers here
        # For this, you need to make use of Conv2D layers and MaxPool2d layers
        self.flatten = Flatten()
        self.Conv2D = nn.Conv2d(1,64,(3,3))
        self.MaxPool2d = nn.MaxPool2d((2,2))
        self.linear = nn.Linear(13,64)
        self.output1 = nn.Linear(64,64)      # for top digit
        self.output2 = nn.Linear(64,64) 
        self.model = nn.Sequential(
              nn.Conv2d(input_dimension, 64, (3, 3)),
              nn.MaxPool2d((2, 2))
            )

    def forward(self, x):

        # TODO use model layers to predict the two digits
        
        xf = self.flatten(x)
        conv= self.Conv2D(x)
        maxc = self.MaxPool2d(conv)
        # input= self.linear(maxc)
        # xf = self.flatten(maxc)
        # out=self.linear(xf)
        # out1=self.output1(xf)
        # out2=self.output2(xf)
        conf= self.model
        conf.add_module("linear", torch.nn.Linear(13,64))

        output1 = conf(xf)
        output2 = conf(xf)
        print("test")
        tst=torch.max(output1, 1)[1]
        tst2=torch.max(output2, 1)[1]
        out1=tst.view(tst.numel())
        out2=tst2.view(tst2.numel())
        
        # print(xl[:, 1:2])
        # print(out1)
        return output1,output2

        # return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
