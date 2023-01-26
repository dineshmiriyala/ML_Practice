if __name__ == "__main__":
    #useful imports
    import matplotlib.pyplot as plt
    import numpy
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    #from torch.testing._internal.distributed.rpc.examples.parameter_server_test import batch_size

    #dataset used for this neural network is CIFAR-10 dataset: It has 10 classes of different objects, about 10000 images for
    #each class, it is a predefined dataset from torchvision datasets. I will be building neural network from scratch.
    # I will be using convolution neural networks for this and I will
    #be defining model from scratch. I am using PYTORCH for faster training and GPU support.
    # here is the dataset link for reference https://www.cs.toronto.edu/~kriz/cifar.html

    #classes: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.

    #steps:
    #1. load and normalize the data.
    #2. Define the convolution network.
    #3. Define loss function and optimizer.
    #4. Train the model on training dataset.
    #5. Test the model on test datset.
    #6. Build an web API to classify any given image into these classes.

    #step-1: load and normalize the data.

    #normalizing image in the range of [0,1] for faster calculation

    transform = transforms.Compose([transforms.ToTensor() ,
                                   transforms.Normalize((0.5,0.5,0.5) , (0.5,0.5,0.5))]) #Using mean and standard
                                                                                        # deviation 0.5 to normalize the
                                                                                        #image in range of [-1,1]
                                                                                        #three means and stds for R,G,B

    batch_size = 4

    num_workers = 6#I made it equal to  my CPU cores.

    #load dataset from torchvision

    torchvision.datasets.CIFAR10.url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    trainset = torchvision.datasets.CIFAR10(root= './data' , train = True,
                                            download= True , transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size ,
                                              shuffle=True , num_workers = num_workers)

    testset = torchvision.datasets.CIFAR10(root= './data' , train = False ,
                                           download= True, transform = transform)

    testloader = torch.utils.data.DataLoader(testset , batch_size = batch_size,
                                             shuffle=True , num_workers = num_workers)

    #defining our classes
    classes = ('plane' , 'car' , 'bird', 'cat' , 'deer', 'dog' , 'frog', 'horse', 'ship', 'truck')
    #visualizing images to get an idea about dataset

    def visualize():
        def imshow(image):
            image = image/2 + 0.5 #unnormalize image for better quality
            numpy_image = image.numpy()
            plt.imshow(numpy.transpose(numpy_image , (1,2,0)))
            plt.show()

        dataIter = iter(trainloader)
        images , labels = next(iter(dataIter))

        print(' '.join("%s" % classes[labels[j]] for j in range(batch_size)))
        imshow(torchvision.utils.make_grid(images))

    #visualize()

    #Step-2 define the convolution network

    class Net(nn.Module):
        '''for a simple convolution neural network'''

        def __init__(self):
            super(Net , self).__init__()
            self.conv1 = nn.Conv2d(3,6,5) #3 input, 6 output, 5*5 square convolution kernal
            #3 input because RGB, #6 output features, 5*5 pixel size or convolution kernal area.
            self.pool = nn.MaxPool2d(2,2)
            #making out 5*5 in to 2*2 by selecting max value from each square pixel. This gives us a little head room while
            #working with not comprosing with features.
            self.conv2 = nn.Conv2d(6, 16 , 5)
            #this will take input from conv1 layer which gives output of 6
            self.fc1 = nn.Linear(16*5*5 , 120)
            self.fc2 = nn.Linear(120 , 84)
            self.fc3 = nn.Linear(84 , 10)
            #fc means fully connected layers. I used Linear transformation which is g(Wx+b)
        def forward(self , x):
            '''the forward propagation algorithm'''
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1 , 16* 5 * 5)
            #view is used to reshape tensors. -1 is because is used because we dont know the number of rows.
            # 16*5*5 are the number of columns. This helps us flatten the output. We can also use 'x.flatten(1)'
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            #I used Rectified linear unit as activation function.
            x = self.fc3(x)

            return x

    net = Net()
    print(net)

#step-3 define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)