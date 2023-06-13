# **Drone Detection** 

## **Tutorial2**

The focus of this tutorial is on using the PyTorch API for common deep learning model development tasks.

### **How to Install PyTorch**

Before installing PyTorch, ensure that you have Python installed, such as Python 3.6 or higher. The most common, and perhaps simplest, way to install PyTorch on your workstation is by using pip.

For example, on the command line, you can type:
```
sudo pip install torch
```
Perhaps the most popular application of deep learning is for computer vision, and the PyTorch computer vision package is called “torchvision.”

Installing torchvision is also highly recommended and it can be installed as follows:
```
sudo pip install torchvision
```

### **PyTorch Deep Learning Model Life-Cycle**

A model has a life-cycle, and this very simple knowledge provides the backbone for both modeling a dataset and understanding the PyTorch API.

The five steps in the life-cycle are as follows:

1. Prepare the Data
2. Define the Model
3. Train the Model
4. Evaluate the Model
5. Make Predictions

### Step1: Prepare the Data

The first step is to load and prepare your data.

Neural network models require numerical input data and numerical output data.
PyTorch provides the Dataset class that you can extend and customize to load your dataset.
Once loaded, PyTorch provides the DataLoader class to navigate a Dataset instance during the training and evaluation of your model.

A DataLoader instance can be created for the training dataset, test dataset, and even a validation dataset.

The `random_split()` function can be used to split a dataset into train and test sets. Once split, a selection of rows from the Dataset can be provided to a DataLoader, along with the batch size and whether the data should be shuffled every epoch.

### Step2: Define the Model
The next step is to define a model.The idiom for defining a model in PyTorch involves defining a class that extends the Module class.

The constructor of your class defines the layers of the model and the forward() function is the override that defines how to forward propagate input through the defined layers of the model.

### Step3: Train the Model
The training process requires that you define a loss function and an optimization algorithm. 
Training the model involves enumerating the DataLoader for the training dataset.

First, a loop is required for the number of training epochs. Then an inner loop is required for the mini-batches for stochastic gradient descent.

Each update to the model involves the same general pattern comprised of:
* Clearing the last error gradient.
* A forward pass of the input through the model.
* Calculating the loss for the model output.
* Backpropagating the error through the model.
* Update the model in an effort to reduce loss.

### Step4: Evaluate the model

Once the model is fit, it can be evaluated on the test dataset.

This can be achieved by using the DataLoader for the test dataset and collecting the predictions for the test set, then comparing the predictions to the expected values of the test set and calculating a performance metric.

### Step5: Make predictions
A fit model can be used to make a prediction on new data.

For example, you might have a single image or a single row of data and want to make a prediction.

<br>


### **Implementation**
We used the Ionosphere binary (two class) classification dataset to demonstrate an MLP for binary classification.
This dataset involves predicting whether there is a structure in the atmosphere or not given radar returns.

The dataset will be downloaded automatically using Pandas, but you can learn more about it here.

* [Ionosphere Dataset (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv)
* [Ionosphere Dataset Description](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.names)

We used a [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) to encode the string labels to integer values 0 and 1. The model will be fit on 67 percent of the data, and the remaining 33 percent will be used for evaluation, split using the [train_test_split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function.

The complete example containing all the steps is included in the `MLP_for_binary_classification(Ionosphere).py` file. The file can be run in the corresponding directory with the command:
```
python MLP_for_binary_classification\(Ionosphere\).py
```
Running the example first reports the shape of the train and test datasets, then fits the model and evaluates it on the test dataset. Finally, a prediction is made for a single row of data.

An output like this can be expected:
```
235 116
Accuracy: 0.931
Predicted: 1.000 (class=1)
```

<br>

### **Other Elements in the Tutorial2 folder**
<br>

The file `multilayer_perceptron_model.py` includes a simple model example. 

The file can be run in the corresponding directory with the command:
```
python multilayer_perceptron_model.py
```
Output:
```
Sequential(
  (0): Linear(in_features=8, out_features=12, bias=True)
  (1): ReLU()
  (2): Linear(in_features=12, out_features=20, bias=True)
  (3): ReLU()
  (4): Linear(in_features=20, out_features=8, bias=True)
  (5): ReLU()
  (6): Linear(in_features=8, out_features=1, bias=True)
  (7): Sigmoid()
)
```
<br>

The file `training_model.py` includes a simple model example. The for-loop is to get a batch of data and feed into the model.  the model’s output and calculate the loss function. Based on the loss function, the optimizer will fine-tune the model for one step, so it can match better to the training data. 

The file can be run in the corresponding directory with the command:
```
python training_model.py
```

Output:
 ```
Finished epoch 0, latest loss 0.588100016117096
Finished epoch 1, latest loss 0.5735946297645569
Finished epoch 2, latest loss 0.5666378736495972
Finished epoch 3, latest loss 0.553726851940155
Finished epoch 4, latest loss 0.539828360080719
Finished epoch 5, latest loss 0.5264602899551392
Finished epoch 6, latest loss 0.5032461881637573
Finished epoch 7, latest loss 0.48828762769699097
Finished epoch 8, latest loss 0.4559307396411896
Finished epoch 9, latest loss 0.47892749309539795
Finished epoch 10, latest loss 0.4608011245727539
Finished epoch 11, latest loss 0.4507720172405243
Finished epoch 12, latest loss 0.4314222037792206
Finished epoch 13, latest loss 0.4552513360977173
Finished epoch 14, latest loss 0.4473949372768402
Finished epoch 15, latest loss 0.45897677540779114
Finished epoch 16, latest loss 0.43454891443252563
Finished epoch 17, latest loss 0.44270092248916626
Finished epoch 18, latest loss 0.45161205530166626
Finished epoch 19, latest loss 0.45822471380233765
Finished epoch 20, latest loss 0.4451013505458832
Finished epoch 21, latest loss 0.4551088809967041
Finished epoch 22, latest loss 0.45340487360954285
Finished epoch 23, latest loss 0.46186739206314087
Finished epoch 24, latest loss 0.4533328115940094
Finished epoch 25, latest loss 0.4497056007385254
Finished epoch 26, latest loss 0.4420396387577057
Finished epoch 27, latest loss 0.4501945674419403
Finished epoch 28, latest loss 0.43279340863227844
Finished epoch 29, latest loss 0.4181441366672516
Finished epoch 30, latest loss 0.4197627604007721
Finished epoch 31, latest loss 0.4104621410369873
Finished epoch 32, latest loss 0.4067235589027405
Finished epoch 33, latest loss 0.40461334586143494
Finished epoch 34, latest loss 0.3968975841999054
Finished epoch 35, latest loss 0.3886372447013855
Finished epoch 36, latest loss 0.38487792015075684
Finished epoch 37, latest loss 0.37768614292144775
Finished epoch 38, latest loss 0.37914642691612244
Finished epoch 39, latest loss 0.3789803385734558
Finished epoch 40, latest loss 0.365925133228302
Finished epoch 41, latest loss 0.37076759338378906
Finished epoch 42, latest loss 0.3648814558982849
Finished epoch 43, latest loss 0.35842764377593994
Finished epoch 44, latest loss 0.36925649642944336
Finished epoch 45, latest loss 0.35948944091796875
Finished epoch 46, latest loss 0.3634781837463379
Finished epoch 47, latest loss 0.3578653633594513
Finished epoch 48, latest loss 0.36215946078300476
Finished epoch 49, latest loss 0.3545094132423401
Finished epoch 50, latest loss 0.36372455954551697
Finished epoch 51, latest loss 0.3520985245704651
Finished epoch 52, latest loss 0.3544931411743164
Finished epoch 53, latest loss 0.36261212825775146
Finished epoch 54, latest loss 0.3484041094779968
Finished epoch 55, latest loss 0.35019978880882263
Finished epoch 56, latest loss 0.35184431076049805
Finished epoch 57, latest loss 0.3540416955947876
Finished epoch 58, latest loss 0.351951003074646
Finished epoch 59, latest loss 0.3487204313278198
Finished epoch 60, latest loss 0.35395774245262146
Finished epoch 61, latest loss 0.34952613711357117
Finished epoch 62, latest loss 0.3424471318721771
Finished epoch 63, latest loss 0.35338470339775085
Finished epoch 64, latest loss 0.34238430857658386
Finished epoch 65, latest loss 0.3371056616306305
Finished epoch 66, latest loss 0.3432745337486267
Finished epoch 67, latest loss 0.3477122187614441
Finished epoch 68, latest loss 0.3540416955947876
Finished epoch 69, latest loss 0.3423873782157898
Finished epoch 70, latest loss 0.3469897508621216
Finished epoch 71, latest loss 0.3473292887210846
Finished epoch 72, latest loss 0.3411523401737213
Finished epoch 73, latest loss 0.3558720052242279
Finished epoch 74, latest loss 0.3431950509548187
Finished epoch 75, latest loss 0.3480549156665802
Finished epoch 76, latest loss 0.33909857273101807
Finished epoch 77, latest loss 0.35868287086486816
Finished epoch 78, latest loss 0.34689202904701233
Finished epoch 79, latest loss 0.34170296788215637
Finished epoch 80, latest loss 0.3504983186721802
Finished epoch 81, latest loss 0.34609729051589966
Finished epoch 82, latest loss 0.35251593589782715
Finished epoch 83, latest loss 0.3425249457359314
Finished epoch 84, latest loss 0.3500131368637085
Finished epoch 85, latest loss 0.3504064977169037
Finished epoch 86, latest loss 0.3566044867038727
Finished epoch 87, latest loss 0.34742915630340576
Finished epoch 88, latest loss 0.3429732322692871
Finished epoch 89, latest loss 0.34089985489845276
Finished epoch 90, latest loss 0.34332263469696045
Finished epoch 91, latest loss 0.34731173515319824
Finished epoch 92, latest loss 0.3442865014076233
Finished epoch 93, latest loss 0.3488966226577759
Finished epoch 94, latest loss 0.34673425555229187
Finished epoch 95, latest loss 0.3404946029186249
Finished epoch 96, latest loss 0.33688104152679443
Finished epoch 97, latest loss 0.34885433316230774
Finished epoch 98, latest loss 0.3476221263408661
Finished epoch 99, latest loss 0.33356034755706787
 ```

<br>

The file `CNN.py` includes a simple example model. 
Output:
```
Sequential(
  (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): Dropout(p=0.3, inplace=False)
  (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): ReLU()
  (5): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (6): Flatten(start_dim=1, end_dim=-1)
  (7): Linear(in_features=8192, out_features=512, bias=True)
  (8): ReLU()
  (9): Dropout(p=0.5, inplace=False)
  (10): Linear(in_features=512, out_features=10, bias=True)
)
```

<br>

The file `loading_data.py` includes an example related to loading data from torchvision. The dataset that is used is CIFAR-10. It is a dataset of 10 different objects. There is a larger dataset called CIFAR-100, too.

The `torchvision.datasets.CIFAR10` function helps you to download the CIFAR-10 dataset to a local directory. The dataset is divided into training set and test set. You can plot the first 24 images from the downloaded dataset as below. Each image in the dataset is 32×32 pixels picture of one of the following: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.

The file can be run in the corresponding directory with the command:
```
python loading_data.py
```

**Note**: There are 2 types of loading data in the file loading_data.py. So, one of these types is in the comment. You can remove the comment lines for these outputs. 

First output:

![](/tutorial2/CIFAR10.png)

The CIFAR-10 image from the previous example is indeed in the format of numpy array. But for consumption by a PyTorch model, it needs to be in PyTorch tensors. The PyTorch DataLoader can help you make this process smoother.

In the second code snippet, `trainset` is created with `transform` argument so that the data is converted into PyTorch tensor when it is extracted. This is performed in `DataLoader` the lines following it. The `DataLoader` object is a Python iterable, which you can extract the input (which are images) and target (which are integer class labels).

Second outputs:
![](/tutorial2/CIFAR10-2-traindata.png)

![](/tutorial2/CIFAR10-3-testdata.png)

<br>

We can say that `CNN.py` and `loading_data.py` are pre-prepare for the file `training_an_image_classifier.py`. The file `training_an_image_classifier.py` containing these code snippet is created.

The file can be run in the corresponding directory with the command:
```
python training_an_image_classifier.py
```

Output:

```
Files already downloaded and verified
Files already downloaded and verified
Epoch 0: model accuracy 38.47%
Epoch 1: model accuracy 45.87%
Epoch 2: model accuracy 48.87%
Epoch 3: model accuracy 52.98%
Epoch 4: model accuracy 54.50%
Epoch 5: model accuracy 56.49%
Epoch 6: model accuracy 58.19%
Epoch 7: model accuracy 59.07%
Epoch 8: model accuracy 60.95%
Epoch 9: model accuracy 60.90%
Epoch 10: model accuracy 62.95%
Epoch 11: model accuracy 63.23%
Epoch 12: model accuracy 64.52%
Epoch 13: model accuracy 64.80%
Epoch 14: model accuracy 65.34%
Epoch 15: model accuracy 65.78%
Epoch 16: model accuracy 66.10%
Epoch 17: model accuracy 66.55%
Epoch 18: model accuracy 67.45%
Epoch 19: model accuracy 67.54%
```

**Note**: We see "Files already downloaded and verified" since I downloaded the dataset before. You can see a different state. The source I completed the tutorial indicate that you should see the model produced can achieve no less than 70% accuracy. However, my output can achieve 67.54% accuracy. The reason of that may be that I train the model a few time with the same dataset. 

## **Tutorial3**

This tutorial aims to show the difference between CNN and fully connected NN, to understand the structure of convolution, and to examine the effects of the filters on convolution. 

 In the first steps of `CNN.py`, the image to be processed throughout the code has been read and drawn. The drawn image:

 ![tom-bw.png](/tutorial3/tom-bw.png)


If we run the code snippets in the comment lines respectively, the below outputs are obtained. The file `plotting.py`includes the functions for plotting convolution applied images. 

The file can be run in the corresponding directory with the command:
```
python CNN.py
```

First output:

![](/tutorial3/plot_conv_1.png)

Second output:

![](/tutorial3/plot_conv_2.png)

Third output:

![](/tutorial3/plot_conv_3.png)

When different number and size of filters are used:

First output:
![](/tutorial3/plot_convs_1.png)

Second output:
![](/tutorial3/plot_convs_2.png)

Third output:
![](/tutorial3/plot_convs_3.png)

If we use a kernel with no padding, our output image will be smaller. If we show by using larger kernel:

![](/tutorial3/plot_convs_4.png)

If we add padding to the outside of the image to avoid this:

![](/tutorial3/plot_convs_5.png)

Finally, if we want to saw how strides influence the size of the output:
![](/tutorial3/plot_convs_6.png)

When a simple CNN model in PyTorch is made, the below outputs are obtained. (You should run the appropriate part within the code.)

**Note:** The reason why a large part of the code is included in the comment line is that I want it not to reproduce the previous outputs every time I run the code.

Output:

 ```
 ==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 1]                   --
|    └─Conv2d: 2-1                       [-1, 3, 100, 100]         30
|    └─ReLU: 2-2                         [-1, 3, 100, 100]         --
|    └─Conv2d: 2-3                       [-1, 2, 100, 100]         56
|    └─ReLU: 2-4                         [-1, 2, 100, 100]         --
|    └─Flatten: 2-5                      [-1, 20000]               --
|    └─Linear: 2-6                       [-1, 1]                   20,001
==========================================================================================
Total params: 20,087
Trainable params: 20,087
Non-trainable params: 0
Total mult-adds (M): 0.85
==========================================================================================
Input size (MB): 0.04
Forward/backward pass size (MB): 0.38
Params size (MB): 0.08
Estimated Total Size (MB): 0.50
==========================================================================================
 ```

It seems that it has too many parameters.

The parameter numbers are reduced by using pooling layer. 

Output:

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 1]                   --
|    └─Conv2d: 2-1                       [-1, 3, 100, 100]         30
|    └─ReLU: 2-2                         [-1, 3, 100, 100]         --
|    └─MaxPool2d: 2-3                    [-1, 3, 50, 50]           --
|    └─Conv2d: 2-4                       [-1, 2, 50, 50]           56
|    └─ReLU: 2-5                         [-1, 2, 50, 50]           --
|    └─MaxPool2d: 2-6                    [-1, 2, 25, 25]           --
|    └─Flatten: 2-7                      [-1, 1250]                --
|    └─Linear: 2-8                       [-1, 1]                   1,251
==========================================================================================
Total params: 1,337
Trainable params: 1,337
Non-trainable params: 0
Total mult-adds (M): 0.41
==========================================================================================
Input size (MB): 0.04
Forward/backward pass size (MB): 0.27
Params size (MB): 0.01
Estimated Total Size (MB): 0.31
==========================================================================================
``` 
We reduced that last layer to 1,251 parameters. 

In the last part of the code, the number of parameters between CNN and fully connected NN is compared.

The output by using NN model:
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 1]                   --
|    └─Sequential: 2-1                   [-1, 256]                 --
|    |    └─Linear: 3-1                  [-1, 256]                 4,194,560
|    |    └─ReLU: 3-2                    [-1, 256]                 --
|    └─Sequential: 2-2                   [-1, 128]                 --
|    |    └─Linear: 3-3                  [-1, 128]                 32,896
|    |    └─ReLU: 3-4                    [-1, 128]                 --
|    └─Sequential: 2-3                   [-1, 64]                  --
|    |    └─Linear: 3-5                  [-1, 64]                  8,256
|    |    └─ReLU: 3-6                    [-1, 64]                  --
|    └─Sequential: 2-4                   [-1, 16]                  --
|    |    └─Linear: 3-7                  [-1, 16]                  1,040
|    |    └─ReLU: 3-8                    [-1, 16]                  --
|    └─Linear: 2-5                       [-1, 1]                   17
==========================================================================================
Total params: 4,236,769
Trainable params: 4,236,769
Non-trainable params: 0
Total mult-adds (M): 12.71
==========================================================================================
Input size (MB): 0.06
Forward/backward pass size (MB): 0.00
Params size (MB): 16.16
Estimated Total Size (MB): 16.23
==========================================================================================
```

The output by using CNN model:

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 1]                   --
|    └─Sequential: 2-1                   [-1, 3, 128, 128]         --
|    |    └─Conv2d: 3-1                  [-1, 3, 128, 128]         30
|    |    └─ReLU: 3-2                    [-1, 3, 128, 128]         --
|    └─Sequential: 2-2                   [-1, 3, 128, 128]         --
|    |    └─Conv2d: 3-3                  [-1, 3, 128, 128]         84
|    |    └─ReLU: 3-4                    [-1, 3, 128, 128]         --
|    └─Sequential: 2-3                   [-1, 3, 128, 128]         --
|    |    └─Conv2d: 3-5                  [-1, 3, 128, 128]         84
|    |    └─ReLU: 3-6                    [-1, 3, 128, 128]         --
|    └─Sequential: 2-4                   [-1, 3, 128, 128]         --
|    |    └─Conv2d: 3-7                  [-1, 3, 128, 128]         84
|    |    └─ReLU: 3-8                    [-1, 3, 128, 128]         --
|    └─Sequential: 2-5                   [-1, 3, 128, 128]         --
|    |    └─Conv2d: 3-9                  [-1, 3, 128, 128]         84
|    |    └─ReLU: 3-10                   [-1, 3, 128, 128]         --
|    └─Flatten: 2-6                      [-1, 49152]               --
|    └─Linear: 2-7                       [-1, 1]                   49,153
==========================================================================================
Total params: 49,519
Trainable params: 49,519
Non-trainable params: 0
Total mult-adds (M): 5.85
==========================================================================================
Input size (MB): 0.06
Forward/backward pass size (MB): 1.88
Params size (MB): 0.19
Estimated Total Size (MB): 2.13
==========================================================================================
```


## **Tutorial5 (Detector)**


The main point of the tutorial is to create a YOLO(v3) object detector structure. First I created a folder called tutorial5_detector that has my pyhton files and other files related to the tutorial. Then, I created a file called darknet.py in my detector folder. Darknet is the name of the underlying architecture of YOLO. This file contains the code that creates the YOLO network. We supplemented it with a file util.py whic contains the code for various helper functions.

### **Configuration File**
The official code uses a configuration file to build the network.The cfg file describes the layout of the network, block by block. 

I used the official `cfg` file, released by the author to build my network. Download it from [here](https://github.com/elifbuyukorhan/Drone_Detection/tree/devel/tutorial5_detector/cfg) and place it in a folder called `cfg` inside your detector directory. If you're on Linux, `cd` into your network directory and type:

```
mkdir cfg
cd cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

If you open the configuration file, you will see something like this.
```
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear
```

### **Parsing the configuration file**
 
There are a function called parse_cfg which takes the path of the configuration file as the input inside darknet.py file. The idea here is to parse the cfg, and **store every block as a dict**. The attributes of the blocks and their values are stored as key-value pairs in the dictionary. As we parse through the cfg, we keep appending these dicts, denoted by the variable block in our code, to a list blocks. Our function will return this block.

### **Creating the building blocks**

We have 5 types of layers in the list (mentioned above). PyTorch provides pre-built layers for types `convolutional` and `upsample`. We will have to write our own modules for the rest of the layers by extending the nn.Module class.

The create_modules function takes a list blocks returned by the parse_cfg function. Before we iterate over list of blocks, we define a variable net_info to store information about the network.

You can find the detailed information about the modules [here](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/). 

### **Testing Code**

You can test the code by removing comment mark from in my code. The piece of code is located under the `create_modules` function.
```
blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))
```

You will see a long list, (exactly containing 106 items), the elements of which will look like

```
-
-

(9): Sequential(
    (conv_9): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batch_norm_9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leaky_9): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (10): Sequential(
    (conv_10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batch_norm_10): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leaky_10): LeakyReLU(negative_slope=0.1, inplace=True)
  )
  (11): Sequential(
    (shortcut_11): EmptyLayer()
  )

  -
  -

  ```

### **Defining The Network**

In this part of the tutorial, we are going to implement the network architecture of YOLO in PyTorch, so that we can produce an output given an image.

nn.Module class was used to build custom architectures in PyTorch. A network in the darknet.py was defined for our detector. 

The forward pass of the network is implemented by overriding the `forward` method of the `nn.Module` class.

`forward` serves two purposes. First, to calculate the output, and second, to transform the output detection feature maps in a way that it can be processed easier (such as transforming them such that detection maps across multiple scales can be concatenated, which otherwise isn't possible as they are of different dimensions).

`forward` takes three arguments, self, the input x and CUDA, which if true, would use GPU to accelerate the forward pass.

Here, we iterate over self.blocks[1:] instead of self.blocks since the first element of self.blocks is a net block which isn't a part of the forward pass.

Since route and shortcut layers need output maps from previous layers, we cache the output feature maps of every layer in a dict `outputs`. The keys are the the indices of the layers, and the values are the feature maps.

As was the case with `create_modules` function, we now iterate over `module_list` which contains the modules of the network. The thing to notice here is that the modules have been appended in the same order as they are present in the configuration file. This means, we can simply run our input through each module to get our output.

### **Transforming the output**

The outputs of YOLO layer happen at three scales, the dimensions of the prediction maps will be different. Although the dimensions of the three feature maps are different, the output processing operations to be done on them are similar. It would be nice to have to do these operations on a single tensor, rather than three separate tensors.

To remedy like these problems, the function `predict_transform` is used.

The function `predict_transform` lives in the file `util.py` and we imported the function when we used it in `forward` of `Darknet` class.

You can visit the link related to predict_transform function [here](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/). 

### **Testing the forward pass**

There is a function that creates a input called get_test_input at the top of darknet.py. We will pass this input to our network. You can save this [image](https://raw.githubusercontent.com/elifbuyukorhan/Drone_Detection/devel/tutorial5_detector/dog-cycle-car.png?token=GHSAT0AAAAAACDEMPNPMVUIW3LRTXYGLU3KZEDED6A) into your working directory . If you're on linux, then type.

 ```
 wget https://github.com/elifbuyukorhan/Drone_Detection/blob/devel/tutorial5_detector/dog-cycle-car.png
 ```
To test the model up to the mentioned part, you can remove comment marks the following lines of code at the bottom of the darknet.py file.

```
model = Darknet("cfg/yolov3.cfg")
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print (pred)
```

You will see an output like.
```
tensor([[[1.6452e+01, 1.6508e+01, 1.6878e+02,  ..., 5.2611e-01,
          4.8999e-01, 4.7966e-01],
         [1.7080e+01, 1.8504e+01, 1.2612e+02,  ..., 6.0921e-01,
          3.9676e-01, 4.2774e-01],
         [1.3747e+01, 1.3307e+01, 3.2321e+02,  ..., 5.7942e-01,
          4.5787e-01, 5.1598e-01],
         ...,
         [6.0463e+02, 6.0388e+02, 9.9459e+00,  ..., 5.3082e-01,
          3.9327e-01, 4.9988e-01],
         [6.0477e+02, 6.0388e+02, 1.3831e+01,  ..., 4.9478e-01,
          4.3924e-01, 5.0030e-01],
         [6.0318e+02, 6.0463e+02, 3.1248e+01,  ..., 4.9105e-01,
          4.9707e-01, 4.8182e-01]]], device='cuda:0')
```

The shape of this tensor is `1 x 10647 x 85`. The first dimension is the batch size which is simply 1 because we have used a single image. For each image in a batch, we have a 10647 x 85 table. The row of each of this table represents a bounding box. (4 bbox attributes, 1 objectness score, and 80 class scores)

At this point, our network has random weights, and will not produce the correct output. We need to load a weight file in our network. We'll be making use of the official weight file for this purpose.

### **Downloading the Pre-trained Weights**

Download the weights file into your detector directory. Grab the weights file from [here](https://pjreddie.com/media/files/yolov3.weights?ref=blog.paperspace.com). Or if you're on linux,

```
wget https://pjreddie.com/media/files/yolov3.weights
```
### **Loading Weights**

A function called load_weights that is a member function of the `Darknet` class takes one argument other than `self, the path of the weightsfile. 

We completed this section with the load_weights function. You can now load weights in your `Darknet` object by calling the `load_weights` function on the darknet object.

For this again you can remove the relevant ones from the comment marks at the bottom of the darknet.py file.

```
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
```

We must subject our output to objectness score thresholding and Non-maximal suppression, to obtain what I will call in the rest of this post as the true detections.  To do that, we will create some functions in the file `util.py`. You can examine these functions from the file `util.py`. I will not explain too much since this part does not need to be tested. You can find the detailed information [here](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-4/). 

We have built a model which outputs several object detections given an input image. To be precise, our output is a tensor of shape B x 10647 x 85. B is the number of images in a batch, 10647 is the number of bounding boxes predicted per image, and 85 is the number of bounding box attributes.

However, we must subject our output to objectness score thresholding and Non-maximal suppression, to obtain what as the true detections is called, we created a function called `write_results` in the file `util.py`.

The functions takes as as input the `prediction`, `confidence` (objectness score threshold), `num_classes` (80, in our case) and `nms_conf` (the NMS IoU threshold).

Also, there is a function called `bbox_iou` for calculating the IoU in the file `util.py`.

The file detect.py created for tour detector. Neccasary imports done at top of the file detect.py. 

### **Creating Command Line Arguments**

Since `detect.py` is the file that we will execute to run our detector, it's nice to have command line arguments we can pass to it. I've used python's `ArgParse` module to do that.
Important flags are `images` (used to specify the input image or directory of images), `det` (Directory to save detections to), `reso` (Input image's resolution, can be used for speed - accuracy tradeoff), `cfg` (alternative configuration file) and `weightfile`.

### **Loading the Network**

Download the file coco.names from [here](https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/data/coco.names?ref=blog.paperspace.com), a file that contains the names of the objects in the COCO dataset. Create a folder `data` in your detector directory. Equivalently, if you're on linux you can type.
```
mkdir data
cd data
wget https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/data/coco.names
```

The network and load weights are initialized inside the file `detect.py`.

### Reading the input images

The image from the disk, or the images from a directory are read. The paths of the image/images are stored in a list called `imlist`.

`read_dir` is a checkpoint used to measure time. 

We use OpenCV to load the images.

OpenCV loads an image as an numpy array, with BGR as the order of the color channels. PyTorch's image input format is (Batches x Channels x Height x Width), with the channel order being RGB. Therefore, we write the function `prep_image` in `util.py` to transform the numpy array into PyTorch's input format.

we must use a function `letterbox_image` that resizes our image, keeping the aspect ratio consistent, and padding the left out areas with the color (128,128,128). We use the function that takes a OpenCV images and converts it to the input of our network.

### **Printing Time Summary**

At the end of our detector we will print a summary containing which part of the code took how long to execute. This is useful when we have to compare how different hyperparameters effect the speed of the detector. Hyperparameters such as batch size, objectness confidence and NMS threshold, (passed with `bs`, `confidence`, `nms_thresh` flags respectively) can be set while executing the script `detect.py` on the command line.

### Testing The Object Detector

For example, running on terminal,
```
python detect.py --images dog-cycle-car.png --det det
```
produces the output
```
Loading network.....
Network successfully loaded
dog-cycle-car.png    predicted in  1.265 seconds
Objects Detected:    bicycle truck dog
----------------------------------------------------------
SUMMARY
----------------------------------------------------------
Task                     : Time Taken (in seconds)

Reading addresses        : 0.000
Loading batch            : 0.022
Detection (1 images)     : 1.267
Output Processing        : 0.000
Drawing Boxes            : 0.062
Average time_per_img     : 1.351
----------------------------------------------------------
```
An image with name `det_dog-cycle-car.png` is saved in the `det` directory.

![](/tutorial5_detector/det/det_dog-cycle-car.png)