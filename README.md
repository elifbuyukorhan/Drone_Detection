# Drone Detection 


## Tutorial5 (Detector)


The main point of the tutorial is to create a YOLO(v3) object detector structure. First I created a folder called tutorial5_detector that has my pyhton files and other files related to the tutorial. Then, I created a file called darknet.py in my detector folder. Darknet is the name of the underlying architecture of YOLO. This file contains the code that creates the YOLO network. We supplemented it with a file util.py whic contains the code for various helper functions.

### Configuration File
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

### Parsing the configuration file
 
There are a function called parse_cfg which takes the path of the configuration file as the input inside darknet.py file. The idea here is to parse the cfg, and **store every block as a dict**. The attributes of the blocks and their values are stored as key-value pairs in the dictionary. As we parse through the cfg, we keep appending these dicts, denoted by the variable block in our code, to a list blocks. Our function will return this block.

### Creating the building blocks

We have 5 types of layers in the list (mentioned above). PyTorch provides pre-built layers for types `convolutional` and `upsample`. We will have to write our own modules for the rest of the layers by extending the nn.Module class.

The create_modules function takes a list blocks returned by the parse_cfg function. Before we iterate over list of blocks, we define a variable net_info to store information about the network.

You can find the detailed information about the modules [here](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/). 

### Testing Code

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

### Defining The Network

In this part of the tutorial, we are going to implement the network architecture of YOLO in PyTorch, so that we can produce an output given an image.

nn.Module class was used to build custom architectures in PyTorch. A network in the darknet.py was defined for our detector. 

The forward pass of the network is implemented by overriding the `forward` method of the `nn.Module` class.

`forward` serves two purposes. First, to calculate the output, and second, to transform the output detection feature maps in a way that it can be processed easier (such as transforming them such that detection maps across multiple scales can be concatenated, which otherwise isn't possible as they are of different dimensions).

`forward` takes three arguments, self, the input x and CUDA, which if true, would use GPU to accelerate the forward pass.

Here, we iterate over self.blocks[1:] instead of self.blocks since the first element of self.blocks is a net block which isn't a part of the forward pass.

Since route and shortcut layers need output maps from previous layers, we cache the output feature maps of every layer in a dict `outputs`. The keys are the the indices of the layers, and the values are the feature maps.

As was the case with `create_modules` function, we now iterate over `module_list` which contains the modules of the network. The thing to notice here is that the modules have been appended in the same order as they are present in the configuration file. This means, we can simply run our input through each module to get our output.

### Transforming the output

The outputs of YOLO layer happen at three scales, the dimensions of the prediction maps will be different. Although the dimensions of the three feature maps are different, the output processing operations to be done on them are similar. It would be nice to have to do these operations on a single tensor, rather than three separate tensors.

To remedy like these problems, the function `predict_transform` is used.

The function `predict_transform` lives in the file `util.py` and we imported the function when we used it in `forward` of `Darknet` class.

You can visit the link related to predict_transform function [here](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/). 

### Testing the forward pass

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

### Downloading the Pre-trained Weights

Download the weights file into your detector directory. Grab the weights file from [here](https://pjreddie.com/media/files/yolov3.weights?ref=blog.paperspace.com). Or if you're on linux,

```
wget https://pjreddie.com/media/files/yolov3.weights
```
### Loading Weights

A function called load_weights that is a member function of the `Darknet`class takes one argument other than `self, the path of the weightsfile. 

We completed this section with the load_weights function. You can now load weights in your `Darknet` object by calling the `load_weights` function on the darknet object.

For this again you can remove the relevant ones from the comment marks at the bottom of the darknet.py file.

```
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
```

We must subject our output to objectness score thresholding and Non-maximal suppression, to obtain what I will call in the rest of this post as the true detections.  To do that, we will create some functions in the file `util.py`. You can examine these functions from the file `util.py`. I will not explain too much since this part does not need to be tested. You can find the detailed information [here](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-4/). 

