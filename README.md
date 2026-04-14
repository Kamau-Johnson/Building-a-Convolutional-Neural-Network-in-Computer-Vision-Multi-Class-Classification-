# Building-a-Convolutional-Neural-Network-in-Computer-Vision-Multi-Class-Classification-
## 2.3 Multiclass Classification
<p float="left"> 
  <img src="cnn1.png" width="30%" /> 
  <img src="cnn2.png" width="30%" /> 
  <img src="cnn3.png" width="30%" /> 
</p

```
Getting Ready
```
As usual, there are a few things we need to do before we can begin. We'll start by importing the packages we'll need.


```python
import os

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms
from tqdm import tqdm
```

Next, let's also print out the version numbers for our libraries as well as the Python version. This makes our analysis reproducible for anyone who wants to review or reuse our work.


```python
print("torch version : ", torch.__version__)
print("torchvision version : ", torchvision.__version__)
print("torchinfo version : ", torchinfo.__version__)
print("numpy version : ", np.__version__)
print("matplotlib version : ", matplotlib.__version__)
print("PIL version : ", PIL.__version__)

!python --version
```

    torch version :  2.2.2+cu121
    torchvision version :  0.17.2+cu121
    torchinfo version :  1.8.0
    numpy version :  1.26.3
    matplotlib version :  3.9.2
    PIL version :  10.2.0
    Python 3.11.0


As we've done in past lessons, we'll also check if GPUs are available. Remember that some computers come with GPUs, which allow for bigger and faster model training. The `cuda` package is used to access GPUs on Linux and Windows machines in PyTorch; `mps` is used on Macs. 

We'll use the `device` variable later to set the location of our data and model.


```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device} device.")
```

    Using cuda device.


### Exploring and Preparing Our Data

We'll work with images of crop disease from Uganda which we prepared in the previous lesson. You may remember that we created an undersampled dataset that has a uniform distribution across classes. Let's use that dataset.

The data is in the `data_p2` folder within which is the `data_undersampled` folder. In that folder we have the `train` folder that contains the training data.

**Task 2.3.1:** Assign `train_dir` the path to the training data. Follow the pattern of `data_dir`.


```python
data_dir = os.path.join("data_p2", "data_undersampled")
train_dir = os.path.join(data_dir, "train")

print("Data Directory:", data_dir)
print("Training Data Directory:", train_dir)
```

    Data Directory: data_p2/data_undersampled
    Training Data Directory: data_p2/data_undersampled/train


Next let's check what classes we have in the data. Images from each class are contained in a separate subdirectory in `train_dir` where the name of each subdirectory is the name of the class.

**Task 2.3.2:** Create a list of class names using `os.listdir`.


```python
classes = os.listdir(train_dir)

print("List of classes:", classes)
```

    List of classes: ['cassava-healthy', 'cassava-mosaic-disease-cmd', 'cassava-brown-streak-disease-cbsd', 'cassava-green-mottle-cgm', 'cassava-bacterial-blight-cbb']


Following what we did in the previous lesson to standardize the images, we'll again use the same set of transformations:

- Convert any grayscale images to RGB format with a custom class
- Resize the image, so that they're all the same size (we chose $224$ x $224$)
- Convert the image to a Tensor of pixel values
- Normalize the data (we normalize each color channel separately)

Here's the custom transformation that we've used before which converts images to RGB format:


```python
class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
```

Now we'll use `transforms.Compose` from `torchvision` package to compose our pipeline of transformations.

**Task 2.3.3:** Complete the transformation pipeline below. It's missing the last two steps (converting images to PyTorch tensors and normalizing them). In the normalization step, make sure to use the `mean` and `std` values from the previous lesson.


```python
# Define transformation to apply to the images
transform_normalized = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        # Convert images to tensors
        transforms.ToTensor(),
        # Normalize the tensors (copy the mean and std from previous lesson!)
        transforms.Normalize(
            mean=[0.4326, 0.4953, 0.3120], std=[0.2178, 0.2214, 0.2091]
        )
    ]
)

print(type(transform_normalized))
print("-----------------")
print(transform_normalized)
```

    <class 'torchvision.transforms.transforms.Compose'>
    -----------------
    Compose(
        <__main__.ConvertToRGB object at 0x777019e68b50>
        Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True)
        ToTensor()
        Normalize(mean=[0.4326, 0.4953, 0.312], std=[0.2178, 0.2214, 0.2091])
    )


We are now ready to create our dataset with our transformations.

**Task 2.3.4:** Make a normalized dataset using `ImageFolder` from `datasets` and print the length of the dataset.


```python
dataset = datasets.ImageFolder(root=train_dir, transform=transform_normalized)

print('Length of dataset:', len(dataset))
```

    Length of dataset: 7615


### Train and validation splitting

We'll follow good practice and divide our data into two parts. One part will be the data we'll train our model on. The second part will be used to evaluate the model on images it hasn't seen in training.

This is an important step in order for us to check how good the model is. If it makes good predictions on the training data but not on the validation data, we'll know the model's overfit.

**Task 2.3.5:** Use `random_split` to create a 80/20 split (training dataset should have 80% of the data, validation dataset should have 20% of the data).

<div class="alert alert-info" role="alert">
    <p><b>About random number generators</b></p>
<p>The following cell adds a <code>generator=g</code> line of code that is not present in the video. This is something we have added to make sure you always get the same results in your predictions. Please don't change it or remove it.
</p>
</div>


```python
# Important, don't change this!
g = torch.Generator()
g.manual_seed(42)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=g)

print("Length of training dataset:", len(train_dataset))
print("Length of validation dataset:", len(val_dataset))
```

    Length of training dataset: 6092
    Length of validation dataset: 1523


Now let's make sure that the training data indeed contains 80% of the dataset and the validation set 20%.

**Task 2.3.6:** Compute the length of the entire dataset, the training dataset and the validation dataset. We've added the code that computes the percentage of data that's training data and percentage that's validation.


```python
length_dataset = len(dataset)
length_train = len(train_dataset)
length_val = len(val_dataset)

percent_train = np.round(100 * length_train / length_dataset, 2)
percent_val = np.round(100 * length_val / length_dataset, 2)

print(f"Train data is {percent_train}% of full data")
print(f"Validation data is {percent_val}% of full data")
```

    Train data is 80.0% of full data
    Validation data is 20.0% of full data


We're also curious about the breakdown of the classes. We're using the dataset that we prepared in such a way that all classes have the same number of images. But let's make sure that this is indeed true! 

We'll reuse the `class_count` function that we can import from `training.py`. The function goes through a dataset and counts how many images are in each class.

**Task 2.3.7:** Use `class_counts` function on the entire dataset and visualize the results with a bar chart. Note that computing `dataset_counts` may take a long time.


```python
from training import class_counts

dataset_counts = class_counts(dataset)

# Make a bar chart from the function output
dataset_counts.sort_values().plot(kind='bar')
# Add axis labels and title
plt.xlabel("Class Label")
plt.ylabel("Frequency [count]")
plt.title("Distribution of Classes in Entire Dataset");
```


      0%|          | 0/7615 [00:00<?, ?it/s]



    
![png](output_33_1.png)
    


**Task 2.3.8:** Use the `class_counts` function and pandas plotting to make the same plot for the training data.


```python
train_counts = class_counts(train_dataset)

# Make a bar chart from the function output
train_counts.sort_values().plot(kind='bar')

# Add axis labels and title
plt.xlabel("Class Label")
plt.ylabel("Frequency [count]")
plt.title("Distribution of Classes in Training Dataset");
```


      0%|          | 0/6092 [00:00<?, ?it/s]



    
![png](output_35_1.png)
    


Let's make the same plot, but this time for the validation data.

**Task 2.3.9:** Use the `class_counts` function and pandas plotting to get the breakdown across classes for the validation split.


```python
val_counts = class_counts(val_dataset)

# Make a bar chart from the function output
val_counts.sort_values().plot(kind='bar')

# Add axis labels and title
plt.xlabel("Class Label")
plt.ylabel("Frequency [count]")
plt.title("Distribution of Classes in Validation Dataset");
```


      0%|          | 0/1523 [00:00<?, ?it/s]



    
![png](output_38_1.png)
    


From these visualizations, we see that we indeed have a roughly uniform distribution across classes in the entire dataset as well as in the training and validation splits. Some deviations from a uniform distribution are normal given that we're working with a fairly small number of observations when it comes to statistics.

Well done! We're now ready to create `DataLoader` objects. We'll use a batch size of 32 and start with the `DataLoader` for training. Remember that in training we want to shuffle the data after each epoch.

<div class="alert alert-info" role="alert">
<p>Curious to learn more about data loaders (or any other class or function)? You can follow up the object name with a question mark as shown below. After running the code cell, the documentation for that object will be displayed for you.</p>

<p>Another way to bring up the documentation in Jupyter is to put your cursor at the end of object name and use shift+tab 🤓</p>
</div>


```python
DataLoader?
```


    [0;31mInit signature:[0m
    [0mDataLoader[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mdataset[0m[0;34m:[0m [0mtorch[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mdataset[0m[0;34m.[0m[0mDataset[0m[0;34m[[0m[0;34m+[0m[0mT_co[0m[0;34m][0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mbatch_size[0m[0;34m:[0m [0mOptional[0m[0;34m[[0m[0mint[0m[0;34m][0m [0;34m=[0m [0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mshuffle[0m[0;34m:[0m [0mOptional[0m[0;34m[[0m[0mbool[0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0msampler[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mtorch[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0msampler[0m[0;34m.[0m[0mSampler[0m[0;34m,[0m [0mIterable[0m[0;34m,[0m [0mNoneType[0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mbatch_sampler[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mtorch[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0msampler[0m[0;34m.[0m[0mSampler[0m[0;34m[[0m[0mList[0m[0;34m][0m[0;34m,[0m [0mIterable[0m[0;34m[[0m[0mList[0m[0;34m][0m[0;34m,[0m [0mNoneType[0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mnum_workers[0m[0;34m:[0m [0mint[0m [0;34m=[0m [0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcollate_fn[0m[0;34m:[0m [0mOptional[0m[0;34m[[0m[0mCallable[0m[0;34m[[0m[0;34m[[0m[0mList[0m[0;34m[[0m[0;34m~[0m[0mT[0m[0;34m][0m[0;34m][0m[0;34m,[0m [0mAny[0m[0;34m][0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpin_memory[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdrop_last[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mtimeout[0m[0;34m:[0m [0mfloat[0m [0;34m=[0m [0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mworker_init_fn[0m[0;34m:[0m [0mOptional[0m[0;34m[[0m[0mCallable[0m[0;34m[[0m[0;34m[[0m[0mint[0m[0;34m][0m[0;34m,[0m [0mNoneType[0m[0;34m][0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmultiprocessing_context[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mgenerator[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m*[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mprefetch_factor[0m[0;34m:[0m [0mOptional[0m[0;34m[[0m[0mint[0m[0;34m][0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpersistent_workers[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpin_memory_device[0m[0;34m:[0m [0mstr[0m [0;34m=[0m [0;34m''[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m     
    Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
    
    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.
    
    See :py:mod:`torch.utils.data` documentation page for more details.
    
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (Callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (Callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        multiprocessing_context (str or multiprocessing.context.BaseContext, optional): If
            ``None``, the default `multiprocessing context`_ of your operating system will
            be used. (default: ``None``)
        generator (torch.Generator, optional): If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            ``base_seed`` for workers. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers batches prefetched across all workers. (default value depends
            on the set value for num_workers. If value of num_workers=0 default is ``None``.
            Otherwise, if value of ``num_workers > 0`` default is ``2``).
        persistent_workers (bool, optional): If ``True``, the data loader will not shut down
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
        pin_memory_device (str, optional): the device to :attr:`pin_memory` to if ``pin_memory`` is
            ``True``.
    
    
    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.
    
    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.
    
                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.
    
                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.
    
    .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                 :ref:`data-loading-randomness` notes for random seed related questions.
    
    .. _multiprocessing context:
        https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    [0;31mFile:[0m           /usr/local/lib/python3.11/site-packages/torch/utils/data/dataloader.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m     


**Task 2.3.10:** Create the training loader. Make sure to set shuffling to be on.


```python
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(type(train_loader))
```

    <class 'torch.utils.data.dataloader.DataLoader'>


Next let's create a `DataLoader` for validation data.

**Task 2.3.11:** Create the validation loader. Make sure to set shuffling to be off.


```python
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(type(val_loader))
```

    <class 'torch.utils.data.dataloader.DataLoader'>


Next let's check the shape of a batch of images and labels. We'll use the `DataLoader` object and turn it into an iterator by using `iter`. Then with `next` we can fetch a batch of data.

We expect one batch of images to be a 4D tensor with dimension `[32, 3, 224, 224]` and one batch of labels to be a one dimensional tensor of length 32.

**Task 2.3.12:** Print the shape of a batch of images and the shape of a batch of labels.


```python
data_iter = iter(train_loader)
images, labels = next(data_iter)

image_shape = images.shape
print("Shape of batch of images", image_shape)

label_shape = labels.shape
print("Shape of batch of labels:", label_shape)
```

    Shape of batch of images torch.Size([32, 3, 224, 224])
    Shape of batch of labels: torch.Size([32])


Out of curiosity, let's examine the `labels` tensor.


```python
labels
```




    tensor([3, 0, 4, 0, 0, 4, 2, 4, 1, 1, 1, 0, 0, 4, 1, 0, 4, 3, 4, 3, 4, 1, 4, 1,
            1, 3, 0, 3, 4, 3, 2, 3])



### Building a Convolutional Neural Network

As we learned in the previous project, a network architecture suitable for image classification is the convolutional neural network (CNN). It primarily consists of a sequence of convolutional and max pooling layers. These layers are followed by some fully connected layers and an output layer.

Let's build a CNN!

Same as previously, we'll use the `nn.Sequential` class from PyTorch to define the architecture. We'll start with an empty model and append layers to it one by one.


```python
model = torch.nn.Sequential()
```

**Task 2.3.13:** Define the first convolutional layer of our network. Remember that we have three color channels, so set `in_channels=3`. Use $16$ kernels, each of size $3$ and set padding to $1$.


```python
# Convolutional layer 1 (sees 3x224x224 image tensor)
conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3 ,padding=1)
model.append(conv1)

print(model)
```

    Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )


After the first convolutional layer, we'll use the ReLU activation and follow that with a max pooling layer.


```python
max_pool1 = nn.MaxPool2d(2, 2)
model.append(torch.nn.ReLU())
model.append(max_pool1)
```




    Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )



**Task 2.3.14:** Define another convolutional layer of our network. This one should have $32$ kernels. Use kernels of size $3$ and padding of $1$.


```python
# Convolutional layer 2 (sees 16x112x112 tensor)
conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
max_pool2 = nn.MaxPool2d(2, 2)
model.append(conv2)
model.append(torch.nn.ReLU())
model.append(max_pool2)

print(model)
```

    Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )


**Task 2.3.15:** Define the last convolutional layer of our network. This one should have $64$ kernels. Again use kernels of size $3$ and padding of $1$.


```python
# Convolutional layer 3 (sees 32x56x56 tensor)
conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
max_pool3 = nn.MaxPool2d(2, 2)
model.append(conv3)
model.append(torch.nn.ReLU())
model.append(max_pool3)

print(model)
```

    Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU()
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )


Next we'll add the flattening layer and a `Dropout` layer.


```python
model.append(torch.nn.Flatten())
model.append(nn.Dropout(0.5))
```




    Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU()
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): Flatten(start_dim=1, end_dim=-1)
      (10): Dropout(p=0.5, inplace=False)
    )



While we could go straight to our output $5$ classes from here, we'll get better performance by adding another layer. We're getting 64 * 28 * 28 outputs from the flattening layer, let's feed that into a dense layer.

**Task 2.3.16:** Add a `Linear` layer to the model. You'll need to tell it the size of the input, and how many neurons we want in the layer (let's use $500$ neurons).


```python
# Linear layer (64 * 28 * 28 -> 500)
linear1 = torch.nn.Linear(64 * 28 * 28, 500)
model.append(linear1)
model.append(torch.nn.ReLU())
model.append(torch.nn.Dropout())

print(model)
```

    Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU()
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): Flatten(start_dim=1, end_dim=-1)
      (10): Dropout(p=0.5, inplace=False)
      (11): Linear(in_features=50176, out_features=500, bias=True)
      (12): ReLU()
      (13): Dropout(p=0.5, inplace=False)
    )


We are ready to add the final layer to the model. It should have as many neurons as we have classes.

**Task 2.3.17:** Add the output layer to the model. 


```python
# Linear layer (500 -> 5)
output_layer = nn.Linear(500, 5)
model.append(output_layer)

print(model)
```

    Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU()
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): Flatten(start_dim=1, end_dim=-1)
      (10): Dropout(p=0.5, inplace=False)
      (11): Linear(in_features=50176, out_features=500, bias=True)
      (12): ReLU()
      (13): Dropout(p=0.5, inplace=False)
      (14): Linear(in_features=500, out_features=5, bias=True)
    )


We have our model! Let's train it!

### Training the Model

Let's define the loss and what optimizer we'll use. We'll go with the standard loss function in classification problems, cross entropy. For the optimizer we'll chose Adam optimizer as we've done before.

**Task 2.3.18:** Define cross-entropy as the loss function and set Adam optimizer to be the optimizer. You can use the default learning rate `lr=0.001`.


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

print(loss_fn)
print("----------------------")
print(optimizer)
```

    CrossEntropyLoss()
    ----------------------
    Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        capturable: False
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 0.001
        maximize: False
        weight_decay: 0
    )


Let's use the GPU we have at our disposal and place our model on `device`. We'll also print the summary of the model.


```python
model.to(device)
```




    Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU()
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): Flatten(start_dim=1, end_dim=-1)
      (10): Dropout(p=0.5, inplace=False)
      (11): Linear(in_features=50176, out_features=500, bias=True)
      (12): ReLU()
      (13): Dropout(p=0.5, inplace=False)
      (14): Linear(in_features=500, out_features=5, bias=True)
    )




```python
height = 224
width = 224
summary(model, input_size=(batch_size, 3, height, width))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Sequential                               [32, 5]                   --
    ├─Conv2d: 1-1                            [32, 16, 224, 224]        448
    ├─ReLU: 1-2                              [32, 16, 224, 224]        --
    ├─MaxPool2d: 1-3                         [32, 16, 112, 112]        --
    ├─Conv2d: 1-4                            [32, 32, 112, 112]        4,640
    ├─ReLU: 1-5                              [32, 32, 112, 112]        --
    ├─MaxPool2d: 1-6                         [32, 32, 56, 56]          --
    ├─Conv2d: 1-7                            [32, 64, 56, 56]          18,496
    ├─ReLU: 1-8                              [32, 64, 56, 56]          --
    ├─MaxPool2d: 1-9                         [32, 64, 28, 28]          --
    ├─Flatten: 1-10                          [32, 50176]               --
    ├─Dropout: 1-11                          [32, 50176]               --
    ├─Linear: 1-12                           [32, 500]                 25,088,500
    ├─ReLU: 1-13                             [32, 500]                 --
    ├─Dropout: 1-14                          [32, 500]                 --
    ├─Linear: 1-15                           [32, 5]                   2,505
    ==========================================================================================
    Total params: 25,114,589
    Trainable params: 25,114,589
    Non-trainable params: 0
    Total mult-adds (Units.GIGABYTES): 5.24
    ==========================================================================================
    Input size (MB): 19.27
    Forward/backward pass size (MB): 359.79
    Params size (MB): 100.46
    Estimated Total Size (MB): 479.52
    ==========================================================================================



Notice the huge number of parameters that this model has! Are you starting to worry about overfitting? Although neural networks tend to overfit less than other models, we can still overfit with them. We can actually show that using this model. Let's train it for many epochs and demonstrate overfitting.

For training the model, we can import the functions we built in the last project. Same as before, we have a `training.py` file here, which contains the `train` function. Note that the `train` function has been slightly modified - it now returns the losses and accuracies. Otherwise it's the same as in Project 1. The cell below prints out the arguments it needs.


```python
from training import train
```


```python
train?
```


    [0;31mSignature:[0m
    [0mtrain[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mmodel[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0moptimizer[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mloss_fn[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mtrain_loader[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mval_loader[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mepochs[0m[0;34m=[0m[0;36m20[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdevice[0m[0;34m=[0m[0;34m'cpu'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0muse_train_accuracy[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m <no docstring>
    [0;31mFile:[0m      /app/training.py
    [0;31mType:[0m      function


**Task 2.3.19:** Use the `train` function to train the model for 15 epochs. Note that this may take a long time to run.

<div class="alert alert-info" role="alert"> <strong>Regarding Model Training Times</strong>

This task involves training a neural network for <b>15 epochs</b>. As highlighted in the accompanying video, the training process is computationally intensive and can be very time-consuming. On most systems, each epoch may take between 10 and 15 minutes, meaning the entire training process could last well over one hour. In an online lab, this could result in timeouts or interruptions.
<br>

To streamline your learning experience, where the video omits over an hour of training footage, we have provided a pre-trained model for your convenience. This model is an exact replica of the one you have been working on, trained for 15 epochs and carefully serialized using <code>torch.save()</code>.

Upon completing the video for this task you can proceed by loading the pre-trained model indicated in the cell after the empty <code>train(...)</code> one.

<b>We strongly recommend you to use the saved model instead of training your own for 15 epochs</b>
</div>


```python
train?
```


    [0;31mSignature:[0m
    [0mtrain[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mmodel[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0moptimizer[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mloss_fn[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mtrain_loader[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mval_loader[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mepochs[0m[0;34m=[0m[0;36m20[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdevice[0m[0;34m=[0m[0;34m'cpu'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0muse_train_accuracy[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m <no docstring>
    [0;31mFile:[0m      /app/training.py
    [0;31mType:[0m      function



```python
train_losses, valid_losses, train_accuracies, valid_accuracies = train(
    model, optimizer, loss_fn, train_loader, val_loader, epochs=15, device=device
)
```


    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 1
        Training loss: 1.49
        Training accuracy: 0.33
        Validation loss: 1.50
        Validation accuracy: 0.31



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 2
        Training loss: 1.35
        Training accuracy: 0.41
        Validation loss: 1.38
        Validation accuracy: 0.39



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 3
        Training loss: 1.29
        Training accuracy: 0.44
        Validation loss: 1.36
        Validation accuracy: 0.39



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 4
        Training loss: 1.22
        Training accuracy: 0.51
        Validation loss: 1.35
        Validation accuracy: 0.39



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 5
        Training loss: 1.13
        Training accuracy: 0.56
        Validation loss: 1.34
        Validation accuracy: 0.41



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 6
        Training loss: 0.94
        Training accuracy: 0.67
        Validation loss: 1.37
        Validation accuracy: 0.40



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 7
        Training loss: 0.78
        Training accuracy: 0.73
        Validation loss: 1.45
        Validation accuracy: 0.39



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 8
        Training loss: 0.58
        Training accuracy: 0.82
        Validation loss: 1.48
        Validation accuracy: 0.40



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 9
        Training loss: 0.35
        Training accuracy: 0.91
        Validation loss: 1.63
        Validation accuracy: 0.38



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 10
        Training loss: 0.20
        Training accuracy: 0.96
        Validation loss: 1.71
        Validation accuracy: 0.39



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 11
        Training loss: 0.14
        Training accuracy: 0.97
        Validation loss: 1.79
        Validation accuracy: 0.42



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 12
        Training loss: 0.08
        Training accuracy: 0.99
        Validation loss: 1.86
        Validation accuracy: 0.40



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 13
        Training loss: 0.05
        Training accuracy: 0.99
        Validation loss: 2.03
        Validation accuracy: 0.42



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 14
        Training loss: 0.04
        Training accuracy: 0.99
        Validation loss: 1.97
        Validation accuracy: 0.42



    Training:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/191 [00:00<?, ?it/s]



    Scoring:   0%|          | 0/48 [00:00<?, ?it/s]


    Epoch: 15
        Training loss: 0.03
        Training accuracy: 0.99
        Validation loss: 2.20
        Validation accuracy: 0.42


**[RECOMMENDED]** Load the pre-trained model:


```python
model = torch.load("model/model_trained.pth", weights_only=False)
model.to("cuda")
```




    Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU()
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): Flatten(start_dim=1, end_dim=-1)
      (10): Dropout(p=0.5, inplace=False)
      (11): Linear(in_features=50176, out_features=500, bias=True)
      (12): ReLU()
      (13): Dropout(p=0.5, inplace=False)
      (14): Linear(in_features=500, out_features=5, bias=True)
    )



Now let's plot the learning curve, so training and validation loss as a function of epoch number. And let's add a similar plot for training and validation accuracy.

<div class="alert alert-info" role="alert"> <strong>Loading Evaluation Metrics</strong>

The next cell plots the training and validation losses and accuracies resulting from training the model for 15 epochs. Since we're recommending you to train the model for fewer epochs and instead loading the pretrained model, you'll also need to load the metrics that we have saved along with the model.

To do so, execute the following cell that loads the CSV with the values of these metrics that we have saved after our training. Feel free to explore the dataframe freely.
</div>


```python
df = pd.read_csv('post_train_evaluation_metrics.csv')
```


```python
train_losses = df['Train Loss']
valid_losses = df['Validation Loss']

train_accuracies = df['Train Accuracy']
valid_accuracies = df['Validation Accuracy']
```


```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.title("Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(valid_accuracies, label="Validation Accuracy")
plt.title("Accuracy over epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
```


    
![png](output_92_0.png)
    


This training loop results show a common pattern observed in deep learning models: As training progresses, the training accuracy improves significantly, suggesting that the model's effectively learning from the training dataset. However, the validation accuracy does not improve at the same rate, and the validation loss starts to increase after a certain point. This is a classic sign of overfitting.

<div class="alert alert-info" role="alert">
<p>Oh no, we&rsquo;re overfitting! 🤯</p>

<p>Overfitting occurs when a model learns the training data too well, capturing noise and details to the extent that it negatively impacts the model&rsquo;s performance on new data. The symptoms of overfitting in our training results are:</p>

<ul>
<li>High training accuracy: The model performs exceptionally well on the training data.</li>
<li>Increasing validation loss: Despite improvements in training loss and accuracy, the validation loss starts to increase after reaching a certain point.</li>
<li>Stagnant or decreasing validation accuracy: The model&rsquo;s ability to generalize to unseen data does not improve or worsens as training progresses.</li>
</ul>


<p><strong>Addressing Overfitting</strong></p>

<p>To mitigate overfitting, consider the following strategies:</p>

<ul>
<li><p><strong>Data Augmentation</strong>: Augment the training data by applying random transformations (e.g., rotations, scaling, flips) to generate new training samples. This can help the model generalize better.</p></li>
<li><p><strong>Dropout</strong>: Introduce dropout layers into our network. Dropout randomly sets a fraction of input units to 0 during training, which helps prevent the model from becoming too reliant on any single feature.</p></li>
<li><p><strong>Regularization</strong>: Apply regularization techniques, such as L1 or L2 regularization, which add a penalty on the magnitude of network parameters. This can discourage complex models that overfit.</p></li>
<li><p><strong>Early Stopping</strong>: Monitor the model&rsquo;s performance on a validation set and stop training when the validation loss starts to increase, which is a sign that the model's beginning to overfit.</p></li>
<li><p><strong>Reduce Model Complexity</strong>: Simplify your model by reducing the number of layers or the number of units in the layers. A simpler model may generalize better.</p></li>
<li><p><strong>Use More Data</strong>: If possible, adding more data can help the model learn better and generalize well to new, unseen data.</p></li>
<li><p><strong>Batch Normalization</strong>: Although primarily used to help with training stability and convergence, batch normalization can sometimes also help with overfitting by regularizing the model somewhat.</p></li>
<li><p><strong>Cross-validation</strong>: Helps prevent overfitting by testing the model on different parts of the data, ensuring it performs well on new data.</p></li>
</ul>
</div>

We'll use some of these strategies in the following lessons.

Let's conclude this lesson by computing the confusion matrix for our model using the validation data. You may remember that in order to obtain the confusion matrix, we need the predictions of the model and the target values.

We'll obtain the probabilities that our model predicts by using the `predict` function from `training.py`. This function expects the model, the loader and the device as input arguments.

**Task 2.3.20:** Use the `predict` function from `training.py` to compute probabilities that our model predicts on the validation data. The rest of the code provided will take these probabilities and compute the predicted classes.

<div class="alert alert-info" role="alert"> <strong>Use the pre-trained model</strong>

For the following activity, we have assumed that you're using the pre-trained model from activity <code>2.3.19</code>. If you have trained the model by yourself, the activity <code>2.3.20</code> will not evaluate correctly.

Make sure you're loading the model from the pre-trained data as specified above.
</div>


```python
from training import predict

probabilities_val = predict(model, val_loader, device)
predictions_val = torch.argmax(probabilities_val, dim=1)

print(predictions_val)
```


    Predicting:   0%|          | 0/48 [00:00<?, ?it/s]


    tensor([0, 4, 4,  ..., 1, 0, 0], device='cuda:0')


All we need before we can compute the confusion matrix is the target values, so let's compute those.


```python
targets_val = torch.cat(
    [labels for _, labels in tqdm(val_loader, desc="Get Labels")]
).to(device)
```

    Get Labels: 100%|██████████| 48/48 [00:12<00:00,  3.91it/s]


Now we can get the confusion matrix.


```python
cm = confusion_matrix(targets_val.cpu(), predictions_val.cpu())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# Set figure size
plt.figure(figsize=(10, 8))

disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.show()
```


    <Figure size 1000x800 with 0 Axes>



    
![png](output_102_1.png)
    


The confusion matrix looks like quite a mess, which we expect, since the model's so overfit.

### Conclusion

In this lesson, we worked hard to create a CNN model, trained the model for a long time, and what we ended up with was a model that overfits! 😫

But along the way, we practiced and learned some important things:
- Once more we preprocessed our data to make it ready for deep learning. 
- We built a CNN with 3 convolutional and max pooling layers, followed by flattening and two dense layers.
- We used `nn.Sequential` to easily build our model's architecture by defining the order of the layers.
- Training the model for too many epochs produced an overfit model.
- We discussed techniques to combat overfitting, which we'll see in future lessons.


In the next lesson we'll learn that we don't have to build and train our neural network from scratch. That sounds promising!

---
&#169; 2024 by [WorldQuant University](https://www.wqu.edu/)


```python

```


```python

```


```python

```
