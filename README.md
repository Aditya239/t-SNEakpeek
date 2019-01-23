# Visualizing a couple of popular multidimensional datasets using t-SNE

# Overview

The [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.html) maintains hundreds of data sets as a great service to the machine learning community. The datasets have been widely studied, diagnosed and used for benchmarking several ML algorithms. Datasets such as the MNIST handwritten dataset and the age old Iris dataset are still classics in the classification domain of ML. This code obtains the dataset from the scikit library, cleans it appropriately, creates suitably adjusted feature vectors, then uses [T-SNE](https://lvdmaaten.github.io/tsne/) to reduce the dimensionality of the feature vectors to just 2. Then, I use matplotlib to visualize the data. Further along, I explode a subdirectory with snapshots of the sequential visualization and cluster segregation that the T-SNE does in the course of its gradient descent. These images may be easily seamed into a gif or video. I leave it as the whole bunch in the folder. It is illuminating to see the scope and rate of unsupervised clustering.

## Dependencies

* pandas(http://pandas.pydata.org/) 
* numpy (http://www.numpy.org/) 
* scikit-learn (http://scikit-learn.org/) 
* matplotlib (http://matplotlib.org/) 
* scipy (https://www.scipy.org/)
* moviepy (https://pypi.python.org/pypi/moviepy)
* imageio (https://pypi.python.org/pypi/imageio)

Install dependencies via '[pip](https://pypi.python.org/pypi/pip) install'. (i.e pip install pandas). 

## Usage

To run this code, just run the following in terminal: 

`python tsneakpeek_digits.py` for the MNIST handwritten digits set
`python tsneakpeek_iris.py` for the Iris set
