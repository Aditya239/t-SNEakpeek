# Import List
import os
import numpy as np
from images2gif import writeGif
from numpy import linalg
from PIL import Image
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import imageio
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
#########################################################
directory = "Images"
if not os.path.exists(directory):
    os.makedirs(directory)


irises = load_iris()
irises.data.shape
print(irises['DESCR'])
options = ['Setosa', 'Versicolour', 'Virginica']
#print irises

#We order the images according to their targets

X = np.vstack([irises.data[irises.target==i]
               for i in range(3)])
y = np.hstack([irises.target[irises.target==i]
               for i in range(3)])

irises_proj = TSNE(random_state=RS).fit_transform(X)


#A function to create the final t-SNE scatterplot

def scatter(x, colors):
    global options
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 3))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(3):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, options[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
scatter(irises_proj, y)
plt.savefig('irises_tsne-generated.jpg', dpi=120)
# This list will contain the positions of the map points at every iteration

positions = []
total_count = 0
def _gradient_descent(objective, p0, it, n_iter, objective_error=None, n_iter_check=0, 
						n_iter_without_progress=30,
                      momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                      min_grad_norm=0.0000001, min_error_diff=0.0000001, verbose=0,
                      args=[], kwargs=None):
	# Some weird behaviour here. Patched dirtily.
    if min_grad_norm == 0.001:
    	min_grad_norm = 0.0000001
    global total_count
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = 0
    for i in range(it, n_iter):
        # We save the current position.
        positions.append(p.copy())
        total_count = total_count + 1
        new_error, grad = objective(p, *args)
        error_diff = np.abs(new_error - error)
        error = new_error
        grad_norm = linalg.norm(grad)
        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break
        if min_grad_norm >= grad_norm:
            break
        if min_error_diff >= error_diff:
            break

        inc = update * grad >= 0.0
        dec = np.invert(inc)
        gains[inc] += 0.05
        gains[dec] *= 0.95
        np.clip(gains, min_gain, np.inf)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

    return p, error, i
sklearn.manifold.t_sne._gradient_descent = _gradient_descent
X_proj = TSNE(random_state=RS).fit_transform(X)
X_iter = np.dstack(position.reshape(-1, 2) for position in positions)
f, ax, sc, txts = scatter(X_iter[..., -1], y)

def make_frame_mpl(t):
    i = int(t)
    x = X_iter[..., i]
    sc.set_offsets(x)
    for j, txt in zip(range(3), txts):
        xtext, ytext = np.median(x[y == j, :], axis=0)
        txt.set_x(xtext)
        txt.set_y(ytext)
    return mplfig_to_npimage(f)
filenames = []
for x in xrange(total_count):
	img = Image.fromarray(make_frame_mpl(x), 'RGB')
	filename = 'Images/img'+str(x)+'.png'
	img.save(filename)
	img.show()
	filenames.append(filename);