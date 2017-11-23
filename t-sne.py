import os, glob, re, json, cv2
import collections as cl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn import preprocessing

from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

INPUT_IMAGES_PATH = 'input_images'
OUTPUT_JSON_FILE_NAME = 'output/scatter.json'

def scatter_image(feature_x, feature_y, image_paths, title, output_file_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xlim = [min(feature_x)-5, max(feature_x)+5]
    ylim = [min(feature_y)-5, max(feature_y)+5]

    for (x, y, path) in zip(feature_x, feature_y, image_paths):
        img = plt.imread(path)
        img_height, img_width = img.shape[:2]

        bb = Bbox.from_bounds(x, y, img_height/200, img_width/200)
        bb2 = TransformedBbox(bb, ax.transData)
        bbox_image = BboxImage(bb2, norm=None, origin=None, clip_on=False)

        bbox_image.set_data(img)
        ax.add_artist(bbox_image)

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    plt.title(title)
    if output_file_name is not None:
        plt.savefig(output_file_name)
    plt.show()

def preprocess_image(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (size, size), cv2.INTER_LINEAR).astype('float')
    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
    timg = normalized.reshape(np.prod(normalized.shape))
    return timg/np.linalg.norm(timg)

paths = glob.glob(INPUT_IMAGES_PATH + '/*.jpg')
preprocess_images_as_vecs = [preprocess_image(path, 32) for path in paths]

tsne = TSNE(
    n_components=2,
    init='random',
    random_state=101,
    method='exact',
    n_iter=1000,
    verbose=2
).fit_transform(preprocess_images_as_vecs)

# NOTE: OrderedDict でハッシュを作らないのと順番が保持されない
param = cl.OrderedDict()
for (x, y, path) in zip(tsne[:,0], tsne[:,1], paths):
    asin = re.match('^(.*)_1.jpg', os.path.basename(path))[1]
    new_param = cl.OrderedDict()
    new_param['x'] = x
    new_param['y'] = y
    new_param['file_name'] = os.path.basename(path)
    param[asin] = new_param

f = open(OUTPUT_JSON_FILE_NAME, 'a')
json.dump(param, f)

# NOTE: 散布図の出力
# scatter_image(tsne[:,0], tsne[:,1], paths, 'handbag', output_file_name='output/scat.png')
