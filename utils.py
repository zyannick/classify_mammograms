import cv2
from matplotlib import pyplot as plt
import pandas as pd
import os
import pickle as pkl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler

import glob


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


label_to_id_dict = {v: i for i, v in enumerate(['MLO', 'CC'])}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}


def visualize_scatter_with_images(X_2d_data, images,  image_zoom=1.0,  data_dir=''):
    fig, ax = plt.subplots(figsize=(45,45))
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.savefig(os.path.join(data_dir, 'tsne_features_images.png'), dpi=600)


def tsne_scattering(tsne_results, df_subset, data_dir):
    print(tsne_results.shape)
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("tab10", 2),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.savefig(os.path.join(data_dir, 'tsne_features.png'), dpi=600)


def plot_tsne_features( modele_dir , data_dir, image_paths, test_preds, test_labels):

    test_preds = np.squeeze(test_preds)
    test_labels = np.squeeze(test_labels)

    list_data = sorted(glob.glob(os.path.join(data_dir, 'data_*.pkl')))


    all_data = None
    all_target = None
    all_prediction = None

    rescaled_images = []

    for cp in range(len(list_data) - 1):

        with open(list_data[cp], "rb") as fout:
            data = pkl.load(fout)
            data = data.flatten()
            data = np.expand_dims(data, axis=0)

        image_path = image_paths[cp]

        target = test_labels[cp]
        prediction = test_preds[cp]

        phase = (target == prediction)

        img = cv2.imread(image_path, 0)
        img = cv2.resize(img, dsize=(45, 45))
        rescaled_images.append(img)

        target = np.asarray([target])
        prediction = np.asarray([prediction])

        if all_data is None:
            all_data = data
            all_prediction = prediction
            all_target = target
        else:
            all_data = np.append(all_data, data, axis=0)
            all_prediction = np.append(all_prediction, prediction)
            all_target = np.append(all_target, target)


    #for phase in [True, False]:
    phase_features = all_data
    phase_target = all_target

    print('all features')
    print(phase_features.shape)
    print('all target')
    print(phase_target.shape)
    print(np.count_nonzero(all_target))


    nb_features = phase_features.shape[1]
    feat_cols = ['feature_' + str(i) for i in range(nb_features)]

    df = pd.DataFrame(phase_features, columns=feat_cols)



    df['y'] = phase_target

    df['label'] = df['y'].apply(lambda i: str(i))

    print(sorted(sklearn.neighbors.VALID_METRICS['brute']))

    distance_metric = 'l2'

    nb_videos = phase_features.shape[0]

    # For reproducability of the results
    np.random.seed(42)
    rndperm = np.random.permutation(nb_videos)

    df_subset = df.loc[rndperm[:nb_videos], :].copy()
    data_subset = df_subset[feat_cols].values
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_subset)

    df_subset['pca-one'] = pca_result[:, 0]
    df_subset['pca-two'] = pca_result[:, 1]
    df_subset['pca-three'] = pca_result[:, 2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40,
                n_iter=400, n_jobs=8, metric=distance_metric)

    tsne_results = tsne.fit_transform(data_subset)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_results)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    tsne_scattering(tsne_results, df_subset, modele_dir)
    visualize_scatter_with_images(tsne_result_scaled, rescaled_images, image_zoom=0.7, data_dir=modele_dir)
