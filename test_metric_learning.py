from scripts.python.data_loader import CORE50

import torch
from tqdm import tqdm
from faiss_knn import knn
import cv2 as cv
import numpy as np
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import os
import pathlib
import wandb
from umap import UMAP

import matplotlib.pyplot as plt
import seaborn as sns
import time


transforms = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

torch.set_grad_enabled(False)


def load_features_for_testing(fe, test_x, features_size, batch_size=32):

    if pathlib.Path('test_features.pth').exists():
        print('Found saved features')
        saved_feats = torch.load('test_features.pth')
        if saved_feats.shape[1] == features_size:
            return saved_feats

    features = torch.empty((0, features_size), dtype=torch.float32)
    for i in tqdm(range(test_x.shape[0] // batch_size + 1)):
        x_minibatch = test_x[i*batch_size: (i+1)*batch_size]

        x = [transforms(el) for el in x_minibatch.astype(np.uint8)]
        x = torch.stack(x)

        feats = fe(x.cuda()).cpu()

        features = torch.cat((features, feats))
    torch.save(features, 'test_features.pth')
    return features


def visualize_features(x_data, y_data, folder='./visualizations', return_array=False, iter=0):
    # time_start = time.time()
    umap = UMAP()

    results = umap.fit_transform(x_data.squeeze())

    # print('visualization done! Time elapsed: {} seconds'.format(
    #     time.time()-time_start))

    fig = plt.figure(figsize=(16, 10))

    sns.scatterplot(
        x=results[:, 0], y=results[:, 1],
        hue=y_data,
        palette=sns.color_palette("hls", len(set(y_data))),
        legend=None,
    )

    path = pathlib.Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{folder}/{iter}_iteration.png')

    if return_array:
        im = cv.imread(f'{folder}/{iter}_iteration.png')
        im = cv.resize(im, (640, 480))
        return im


def run_experiment(run, cfg=None):
    dataset = CORE50(root='core50/core50_128x128',
                     scenario="nicv2_391", preload=False, run=run)

    name = '_'.join(list(map(str, cfg.values())))

    wandb.init(project='core50_DINO_knn', reinit=True,
               name=name + '_' + str(run), config=cfg)

    # Get the fixed test set
    test_x, test_y = dataset.get_test_set()

    fe = torch.hub.load('facebookresearch/dino:main',
                        cfg['feature_extractor_model'])
    fe.cuda()
    fe.eval()
    classifier = knn('core50.pth', resume=False, knn_size=cfg['N_neighbours'])

    batch_size = 32

    test_x = load_features_for_testing(fe, test_x)

    # loop over the training incremental batches
    for iteration_step, train_batch in tqdm(enumerate(dataset), total=dataset.nbatch[dataset.scenario]):
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y = train_batch

        # train stage
        for i in range(train_x.shape[0] // batch_size + 1):

            x_minibatch = train_x[i*batch_size: (i+1)*batch_size]
            y_minibatch = train_y[i*batch_size: (i+1)*batch_size]

            x = [transforms(el) for el in x_minibatch.astype(np.uint8)]
            x = torch.stack(x)

            feats = fe(x.cuda())
            classifier.add_points(feats.cpu(), y_minibatch)

        # test stage
        preds = np.empty((0))

        start_time = time.time()
        for i in range(test_x.shape[0] // batch_size + 1):
            x_minibatch = test_x[i*batch_size: (i+1)*batch_size]
            y_minibatch = test_x[i*batch_size: (i+1)*batch_size]

            clss, confs, dists = classifier.classify(x_minibatch)
            preds = np.concatenate((preds, clss))

        duration = time.time() - start_time

        M = confusion_matrix(test_y, preds)
        accs = M.diagonal()/M.sum(axis=1)
        print(
            f'{iteration_step}, mean accuracy: {accs.mean():.3f}')

        logs_keys = ['accs/mean', 'accs/std', 'time to test kNN', 'data size']
        logs_vals = [accs.mean(), accs.std(), duration, len(classifier.x_data)]
        logs_dict = dict(zip(logs_keys, logs_vals))

        wandb.log(logs_dict, step=iteration_step)

        # save features visualization to WanDB
        plot = visualize_features(
            classifier.x_data, classifier.y_data, return_array=True, iter=iteration_step)
        wandb.log({"2D visualization": wandb.Image(plot)},
                  step=iteration_step)


if __name__ == "__main__":

    cfg = {
        'feature_extractor_model': 'dino_vitb16',
        'embedding_size': 768,
        'N_neighbours': 10,
        'runs': 1
    }

    for run in range(cfg['runs']):

        run_experiment(run, cfg)
