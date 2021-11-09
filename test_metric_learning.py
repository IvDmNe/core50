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


transforms = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

torch.set_grad_enabled(False)


def load_features_for_testing(fe, test_x, batch_size=32):
    features = torch.empty((0, 384), dtype=torch.float32)

    if pathlib.Path('test_features.pth').exists():
        print('Found saved features')
        return torch.load('test_features.pth')

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


if __name__ == "__main__":
    # Create the dataset object for example with the "NIC_v2 - 79 benchmark"
    # and assuming the core50 location in ~/core50/128x128/
    dataset = CORE50(root='core50/core50_128x128',
                     scenario="nicv2_391", preload=False)

    wandb.init(project='core50_DINO_knn')

    # Get the fixed test set
    test_x, test_y = dataset.get_test_set()

    fe = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    fe.cuda()
    fe.eval()
    classifier = knn('core50.pth', save_to_file=False)

    batch_size = 32

    test_x = load_features_for_testing(fe, test_x)

    total_trained_classes = set()

    # loop over the training incremental batches
    for iteration_step, train_batch in tqdm(enumerate(dataset), total=dataset.nbatch[dataset.scenario]):
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y = train_batch

        print('current classes: ', np.unique(train_y).astype(int))

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
        for i in range(test_x.shape[0] // batch_size + 1):
            x_minibatch = test_x[i*batch_size: (i+1)*batch_size]
            y_minibatch = test_x[i*batch_size: (i+1)*batch_size]

            clss, confs, dists = classifier.classify(x_minibatch)
            preds = np.concatenate((preds, clss))

        M = confusion_matrix(test_y, preds)
        accs = M.diagonal()/M.sum(axis=1)
        # total_trained_classes += set(train_y)
        print(
            f'{iteration_step}, mean accuracy: {accs.mean():.3f}')

        logs_keys = ['accs/mean', 'accs/std']
        logs_vals = [accs.mean(), accs.std()]
        logs_dict = dict(zip(logs_keys, logs_vals))

        wandb.log(logs_dict, step=iteration_step)

        # save features visualization to WanDB
        plot = visualize_features(
            classifier.x_data, classifier.y_data, return_array=True, iter=iteration_step)
        wandb.log({"2D visualization": wandb.Image(plot)}, step=iteration_step)
