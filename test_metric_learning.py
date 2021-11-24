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

aug_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAdjustSharpness(
        sharpness_factor=2),
    transforms.RandomAutocontrast(),
    transforms.RandomResizedCrop(
        scale=(0.16, 1), ratio=(0.75, 1.33), size=224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

std_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


torch.set_grad_enabled(False)


def load_features_for_testing(fe, test_x, features_size, batch_size=32, postfix=None):

    if pathlib.Path(f'test_features{postfix}.pth').exists():
        print('Found saved features')
        saved_feats = torch.load(f'test_features{postfix}.pth')
        if saved_feats.shape[1] == features_size:
            return saved_feats

    features = torch.empty((0, features_size), dtype=torch.float32)
    for i in tqdm(range(test_x.shape[0] // batch_size + 1)):
        x_minibatch = test_x[i*batch_size: (i+1)*batch_size]

        x = [std_transforms(el) for el in x_minibatch.astype(np.uint8)]
        x = torch.stack(x)

        feats = fe(x.cuda()).cpu()

        features = torch.cat((features, feats))
    torch.save(features, f'test_features{postfix}.pth')
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
    dataset = CORE50(root='core50_dataset/core50_128x128',
                     scenario="nicv2_391", preload=False, run=run)

    # set name for experiment
    name = f"{cfg['feature_extractor_model']}_{cfg['embedding_size']}_{cfg['N_neighbours']}n_{cfg['runs']}r"
    if cfg['augmentation']:
        name += '_aug'
    if cfg['pca']:
        name += f'_{cfg["pca"].split("_")[2]}pca'
    # name = '_'.join(list(map(str, cfg.values())))

    print(name)

    wandb.init(project='core50_DINO_knn', reinit=True,
               name=name + '_' + str(run), config=cfg)

    transforms = aug_transforms if cfg['augmentation'] else std_transforms

    # Get the fixed test set
    test_x, test_y = dataset.get_test_set()
    # print(test_x.shape)

    fe = torch.hub.load('facebookresearch/dino:main',
                        cfg['feature_extractor_model'])
    fe.cuda()
    fe.eval()
    classifier = knn('core50.pth', resume=False, knn_size=cfg['N_neighbours'])

    batch_size = 512

    test_x = load_features_for_testing(
        fe, test_x, cfg['embedding_size'], batch_size=batch_size, postfix=f'_{cfg["feature_extractor_model"]}')

    # prepare pca 
    if cfg['pca']:
        pca_size = cfg['pca'].split('_')[2]
        pca_trained_arch = cfg['pca'].split('_')[-1][:-4]
        print(pca_trained_arch)
        pca=  torch.load(cfg['pca'])

        assert cfg['feature_extractor_model'].split('_')[-1] == pca_trained_arch
        test_x = pca.transform(test_x.numpy().astype(np.float32))

    # loop over the training incremental batches
    total_pbar = tqdm(enumerate(dataset), total=dataset.nbatch[dataset.scenario])
    for iteration_step, train_batch in total_pbar:
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y = train_batch


        # train stage
        
        # for _ in range(4):
        for i in range(train_x.shape[0] // batch_size + 1):

            x_minibatch = train_x[i*batch_size: (i+1)*batch_size]
            y_minibatch = train_y[i*batch_size: (i+1)*batch_size]

            x = [transforms(el)
                    for el in x_minibatch.astype(np.uint8)]
            x = torch.stack(x)

            feats = fe(x.cuda()).cpu()
            if cfg['pca']:
                feats = pca.transform(feats.numpy()).astype(np.float32)
            classifier.add_points(feats, y_minibatch)

        # test stage
        preds = np.empty((0))

        test_batch_size = 4096 * 8
        start_time = time.time()
        for i in tqdm(range(test_x.shape[0] // test_batch_size + 1), desc='test'):
            x_minibatch = test_x[i*test_batch_size: (i+1)*test_batch_size]
            y_minibatch = test_y[i*test_batch_size: (i+1)*test_batch_size]

            clss, confs, dists = classifier.classify(x_minibatch)
            preds = np.concatenate((preds, clss))

        duration = time.time() - start_time

        M = confusion_matrix(test_y, preds)
        accs = M.diagonal()/M.sum(axis=1)
        total_pbar.set_description(f'{iteration_step}, acc: {accs.mean():.3f}')


        logs_keys = ['accs/mean', 'accs/std', 'time to test kNN', 'data size']
        logs_vals = [accs.mean(), accs.std(), duration, len(classifier.x_data)]
        logs_dict = dict(zip(logs_keys, logs_vals))

        wandb.log(logs_dict, step=iteration_step)

        # # save features visualization to WanDB
        # plot = visualize_features(
        #     classifier.x_data, classifier.y_data, return_array=True, iter=iteration_step)
        # wandb.log({"2D visualization": wandb.Image(plot)},
        #           step=iteration_step)


if __name__ == "__main__":
    # archs = ['dino_vitb8', #'dino_xcit_small_12_p8'
    # ]

    fs = os.listdir('pca')
    fs.sort(key=lambda x: int(x.split('_')[2]))
    for pca in fs[-4:]:

        cfg = {
            'feature_extractor_model': 'dino_vits16',
            'embedding_size': 384,
            'N_neighbours': 10,
            'runs': 1,
            'augmentation': False,
            'pca': 'pca/'+ pca
        }

        for run in range(cfg['runs']):

            run_experiment(run, cfg)
