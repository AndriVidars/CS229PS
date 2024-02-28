import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def init_centroids(num_clusters, image):
    H, W, _ = image.shape
    r_idx = np.random.randint(0, H, size=num_clusters)
    c_idx = np.random.randint(0, W, size=num_clusters)
    
    return image[r_idx, c_idx, :]

def closest_centroid(x, centroids):
    dists = np.sum((centroids - x)**2, axis=1)
    return np.argmin(dists)

def update_centroids_iter(cluster_map):
    new_centroids = []
    for _, pixel_vals in cluster_map.items():
        c_vals = np.array(pixel_vals)
        center = np.average(c_vals, axis=0)
        new_centroids.append(center)
    
    return np.array(new_centroids)


def update_centroids(centroids, image, max_iter=30, print_every=10):
    H, W, _ = image.shape
    
    for i in range(max_iter):
        cluster_map = defaultdict(list)
        for x in range(H):
            for y in range(W):
                pixel_val = image[x, y, :]
                c = closest_centroid(pixel_val, centroids)
                cluster_map[c].append(pixel_val)
        
        old_centroids = centroids.copy()
        centroids = update_centroids_iter(cluster_map)
        if i % print_every == 0:
            print(f'Iter {i}: \n{centroids}\n')
        
        if np.linalg.norm(centroids-old_centroids) < 1e-5:
            print(f'Reached convergence threshold at iter {i}')
            break

    return centroids


def update_image(image, centroids):
    H, W, _ = image.shape
    centroids = np.round(centroids).astype(int)
    init_img = image.copy()
    for x in range(H):
        for y in range(W):
            pixel_val = image[x, y, :]
            c = closest_centroid(pixel_val, centroids)
            image[x, y, :] = centroids[c, :]

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
