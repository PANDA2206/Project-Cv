import os
import shlex
import argparse
from tqdm import tqdm
import concurrent.futures


# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap

def parseArgs(parser):
    parser.add_argument('--labels_test', default='/Users/pankajrathi/Projcv/exercise3/icdar17_local_features/icdar17_labels_test.txt'
                        ,help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train',default ='/Users/pankajrathi/Projcv/exercise3/icdar17_local_features/icdar17_labels_train.txt',
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',default='/Users/pankajrathi/Projcv/exercise3/icdar17_local_features/test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',default= '/Users/pankajrathi/Projcv/exercise3/icdar17_local_features/train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float,
                        help='C parameter of the SVM')
    return parser
def getFiles(folder, pattern, labelfile):
    """
    returns files and associated labels by reading the labelfile
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()

    # get filenames from labelfile
    all_files = []
    labels = []
    # check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')

        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[indices]
        descriptors.append(desc)

    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

#
def dictionary(descriptors, n_clusters):
    """
    return cluster centers for the descriptors
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # kmeans = MiniBatchKMeans(n_clusters,batch_size= 10000,random_state = 100)
    # kmeans.fit(descriptors)
    # codebook = kmeans.cluster_centers_

    # TODO
    # Use MiniBatchKMeans to  create the codebook
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=100, batch_size=1000)

    # Fit the k-means model to the descriptors
    kmeans.fit(descriptors)

    # Get the cluster centers (codebook)
    codebook = kmeans.cluster_centers_

    return codebook


def assignments(descriptors, clusters):
    """
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """

    # Create a BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors using KNN
    matches = bf.knnMatch(descriptors, clusters, k=100)

    # Create an array to store the assignment matrix
    assignment = np.zeros((len(descriptors), clusters.shape[0]))

    # Iterate through matches and update the assignment matrix
    for i, match in enumerate(matches):
        for m in match:
            descriptor_index = m.queryIdx
            cluster_index = m.trainIdx
            assignment[descriptor_index, cluster_index] +=m.distance

    # Normalize assignment values to sum to 1 along the clusters axis
    assignment /= np.sum(assignment, axis=1, keepdims=True)

    return assignment


def vlad(files, mus, powernorm, gmp=True, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters:
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers 100* 64
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        a = assignments(desc, mus)


        T,D = desc.shape
        f_enc = np.zeros((D*K), dtype=np.float32)
        # Create a full association matrix
        full_association_matrix = np.zeros((T, K), dtype=np.float32)
        for k in range(mus.shape[0]):
            cluster_assigned_indices = a[:, k] > 0

            # Check if there are positive cases
            # if np.any(cluster_assigned_indices):
            cluster_descriptors = desc[cluster_assigned_indices, :]
            # # Compute residuals
            residuals = cluster_descriptors - mus[k, :]
            # # Flatten and aggregate residuals
            f_enc[k * D:(k + 1) * D] = residuals.flatten().sum(axis=0)

        # c) power normalization
        if powernorm:
            f_enc = np.sign(f_enc) * np.abs(f_enc) ** 0.5

        # l2 normalization
        f_enc = normalize(f_enc.reshape(1, -1), norm='l2').flatten()

        encodings.append(f_enc)

    return np.vstack(encodings)

def loop(i, encs_test, encs_train, C):
    # Extract the i-th test encoding
    x_test = encs_test[i].reshape(1, -1)

    # Combine test and train for SVM training
    X_combined = np.concatenate([encs_train, x_test], axis=0)

    # Set up labels for training
    y_train = np.zeros(len(X_combined))
    y_train[:len(encs_train)] = -1

    # Train LinearSVC
    clf = LinearSVC(C=C, class_weight='balanced', dual=True,max_iter=1000)
    clf.fit(X_combined, y_train)

    # Feature transformation
    x_transformed = clf.coef_.reshape(1, -1)

    return x_transformed

def esvm(encs_test, encs_train, C=10000):
    """
    compute a new encoding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives
    parameters:
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """
    # Use ThreadPoolExecutor for better compatibility with some environments
    with concurrent.futures.ThreadPoolExecutor() as executor:
        new_encs = list(tqdm(executor.map(lambda i: loop(i, encs_test, encs_train, C), range(len(encs_test))), total=len(encs_test)))

    # Concatenate the results
    new_encs = np.concatenate(new_encs, axis=0)
    return new_encs


def distances(encs):
    """
    compute pairwise distances

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    norm_encs = normalize(encs, norm='l2', axis=1)  # L2-normalize the encodings
    dists = 1 - np.dot(norm_encs, norm_encs.T)

    # Mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)

    return dists
#%%
def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k]] == labels[r]:
                rel += 1
                precisions.append(rel / float(k+1))
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))



#%%
if __name__ == '__main__':
    args = argparse.Namespace()


    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(100) # fix random seed

    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix, args.labels_train)
    print('#train: {}'.format(len(files_train)))
    if not os.path.exists('mus.pkl.gz'):
        descriptors = []
        for f in tqdm(files_train):
            with gzip.open(f, 'rb') as ff:
                desc = cPickle.load(ff, encoding='latin1')
                descriptors.extend(desc)

        # Convert descriptors to a numpy array
        descriptors = np.vstack(descriptors)
        mus = dictionary(descriptors[np.random.choice(descriptors.shape[0], 500000, replace=False)], args.n_clusters)

        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)
    # print('mus shape:', mus.shape if descriptors is not None else None)
    # print('descriptors shape:', descriptors.shape if descriptors is not None else None)

    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix, args.labels_test)
    print('#test: {}'.format(len(files_test)))
    fname = 'enc_test_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        enc_test = vlad(files_test, mus, powernorm=args.powernorm, gmp=args.gmp, gamma=args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)

    # Cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) Compute exemplar SVMs
    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        enc_train = vlad(files_train, mus, powernorm=args.powernorm, gmp=args.gmp, gamma=args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    # Save intermediate results for debugging
    with gzip.open('enc_train_intermediate.pkl.gz', 'wb') as fOut:
        cPickle.dump(enc_train, fOut, -1)

    print('> E-SVM computation')
    new_enc_test = esvm(enc_test, enc_train, args.C)

    # Save intermediate results for debugging
    with gzip.open('new_enc_test_intermediate.pkl.gz', 'wb') as fOut:
        cPickle.dump(new_enc_test, fOut, -1)

    # Eval

    evaluate(new_enc_test, labels_test)
    print('> evaluate after E-SVM')