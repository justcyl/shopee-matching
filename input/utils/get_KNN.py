import gc

import numpy as np
import pandas as pd

# from sklearn.neighbors import NearestNeighbors
import cuml, cupy

import  eval_preds

def get_valid_neighbors(df, embeddings, destination, threshold, KNN = 55):
    print(f"Finding similar by cosine KNN..., len of train: {len(df)}, KNN={KNN}")
    model = cuml.NearestNeighbors(n_neighbors = KNN, metric = 'cosine')
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    predictions = []
    for k in range(embeddings.shape[0]):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k,idx]
        posting_ids = np.array(df['posting_id'].iloc[ids].values)
        # predictions.append(posting_ids)
        predictions.append(' '.join(posting_ids))
    
    df[destination] = predictions

    df['precision'],df['recall'],df['f1'] =  eval_preds.get_score(df['target'], df[destination])
    # if COMPUTE_CV:
        # df['precision'],df['recall'],df['f1'] = get_score(df['target'], df[destination])

    del model, distances, indices
    gc.collect()
    return df