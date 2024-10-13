# this code does the following:
# 1. Loads the LLM generated data, and break reasoning paragraphs into sentences
# 2. Embed each sentence with s-BERT
# 3. For each proposition:
#     a. Split sentences into sentences that agree and disagree with the proposition
#     b. Cluster the embeddings of each sentence set using DBSCAN
#     c. Remove small clusters and outliers
#     d. Find centroid sentences for each cluster
#     e. Use LLM to filter out centroid sentences that are not argumentative, and therefore not tropes
# 4. Generate a new DataFrame with the centroids for further analysis

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from typing import List, Dict
from glob import glob
import torch
import logging
from trope_filter import TropeFilter
import argparse
import spacy
from tqdm import tqdm
import os

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000
tqdm.pandas()

# set logger with nice formatting
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


MIN_CLUSTER_SIZE = 10
TEXT_COLUMN = "response"
DEBUG_PROPOSITION = "Abortion, when the woman’s life is not threatened, should always be illegal."


def parse_args():
    parser = argparse.ArgumentParser(description="Extract tropes from LLM generated data")
    parser.add_argument("data_path", type=str, default="compass", help="the data")
    parser.add_argument("save_path", type=str, default="data_with_tropes.csv")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model to use")
    parser.add_argument("--eps", type=float, default=0.15, help="DBSCAN epsilon parameter")
    parser.add_argument("--min_samples", type=int, default=10, help="DBSCAN min_samples parameter")
    parser.add_argument("--no_filter", action="store_true", help="Do not filter out non-tropes")
    parser.add_argument("--local_filter", action="store_true", help="Use local filtering model instead of OpenAI API")
    parser.add_argument("--filtration_model_id", type=str, default="gpt-4o-mini", help="Model to use for trope filtration")
    parser.add_argument("--open_ai_credentials_path", type=str, default="tropes/secrets.json", help="Path to OpenAI credentials, a json file")
    parser.add_argument("--verbose", action="store_true", help="Print more information")
    parser.add_argument("--debug", action="store_true", help="debug mode with smaller data")
    parser.add_argument("--centroids_method", type=str, default="euclidian", help="Method to calculate the centroid of the cluster")
    return parser.parse_args()


def sentence_tokeniser(paragraph: str) -> List[str]:
    # use spacy sentence tokeniser to break a paragraph into sentences
    return [s.text for s in nlp(paragraph).sents]



def read_data(args) -> List[str]:
    """
    Read the data from the given path
    """
    all_files = glob(f"{args.data_path}/*.csv")
    data = pd.concat((pd.read_csv(f) for f in all_files))
    data = data.dropna(subset=[TEXT_COLUMN, "selection"])
    data = data[data[TEXT_COLUMN].apply(lambda x: isinstance(x, str))]
    data = data[data["selection"].apply(lambda x: isinstance(x, str))]
    data["agreement"] = data["selection"].apply(
        lambda x: "disagree" if "disagree" in x.lower() else "agree"
    )
    if args.debug:
        data = data[data["proposition"] == DEBUG_PROPOSITION]
    return data


def embed_sentences(
    sentences: List[str], model: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Function to embed the sentences using the given model
    """
    embedder = SentenceTransformer(
        model, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    embeddings = embedder.encode(sentences, show_progress_bar=True)
    return embeddings


def cluster_sentences(
    embeddings: np.ndarray, eps: float = 0.05, min_samples: int = 10
) -> np.ndarray:
    """
    Function to cluster the embeddings using DBSCAN
    """
    db = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(embeddings)
    print(f"Number of clusters: {len(set(clusters))}")
    return clusters


def find_centroids(data: pd.DataFrame, method: str ="euclidian") -> List[str]:
    # find the centroid sentence of each cluster
    centroids = dict()
    data["trope"] = [None] * data.shape[0]
    centroid_idxs = []
    for i in set(data["cluster"]):
        cluster = data[data["cluster"] == i]
        cluster_embeddings = cluster["embeddings"].tolist()
        # find the closest sentence to the centroid
        if method == "euclidian":
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        elif method == "cosine":
            # calc the centroid of the cluster based on the cosine angle
            # normalize the embeddings by their length
            cluster_embeddings = cluster_embeddings / np.linalg.norm(cluster_embeddings, axis=1)[:, None]
            centroid = np.sum(cluster_embeddings, axis=0)
            # Normalize the centroid by length
            centroid = centroid / np.linalg.norm(centroid)
            # calc the cosine distance between the centroid and the embeddings
            distances = np.dot(cluster_embeddings, centroid)
        else:
            raise ValueError("method should be either 'euclidian' or 'cosine'")
        centroid_idx = np.argmin(distances)
        centroid_idxs.append(centroid_idx)
        centroids[i] = cluster.iloc[centroid_idx]["sentences"]
        data.loc[data["cluster"] == i, "trope"] = centroids[i]
    return data, centroid_idxs


def filter_centroids(data: pd.DataFrame, trope_filter: TropeFilter, args) -> List[int]:
    proposition = data["proposition"].iloc[0]
    centroids_per_proposition = [
        {"proposition": proposition, "id": i, "sentence": trope}
        for i, trope in enumerate(set(data["trope"]))
    ]

    filtered_centroids: List[Dict] = trope_filter.filter_tropes(centroids_per_proposition, verbose=args.verbose)
    filtered_centroids = [centroid["sentence"] for centroid in filtered_centroids]
    data = data[data["trope"].isin(filtered_centroids)]
    return data


def main(args):
    
    os.makedirs(args.save_path, exist_ok=True)
    save_path = f"{args.save_path}/data_with_tropes_{args.eps}_{args.min_samples}"

    data = read_data(args)
    logger.info(f"Number of responses: {len(data)}")

    # break each "response" into sentences and duolicate the rest of the data in the dataframe
    logger.info("Spliting to sentences ...")
    data["sentences"] = data[TEXT_COLUMN].progress_apply(sentence_tokeniser)
    data = data.explode("sentences")
    
    # remove sentences that are too short
    data = data[data["sentences"].apply(lambda x: len(x) > 20)]
    data = data.reset_index(drop=True)
    logger.info(f"Number of sentences: {len(data)}")

    # embed the sentences
    logger.info("Embedding sentences...")
    embeddings = embed_sentences(data["sentences"], model=args.model)
    data["embeddings"] = list(embeddings)

    all_data = []
    metrics = []
    trope_filter = TropeFilter(args)
    groups = data.groupby(["proposition", "agreement"])
    for name, group in groups:
        group_metrics = {}
        try:
            group_metrics["initial_number_of_sentences"] = group.shape[0]
            group = group.copy()
            logger.info(f"processing proposition: {name[0]}, agreement: {name[1]}")
            # cluster the sentences of the group
            logger.info("Clustering sentences...")
            clusters = cluster_sentences(group["embeddings"].tolist(), eps=args.eps, min_samples=args.min_samples)
            group["cluster"] = clusters
            group_metrics["n_clusters_pre_filter"] = len(set(clusters))
            group_metrics["n_outliers_pre_filter"] = len(group[group["cluster"] == -1])
            logger.info(f"Number of clusters: {group_metrics['n_clusters_pre_filter']}")
            logger.info(f"Number of outliers: {group_metrics['n_outliers_pre_filter']}")
            

            # remove outliers and small and huge clusters
            logger.info("Filtering clusters...")
            group = group[group["cluster"] != -1]
            group = group.groupby("cluster").filter(lambda x: len(x) > MIN_CLUSTER_SIZE)
            group = group.groupby("cluster").filter(lambda x: len(x) < group.shape[0] / 2)
            group_metrics["n_sentences_after_filter"] = group.shape[0]
            group_metrics["n_clusters_post_filter"] = len(set(group["cluster"]))
            logger.info(f'Number of sentences after filtering: {group_metrics["n_sentences_after_filter"]}')
            logger.info(f'Number of clusters after filtering: {group_metrics["n_clusters_post_filter"]}')

            # find the centroid sentence of each cluster
            _, centroid_idxs = find_centroids(group.copy(), "cosine")
            group_metrics["centroid_id_cosine"] = centroid_idxs
            group, centroid_idxs = find_centroids(group, method=args.centroids_method)
            group_metrics["centroid_id"] = centroid_idxs
            # filter out non-tropes using LLM
            if not args.no_filter:
                group = filter_centroids(group, trope_filter, args)
            group_metrics["n_tropes"] = len(set(group["trope"]))
            logger.info(f"Number of tropes: {group_metrics['n_tropes']}")
            
            all_data.append(group)
            metrics.append(group_metrics)
        except IndexError:
            logger.error(f"Error with proposition: {name[0]}, agreement: {name[1]}")
            continue
    all_data_df = pd.concat(all_data)
    all_data_df = all_data_df.drop(columns=["embeddings"])
    all_data_df.to_csv(save_path + "data_with_tropes_filtered.csv", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_path + "metrics.csv", index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
