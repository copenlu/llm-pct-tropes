from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from lexical_diversity import lex_div as ld

class EmbeddingsDiversity(object):
    def __init__(
        self, texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32
    ):
        device = 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        self.device = device
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.embeddings = self.embedder.encode(
            texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        )

    def average_cosine_dist(self) -> float:
        """
        Measures the diversity by calculating the average cosine distance of every embedding to the rest of the corpus
        """
        similarities = util.dot_score(
            self.embeddings, self.embeddings
        )  # A matrix of similariries, each appears twice
        similarities = similarities.cpu().numpy()  # Convert to numpy array
        similarities *= 1 - np.tri(*similarities.shape)  # zero out upper triagonal and diagonal]
        
        average_similarity = similarities.sum() / (similarities != 0).sum()  # mean of non-zero elements
        average_distance = 1 - average_similarity
        return average_distance

    def average_closest_neighbour(self) -> float:
        """
        Measures the diversity by calculating the average cosine distance of every embedding to the rest of the corpus
        """
        similarities = util.dot_score(
            self.embeddings, self.embeddings
        )  # A matrix of similariries, each appears twice
        similarities = similarities.cpu().numpy()  # Convert to numpy array
        similarities *= 1 - np.eye(similarities.shape[0])  # zero out the diagonal
        
        similarities = np.max(similarities, axis=1)
        average_similarity = np.mean(similarities)
        average_distance = 1 - average_similarity
        return average_distance
    
    def full_report(self) -> Dict[str, float]:
        """
        Returns a dictionary with all diversity metrics
        """
        diversity_metrics = {
            "average_cosine_dist": self.average_cosine_dist(),
            "average_closest_neighbour": self.average_closest_neighbour(),
        }
        return diversity_metrics


class LexicalDiversity(object):
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.self_bleu_weights = [
            (1.0 / 2.0, 1.0 / 2.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0),
        ]

    def self_bleu(self) -> Dict[str, float]:
        """
        Measures the diversity by calculating the self BLEU score
        """
        total_bleu = np.zeros(len(self.self_bleu_weights))
        texts = [ld.tokenize(text) for text in self.texts]
        for i in range(len(texts)):
            hypothesis = texts[i]
            references = texts[:i] + texts[i + 1 :]
            total_bleu += sentence_bleu(
                references=references,
                hypothesis=hypothesis,
                smoothing_function=SmoothingFunction().method1,
                weights=self.self_bleu_weights,
            )
        self_bleu_score = total_bleu / len(self.texts)
        self_bleu_score = 1 - self_bleu_score
        self_bleu_score = {f"SELF_BLEU_{i+2}": self_bleu_score[i] for i in range(len(self_bleu_score))}
        return self_bleu_score
    
    def n_gram_entropy(self, min_gram: int = 1, max_gram: int = 2) -> float:
        """
        Measures the diversity by calculating the normalised n-gram entropy of the corpus
        """
        vectorizer = CountVectorizer(ngram_range=(min_gram, max_gram))
        document = " ".join(self.texts)
        n_gram_counts = vectorizer.fit_transform([document])
        n_gram_counts = n_gram_counts.toarray()[0]
        n_gram_counts = n_gram_counts / n_gram_counts.sum()
        n_gram_entropy = entropy(n_gram_counts)
        normalised_entropy = n_gram_entropy / np.log(len(document.split()))
        return normalised_entropy
    
    def ld_report(self) -> Dict[str, float]:
        """
        Lexical diversity using the functions defined in lex_div
        """
        texts = [ld.flemmatize(text) for text in self.texts]
        texts = sum(texts, start=[])
        report = {"ttr": ld.ttr(texts),
                  "root_ttr": ld.root_ttr(texts),
                  "log_ttr": ld.log_ttr(texts),
                  "maas_ttr": ld.maas_ttr(texts),
                  "mattr": ld.mattr(texts),
                  "hdd": ld.hdd(texts),
                  "mtld": ld.mtld(texts),
                  "mtld_ma_wrap": ld.mtld_ma_wrap(texts),
                  "mtld_ma_bid": ld.mtld_ma_bid(texts)}
        return report
    
    def full_report(self) -> Dict[str, float]:
        """
        Full report of diversity metrics
        """
        report = self.ld_report()
        report.update(self.self_bleu())
        report.update({"ngram_entropy": self.n_gram_entropy()})
        return report