"""
Parse a collection of markdown documents, get their embeddings from the
universal sentence encoder, then add similar docs as backlinks at the end
of each markdown document.
"""

from typing import List
import numpy as np
from glob import glob
from tqdm import tqdm

from utils import preprocess, DocSim, add_links, make_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def generate_links(list_of_filenames: List[str], message_embeddings: np.array):

    # compute correlations between docs and construct similarity graph
    print("Computing similarities ...")
    ds = DocSim(message_embeddings)
    ds.build_graph(list_of_filenames)

    return ds


def load_docs(list_of_filenames):
    docs = []
    for filename in list_of_filenames:
        f = open(filename, "r")
        docs.append(f.read())
    return docs


if __name__ == "__main__":
    list_of_filenames = glob("docs/*.md")

    # load docs into memory
    print("Loading docs ...")
    docs = load_docs(list_of_filenames)

    # remove urls, LaTeX, frontmatter, etc. and get embeddings.
    print("Generating embeddings ...")
    new_docs = [preprocess(doc) for doc in docs]

    def embed(docs: List[str]) -> np.array:
        # tfidf vectorizer
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split())
        tfidf_matrix = tfidf.fit_transform(docs)
        # compute and return cosine similarity matrix
        return cosine_similarity(tfidf_matrix)

    message_embeddings = embed(new_docs)
    ds = generate_links(list_of_filenames, message_embeddings)

    WRITE_TO_FILE = True
    if WRITE_TO_FILE:
        # Add backlinks to the documents
        print("Adding backlinks ...")
        for i, name in enumerate(tqdm(list_of_filenames)):
            active_doc = docs[i]
            text = make_text(ds.graph[name])
            new_doc = add_links(active_doc, text)
            with open(name, "w") as fp:
                fp.write(new_doc)
