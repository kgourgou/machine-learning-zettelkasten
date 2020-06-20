"""
Parse a collection of markdown documents, get their embeddings from the
universal sentence encoder, then add similar docs as backlinks at the end
of each markdown document.
"""

import tensorflow_hub as hub
from glob import glob
from tqdm import tqdm

from utils import preprocess, DocSim, add_links, make_text

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)


def embed(input):
    return model(input)


filenames = glob('../my-markdown-notes/*.md')
filenames.remove('../my-markdown-notes/README.md')
filenames

# load docs into memory
docs = []
for filename in filenames:
    f = open(filename, 'r')
    docs.append(f.read())

clean_filenames = [name.split('/')[2].replace('.md', '') for name in filenames]

# remove urls, LaTeX, frontmatter, etc. and get embeddings.
new_docs = [preprocess(doc) for doc in docs]
message_embeddings = embed(new_docs)

# compute correlations between docs and construct similarity graph
ds = DocSim(message_embeddings)
ds.build_graph(filenames)

# Add backlinks to the documents
for i, name in enumerate(tqdm(filenames)):
    active_doc = docs[i]
    text = make_text(ds.graph[name])
    new_doc = add_links(active_doc,text)
    with open(name,'w') as fp:
        fp.write(new_doc)
