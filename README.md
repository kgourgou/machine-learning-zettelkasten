# Model-based zettelkasten

I was reading about the [zettelkasten method](https://zettelkasten.de/) of
taking notes recently. Taking some time to think about connections between
different notes seems to me to be quite useful in order to generate new ideas. 
However, I'm quite lazy at this point to go through my archive of markdown notes
and connect them to each other. NLP to the rescue! 

This code grabs a [Universal Sentence
Encoder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)
model from tensorflow hub, then loads the markdown files, does some light
preprocessing to get rid of LaTeX, urls, etc., and generates embeddings. With
those, I can then compare the similarity of documents and create backlinks from
one doc to another. Then, I can jump between docs while reading
with Typora.


The code is simple and `run_me.py` is the main script. Be sure to change the paths to where you have your own docs. 

**This code overwrites the original markdown documents with no backup. Be sure to backup your docs before using.**

I haven't tested a lot, so expect bugs. 
