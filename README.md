# Model-based zettelkasten

I was reading about the [zettelkasten method](https://zettelkasten.de/) of
taking notes recently. Taking some time to think about connections between
different notes seems to me to be quite useful in order to generate new ideas.
However, I'm quite lazy at this point to go through my archive of markdown notes
and connect them to each other. NLP to the rescue!

This code uses sklearn and cosine similarity to insert backlinks to a collection of markdown files. Presto -- automatic zettelkasten!

The code is simple and `run_me.py` is the main script. Be sure to change the paths to where you have your docs.

**This code overwrites the original markdown documents with no backup. Be sure to backup your docs before using.**

I haven't tested a lot, so expect bugs.
