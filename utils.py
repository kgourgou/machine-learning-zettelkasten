import re
import numpy as np


def preprocess(doc):
    """
    Remove urls, linebreaks, dashes, equations, html, etc.

    Most of it is done through regular expressions.
    """
    new_doc = doc.replace('\n', ' ')

    operations = [
        r'\$\$(.+?)\$\$', r'\$(.+?)\$', r'\((.+?)\)', 'r\[(.+?)\]', r'\#',
        r'---(.+?)---', r'[\*\-\_\\]', r'<[^>]*>'
    ]

    for op in operations:
        new_doc = re.sub(op, '', new_doc)

    return new_doc


class DocSim:
    def __init__(self, message_embeddings):
        corr = np.inner(message_embeddings, message_embeddings)
        np.fill_diagonal(corr, 0)  # fill diagonal with zeros.
        self.corr = corr

    def return_corr_docs(self, index):
        """
        Returns top-5 correlated documents.

        TODO threshold the correlation instead of returning top-5.
        """
        return self.corr[index, :].argsort()[::-1][0:5]

    def build_graph(self, labels):
        """
        labels: list of str. For example, you can use the
        filenames (with the complete path).

        Should be in the same order as the message embeddings.
        """
        self.graph = {}
        for i, name in enumerate(labels):
            self.graph[name] = [
                labels[key] for key in self.return_corr_docs(i)
            ]


def add_links(doc, rel_links_text, link_section_name = "##### links"):
    """
    Add links to a doc. There is an assumption here
    that the link section will be placed at the
    end of the file.

    doc: str, contains a single document.
    rel_links_text: pre-formatted str with the links.
    """

    link_index = doc.find(link_section_name)
    if link_index >= 0:
        # delete old links section
        doc = doc.replace(doc[link_index:], '')

    # add new links
    return doc + f"\n{link_section_name}:\n" + rel_links_text


def make_text(rel_links):
    """

    rel_links: list of str with the relevant links for a document.
    should be what is returned by DocSim.graph[label].

    >> make_text(['../../a.md','../b.md'])
    "link: [a.md](a.md)\nlink: [b.md](b.md)\n"

    As I have a flat hierarchy, I don't need the
    full path.
    """

    text = ""
    for link in rel_links:
        filename = link.split('/')[-1]
        text += f'link: [{filename}]({filename})\n'
    return text
