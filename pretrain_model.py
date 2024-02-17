import pandas as pd
from gensim.models import Word2Vec
data=pd.read_csv('data/Davis/Davis_protein_mapping.csv')
protein=data['PROTEIN_SEQUENCE']
def seq_to_kmers(seq, k=3):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]
class Corpus(object):
    """ An iteratable for training seq2vec models. """

    def __init__(self,data, ngram):
        self.df = data
        self.ngram = ngram

    def __iter__(self):
        for no, data in enumerate(self.df):
            yield  seq_to_kmers(data,self.ngram)
sent_corpus = Corpus(protein,3)
model = Word2Vec(vector_size=100, window=5, min_count=0, workers=6)
model.build_vocab(sent_corpus)
model.train(sent_corpus,epochs=30,total_examples=model.corpus_count)
model.save("word2vec_Davis.model")
#model=Word2Vec.load('word2vec_30_DrugBank.model')
print('model end!')