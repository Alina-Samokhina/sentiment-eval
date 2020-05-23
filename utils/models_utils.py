import numpy as np
from gensim.models import KeyedVectors


def get_pretrained_embeddings(general_emb_file = 'fasttext_rus.vec'):
    '''
    We use pretrained fasttext embeddings from deeppavlov.
    This function load them from file
    '''
    ft_model = KeyedVectors.load_word2vec_format(general_emb_file)
    unk_weights_ft = np.mean(ft_model.wv.vectors, axis = 0)
    pad_weights_ft = np.zeros(300)
    ft_model.add('<UNK>', unk_weights_ft)
    ft_model.add('<PAD>', pad_weights_ft)

    return ft_model
