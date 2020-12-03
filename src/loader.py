import numpy as np
import pandas as pd
import tqdm
import sys
import pickle
import gensim
import os
from collections import OrderedDict
def read_stopwords(infile):
    res = {}
    for line in open(infile,encoding='utf-8'):
        line = line.strip()
        res[line] = 1
    return res


def read_input(infile, stopwords):
    W2V, W2V_VOCAB, word2ix,W2V_value = get_W2V()
    lineid = 0
    field_num = 0
    datas = []
    docid = 0
    datasdict = {'id_left':[],'text_right':[],'id_right':[],'text_left':[],'label':[]}
    for line in open(infile,encoding='utf-8'):
        print(lineid)
        lineid += 1
        if lineid == 1:
            field_num = len(line.split('|'))
            continue
        line = line.strip()
        fields = line.split('|')
        if len(fields) != field_num:
            sys.stderr.write('format error1\t' + line + '\n')
            continue
        cur_data = {}
        cur_data['label'] = int(fields[0])

        cur_data['doc1'] = {}
        cur_data['doc1']['orig'] = fields[5].strip()
        cur_data['doc1']['text'] = cur_data['doc1']['orig'].replace(' ', '')
        cur_data['doc1']['tokens'] = cur_data['doc1']['orig'].split(' ')
        cur_data['doc1']['tokens_without_stopwords'] = [word2ix[w] for w in cur_data['doc1']['tokens'] if w not in stopwords]
        cur_data['doc1']['docid'] = docid
        docid += 1

        cur_data['doc2'] = {}
        cur_data['doc2']['orig'] = fields[6].strip()
        cur_data['doc2']['text'] = cur_data['doc2']['orig'].replace(' ', '')
        cur_data['doc2']['tokens'] = cur_data['doc2']['orig'].split(' ')
        cur_data['doc2']['tokens_without_stopwords'] = [word2ix[w] for w in cur_data['doc2']['tokens'] if w not in stopwords]
        cur_data['doc2']['docid'] = docid
        docid += 1
        datas.append(cur_data)
        datasdict['label'].append(int(fields[0]))
        datasdict['text_left'].append(cur_data['doc1']['tokens_without_stopwords'])
        datasdict['id_left'].append('L-%s' % lineid)
        datasdict['text_right'].append(cur_data['doc2']['tokens_without_stopwords'])
        datasdict['id_right'].append('R-%s' % lineid)
    return datas,datasdict

def load_w2v(fin, type, vector_size):
    """
    Load word vector file.
    :param fin: input word vector file name.
    :param type: word vector type, "Google" or "Glove" or "Company".
    :param vector_size: vector length.
    :return: Output Gensim word2vector model.
    """
    model = {}
    if type == "Google" or type == "Glove":
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fin, binary=True)
    elif type == "Company":
        model["<PAD>"] = np.zeros(vector_size)
        model["<OOV>"] = np.random.uniform(-0.25, 0.25, vector_size)
        with open(fin, "r", encoding="utf-8") as fread:
            for line in fread.readlines():
                line_list = line.strip().split(" ")
                word = line_list[0]
                word_vec = np.fromstring(" ".join(line_list[1:]),
                                         dtype=float, sep=" ")
                model[word] = word_vec
    else:
        print("type must be Glove or Google or Company.")
        sys.exit(1)
    print(type)
    return model
class newdict(dict):
    def __missing__(self, key):
            """Map out-of-vocabulary terms to index 1."""
            return 1
        
    
def transform_w2v(W2V, vector_size):
    W2V = dict((k, W2V[k]) for k in W2V.keys()
               if len(W2V[k]) == vector_size)
    W2V = OrderedDict(W2V)
    W2V_value = [W2V[k] for k in W2V.keys()]
    W2V_VOCAB = W2V.keys()
    W2V_VOCAB = [w for w in W2V_VOCAB]
    word2ix = newdict({word: i for i, word in enumerate(W2V)})
    return W2V, W2V_VOCAB, word2ix,W2V_value
def load_W2V_VOCAB(language):
    if language == "Chinese":
        print("load w2v vocabulary ...")
        W2V_VOCAB_PKL_FILE = "/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/word2vec/w2v-zh.vocab.pkl"
        if not os.path.exists(W2V_VOCAB_PKL_FILE):
            W2V = load_w2v("/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/word2vec/w2v-zh.model",
                           "Company", 200)
            W2V_VOCAB = set(W2V.keys())  # must be a set to accelerate remove_OOV
            pickle.dump(W2V_VOCAB, open(W2V_VOCAB_PKL_FILE, "wb"))
        else:
            W2V_VOCAB = pickle.load(open(W2V_VOCAB_PKL_FILE, "rb"))
        return W2V_VOCAB
    elif language == "English":
        print("load w2v vocabulary ...")
        W2V_VOCAB_PKL_FILE = "../../../../data/raw/Google-w2v/GoogleNews-vectors-negative300.vocab.pkl"
        if not os.path.exists(W2V_VOCAB_PKL_FILE):
            W2V = load_w2v("../../../../data/raw/Google-w2v/GoogleNews-vectors-negative300.bin",
                           "Google", 300)
            W2V_VOCAB = set(W2V.wv.vocab.keys())  # must be a set to accelerate remove_OOV
            pickle.dump(W2V_VOCAB, open(W2V_VOCAB_PKL_FILE, "wb"))
        else:
            W2V_VOCAB = pickle.load(open(W2V_VOCAB_PKL_FILE, "rb"))
        return W2V_VOCAB

def get_W2V():
    W2V = load_w2v("/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/word2vec/w2v-zh.model",
                           "Company", 200)
    return transform_w2v(W2V,200)
if __name__ == '__main__':
    event_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/same_event_doc_pair.txt'
    stopwords_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/stopwords-zh.txt'

    #stopwords = read_stopwords(stopwords_file)
    #d1,d2 = event_datas = read_input(event_file, stopwords)
    # story_datas = read_input(story_file, stopwords)
    #Dframe = pd.DataFrame(d2)
    #print(Dframe[:10])
    #print(len(Dframe))
    #print(event_datas)
    W2V = load_w2v("/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/word2vec/w2v-zh.model",
                           "Company", 200)