import numpy as np
import pandas as pd
import tqdm
import sys

def read_stopwords(infile):
    res = {}
    for line in open(infile,encoding='utf-8'):
        line = line.strip()
        res[line] = 1
    return res


def read_input(infile, stopwords):
    lineid = 0
    field_num = 0
    datas = []
    docid = 0
    datasdict = {'text_right':[],'text_left':[],'label':[]}
    for line in open(infile,encoding='utf-8'):
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
        cur_data['doc1']['tokens_without_stopwords'] = [w for w in cur_data['doc1']['tokens'] if w not in stopwords]
        cur_data['doc1']['docid'] = docid
        docid += 1

        cur_data['doc2'] = {}
        cur_data['doc2']['orig'] = fields[6].strip()
        cur_data['doc2']['text'] = cur_data['doc2']['orig'].replace(' ', '')
        cur_data['doc2']['tokens'] = cur_data['doc2']['orig'].split(' ')
        cur_data['doc2']['tokens_without_stopwords'] = [w for w in cur_data['doc2']['tokens'] if w not in stopwords]
        cur_data['doc2']['docid'] = docid
        docid += 1
        datas.append(cur_data)
        datasdict['label'].append(int(fields[0]))
        datasdict['text_left'].append(fields[5].strip())
        datasdict['text_right'].append(fields[6].strip())
    return datas,datasdict
if __name__ == '__main__':
    event_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/same_event_doc_pair.txt'
    stopwords_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/stopwords-zh.txt'

    stopwords = read_stopwords(stopwords_file)
    d1,d2 = event_datas = read_input(event_file, stopwords)
    # story_datas = read_input(story_file, stopwords)
    Dframe = pd.DataFrame(d2)
    print(Dframe[:10])
    print(len(Dframe))
    #print(event_datas)