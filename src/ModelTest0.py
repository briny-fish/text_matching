import torch
import matchzoo as mz
from matchzoo import preprocessors
import nltk
import typing
import pandas as pd
from loader import *
import numpy as np
from utils import *
from matchzoo.engine.base_metric import ClassificationMetric

class Recall(ClassificationMetric):
    """Recall metric."""

    ALIAS = ['recall', 'rec']

    def __init__(self):
        """:class:`Recall` constructor."""

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate Recall.

        Example:
            >>> import numpy as np
            >>> y_true = np.array([1])
            >>> y_pred = np.array([[0, 1]])
            >>> Recall()(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Recall.
        """
        y_pred = np.argmax(y_pred, axis=1)
        y1size = np.sum(y_true)
        return np.sum((y_pred == y_true) == y_true) / float(y1size)
class f1score(ClassificationMetric):
    """f1score metric."""

    ALIAS = ['f1score', 'f1']

    def __init__(self):
        """:class:`f1score` constructor."""

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate f1score.

        Example:
            >>> import numpy as np
            >>> y_true = np.array([1])
            >>> y_pred = np.array([[0, 1]])
            >>> f1score()(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: f1score.
        """
        y1size = np.sum(y_true==1)
        y_pred = np.argmax(y_pred, axis=1)
        recall =  float(np.sum((y_pred == y_true) == y_true) / float(y1size))
        acc = float(np.sum(y_pred == y_true) / float(y_true.size))
        return 2*(acc*recall)/(acc+recall)

class new_pre(preprocessors.BasicPreprocessor):
    def __init__(self,
                 truncated_mode: str = 'pre',
                 truncated_length_left: int = None,
                 truncated_length_right: int = None,
                 filter_mode: str = 'df',
                 filter_low_freq: float = 1,
                 filter_high_freq: float = float('inf'),
                 remove_stop_words: bool = False,
                 ngram_size: typing.Optional[int] = None):
        """Initialization."""
        super().__init__(truncated_mode=truncated_mode,
            truncated_length_left=truncated_length_left,
            truncated_length_right=truncated_length_right,
            filter_mode=filter_mode,
            filter_low_freq=filter_low_freq,
            filter_high_freq=filter_high_freq,
            remove_stop_words=remove_stop_words,
            ngram_size=ngram_size)
    @classmethod
    def _default_units(cls) -> list:
        """Prepare needed process units."""
        return [
            mz.preprocessors.units.tokenize.Tokenize(),
            mz.preprocessors.units.punc_removal.PuncRemoval(),
        ]

#nltk.download('punkt')
W2V, W2V_VOCAB, word2ix,W2V_value = get_W2V()
print(len(W2V_value))
print(len(W2V_VOCAB))
print(len(word2ix))
#print(W2V_VOCAB[:10])
#print(W2V_value[:10])
#print('-------------')
event_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/same_event_doc_pair.txt'
#print('-------------')
stopwords_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/stopwords-zh.txt'
#print('-------------')
d,datas = read_input(event_file,stopwords_file)
#print('-------------')
Datas = pd.DataFrame(datas)
print(len(Datas))
Datas['label']=Datas['label'].astype(int)
#print(Datas['label'])
class_task = mz.tasks.Classification(num_classes = 2)
class_task.metrics = ['acc','rec','f1']
class_task.losses = ['cross_entropy']
sent_len = 200
print(Datas.iloc[1])
print(len(Datas))
print(Datas.iloc[0])

#print(train_pack.frame()['label'])
#preprocessor = new_pre(truncated_mode='post',truncated_length_left=200,truncated_length_right=200)
#truncated_mode=post，截取truncatedlenth的内容，pre的话就是从后面删掉这么长的内容
#train_processed = preprocessor.fit_transform(data_pack)
#print(len(train_processed.frame()))
#tmp = train_processed.frame()
#valid_out = None
#cnt = 0

#print(len(preprocessor.context['vocab_unit'].state['index_term']))
#vocab_size = len(preprocessor.context['vocab_unit'].state['index_term'])
#Max = 0
#index2term = preprocessor.context['vocab_unit'].state['index_term']
#term2index = preprocessor.context['vocab_unit'].state['term_index']

valid_processed = Datas[int(29063*0.8):]
train_processed = Datas[:int(29063*0.8)]

# 定义数据
#print(train)
#print(len(valid))
train_pack = mz.pack(train_processed,'classification')
train_pack.append_text_length(inplace=True, verbose=1)
valid_pack = mz.pack(valid_processed,'classification')
valid_pack.append_text_length(inplace=True, verbose=1)
#print(train_processed.frame()['text_left'][:10])
print(len(train_pack.frame()))
print(len(valid_pack.frame()))

trainset = mz.dataloader.Dataset(
    data_pack=train_pack,
    mode='point',
    batch_size=7
)
validset = mz.dataloader.Dataset(
    data_pack=valid_pack,
    mode='point',
    batch_size=7
)

padding_callback = mz.models.ESIM.get_default_padding_callback(pad_word_mode='post')

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    stage='dev',
    callback=padding_callback
)

model = mz.models.ESIM()

#model.params['embedding']=np.array(W2V_value)
#model.params['embedding_freeze']=False

model.params['task'] = class_task
model.embedding['embedding']=W2V_value
model.params['embedding_output_dim'] = 50#如果有W2V则会更新成W2V的size（这里是200）
model.params['embedding_input_dim'] = len(W2V_VOCAB)
model.embedding = model._make_default_embedding_layer()
model.guess_and_fill_missing_params()
model.build()
print(model.params)

learn_rate = 4e-4
optimizer = torch.optim.Adam(model.parameters(),lr = learn_rate)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    epochs=60
)

trainer.run()