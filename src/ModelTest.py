import torch
import matchzoo as mz
import nltk
import numpy as np
from loader import read_input
from utils import *
import numpy as np
from matchzoo.engine.base_metric import ClassificationMetric
#nltk.download('punkt')


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
        return np.sum(np.logical_and((y_pred == y_true),y_true)) / float(y1size)
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
        recall =  float(np.sum(np.logical_and((y_pred == y_true),y_true)) / float(y1size))
        acc = float(np.sum(y_pred==y_true))/float(y_true.size)
        return 2*(acc*recall)/(acc+recall)
#event_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/same_event_doc_pair.txt'
#stopwords_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/stopwords-zh.txt'
#datas = read_input(event_file,stopwords)
class_task = mz.tasks.Classification(num_classes = 2)
class_task.metrics = ['acc','rec','f1']
class_task.losses = ['cross_entropy']
train_pack = mz.datasets.wiki_qa.load_data('train',task=class_task)
valid_pack = mz.datasets.wiki_qa.load_data('dev',task=class_task)


preprocessor = mz.models.ESIM.get_default_preprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)
#print(len(train_processed[train_processed.frame()['label']==1]))
#print(train_processed.frame()[:2]['text_right'])
trainset = mz.dataloader.Dataset(
    data_pack=train_processed,
    mode='point',
    batch_size=512
)
validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point',
    batch_size=512
)

padding_callback = mz.models.ESIM.get_default_padding_callback()

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
model.params['task'] = class_task
model.params['embedding_output_dim'] = 50
model.params['embedding_input_dim'] = preprocessor.context['embedding_input_dim']
model.guess_and_fill_missing_params()
model.build()
print(model.params)

learn_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr = learn_rate)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    epochs=10
)

trainer.run()