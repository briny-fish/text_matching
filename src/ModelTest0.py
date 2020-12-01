import torch
import matchzoo as mz
import nltk
import pandas as pd
from loader import *
#nltk.download('punkt')
print('-------------')
event_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/same_event_doc_pair.txt'
print('-------------')
stopwords_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/stopwords-zh.txt'
print('-------------')
d,datas = read_input(event_file,stopwords_file)
print('-------------')
Datas = pd.DataFrame(datas)
Datas['label']=Datas['label'].astype(int)
print(Datas['label'])
class_task = mz.tasks.Classification(num_classes = 2)
class_task.metrics = ['acc']
class_task.losses = ['cross_entropy']

data_pack = mz.pack(Datas,'classification')
print(data_pack.frame()['label'])
train = data_pack.frame()[data_pack.frame()['id_left'] <= ("L-%s" % int(29063 * 0.8))]
valid = data_pack.frame()[data_pack.frame()['id_left'] > ("L-%s" % int(29063 * 0.8))]
# 定义数据

train_pack = mz.pack(train,'classification')
valid_pack = mz.pack(valid,'classification')
print(train_pack.frame()['label'])
preprocessor = mz.models.ESIM.get_default_preprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)
print(train_processed.frame()['text_left'][:10])
#print(len(train_processed[train_processed.frame()['label']==1]))
#print(train_processed.frame()[:2]['text_right'])
trainset = mz.dataloader.Dataset(
    data_pack=train_processed,
    mode='point',
    batch_size=64
)
validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point',
    batch_size=64
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
model.params['embedding_output_dim'] = 100
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