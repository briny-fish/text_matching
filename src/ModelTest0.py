import torch
import matchzoo as mz
import nltk
import pandas as pd
from loader import *

#nltk.download('punkt')
W2V, W2V_VOCAB, word2ix,W2V_value = get_W2V()
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
Datas['label']=Datas['label'].astype(int)
#print(Datas['label'])
class_task = mz.tasks.Classification(num_classes = 2)
class_task.metrics = ['acc']
class_task.losses = ['cross_entropy']

data_pack = mz.pack(Datas,'classification')
#print(data_pack.frame()['label'])
train = data_pack.frame()[:int(29063*0.8)]
valid = data_pack.frame()[int(29063*0.8):]
# 定义数据
print(train)
print(len(valid))
train_pack = mz.pack(train,'classification')
valid_pack = mz.pack(valid,'classification')
#print(train_pack.frame()['label'])
preprocessor = mz.models.ESIM.get_default_preprocessor(truncated_mode='post',truncated_length_left=200,truncated_length_right=200)
#truncated_mode=post，截取truncatedlenth的内容，pre的话就是从后面删掉这么长的内容
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)
#print(train_processed.frame()['text_left'][:10])
print(len(train_processed.frame()))
print(len(valid_processed.frame()))
tmp = train_processed.frame()
valid_out = None
cnt = 0
print(preprocessor.context['vocab_unit'].state.keys())
for i in range(10):
    for j in range(len(tmp['text_left'].iloc[i])):
        print(tmp['text_left'].iloc[i][j])
        if(preprocessor.context['vocab_unit'].state['index_term'][tmp['text_left'].iloc[i][j]] in W2V_VOCAB):
            tmp['text_left'].iloc[i][j]=word2ix[preprocessor.context['vocab_unit'].state['index_term'][tmp['text_left'].iloc[i][j]]]
        else:
            tmp['text_left'].iloc[i][j]=2
        print(W2V_VOCAB[tmp['text_left'].iloc[i][j]])
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
model.params['embedding']=np.array(W2V_value)
model.embedding = model._make_default_embedding_layer()
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

#trainer.run()