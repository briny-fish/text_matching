import torch
import matchzoo as mz
import nltk
from loader import read_input
#nltk.download('punkt')
event_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/same_event_doc_pair.txt'
stopwords_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/stopwords-zh.txt'
datas = read_input(event_file,stopwords)
class_task = mz.tasks.Classification(num_classes = 2)
class_task.metrics = ['acc']
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

#trainer.run()