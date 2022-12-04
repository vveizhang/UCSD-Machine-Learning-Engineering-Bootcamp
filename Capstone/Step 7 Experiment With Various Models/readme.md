<p align="center">
<br>
</p>

## Step 7: Experiment With Various Models


You can also check the finished [Sagemaker notebook](https://github.com/vveizhang/Bitcoin_Social_Media_Sentiment_Analysis/blob/main/src/pyTorchInference.ipynb) here:

#### Use wandb sweep grid serach to optimize the hyperparameter

```python
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        returned = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask)
        pooled_output = returned["pooler_output"]
        output = self.drop(pooled_output)
        return self.out(output)

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
data_loader = create_data_loader(df, tokenizer, BATCH_SIZE, max_len=300)
```


```python
import wandb
wandb.login()
sweep_config = {'method': 'grid'}
metric = {'name': 'val_acc','goal': 'maximize'}
sweep_config['metric'] = metric
parameters_dict = {
    'optimizer': {'values': ['adam', 'sgd',"AdamW"]},
    'learning_rate': {'values': [5e-3, 1e-4, 3e-5, 6e-5, 1e-5]},
    'epochs': {'values': [2,4,6,8,10]}}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="pytorch-sweep")

def train(config=None):
  with wandb.init(config=config):
    config = wandb.config

    EPOCHS = config.epochs
    model = SentimentClassifier(3).to(device)
    optimizer = build_optimizer(model,config.optimizer,config.learning_rate)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
      print(f'Epoch {epoch + 1}/{EPOCHS}')
      print('-' * 10)
      train_acc, train_loss = train_epoch(
        model,train_data_loader,loss_fn,optimizer,device,scheduler,len(df_train))
      print(f'Train loss {train_loss} accuracy {train_acc}')
      val_acc, val_loss = eval_model(model,test_data_loader,loss_fn,device,len(df_test))
      print(f'Val   loss {val_loss} accuracy {val_acc}')
        
wandb.agent(sweep_id, train)
```
The wandb will generate a parallel coordinates plot, a parameter importance plot, and a scatter plot when you start a W&B Sweep job. 

<p align="center">
<img src="/imgs/para_coord-1127.png">
<br>
<em>parallel coordinates plot</em></p>

The best model: {'epochs':8,
                  'learning_rate':3e-5,
                    'optimizer': "Adam"}