# Step 9: Scale Your ML (DL) Prototype

To scale the model prototype, the easiest way to scal the model prototype is containerization.
Containerization is a software deployment process that bundles an application’s code with all the files and libraries it needs to run on any infrastructure. Traditionally, to run any application on your computer, you had to install the version that matched your machine’s operating system. For example, you needed to install the Windows version of a software package on a Windows machine. However, with containerization, you can create a single software package, or container, that runs on all types of devices and operating systems. 

1. Create docker file
```bash
# base image
FROM python:3.7.4-slim-stretch

# exposing default port for streamlit
EXPOSE 8502

# making directory of app
WORKDIR /streamlit-Transformer-RNA-classifier-app

# copy over requirements
COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip3 install -r requirements.txt

# copying all files over
COPY . .

# download model file
RUN apt-get update
RUN apt-get  -qq -y install wget
RUN wget -O ./model/Bert-btc-model.pth "https://www.dropbox.com/s/0nq3ukfhe99t9pv/model.pth?dl=1"

# cmd to launch app when container is run
CMD streamlit run ./src/app.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
```

2. Create requirements file

```python
-f https://download.pytorch.org/whl/torch_stable.html 
torch==1.10.1+cpu
streamlit==1.4.0
transformers
pandas
numpy
tqdm
```

3. Create python functions file

```python
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import json
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        returned = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask)
        pooled_output = returned["pooler_output"]
        output = self.drop(pooled_output)
        return self.out(output)

@st.cache
def load_model(model_dir="https://www.dropbox.com/s/0nq3ukfhe99t9pv/model.pth?dl=1"):    
    model = SentimentClassifier(3).to(device)
    with open(model_dir, "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def input_fn(input_data, content_type= 'application/json'):
    input = json.loads(input_data)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    review = input["text"]
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=300,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt')
    input_ids = encoding['input_ids'].flatten()
    attention_mask = encoding['attention_mask'].flatten()
    return {'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten()}


def run_model(input, model):
    with torch.no_grad():       
        prediction = model(input['input_ids'].unsqueeze(0).to(device),input['attention_mask'].unsqueeze(0).to(device))
        _,result = torch.max(prediction, dim=1)
        result = result.cpu().numpy()[0]
        return(result)
```

4. Create python streamlit app file
```python
import streamlit as st
from funcs import load_model, input_fn, run_model

st.title("Bert-based bitcoin social media sentiment analysis")
st.write('This app uses Bert to identify the sentiment of any bitcoin social media')
st.markdown('The source code for this app can be found in this GitHub repo: [GPT2-News-Classifier](https://github.com/vveizhang/transformer_predict_circRNA).')

example_text = """
"I think bitcoin is worthless. I will never buy it"
"""

input_text = st.text_area(
    label="Input/Paste News here:",
    value="",
    height=30,
    placeholder="Example:{}".format(example_text)
    )

# load model here to save
model = load_model()

if input_text == "":
    input_text = example_text

if st.button("Run Bert!"):
    if len(input_text) < 300:
        st.write("Please input more text!")
    else:
        with st.spinner("Running..."):

            model_input = input_fn(input_text)
            model_output = run_model(model, *model_input)
            st.write("Predicted sentiment:")
            st.write(model_output)
```


5. Build docker image file
Then, copy the code into the cloud using git:

git clone https://github.com/vveizhang/Bitcoin_Social_Media_Sentiment_Analysis.git
Afterwards, go into the ec2-docker folder to build and run the image:
```bash
cd ec2-docker/
docker image build -t streamlit:BertSentiment .
```




