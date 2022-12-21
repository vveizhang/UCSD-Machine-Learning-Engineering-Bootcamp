<p align="center">
<br>
</p>

# Step 12: Design Your Deployment Solution Architecture

## 1. diagram of the deployment architecture

<p align="center">
<img src="https://github.com/vveizhang/UCSD-Machine-Learning-Engineering-Bootcamp/blob/main/Capstone/Step%2012%20Design%20Your%20Deployment%20Solu/imgs/diagram.png">
<br>
<em>diagram of the deployment architecture</em></p>

## 2. the deployment architecture and automation

I would like to build a automatic prediction system, which will automaticly web scraping reddit comments data and Bitcoin price data of that day, and predicti the Bitcoin price next day. Then output the price plot in a dashboard. I will use AWS Lambda, Event bridge and EC2 crontab to achive automation.

### 2.1 AWS Event bridge triger Lambda to scrape the reddit comments and save to AWS S3.

```python
def data_prep_comments(term, start_time, end_time, filters, limit):
    if (len(filters) == 0):
        filters = ['id', 'author', 'created_utc','body', 'permalink', 'subreddit']

    comments = list(api.search_comments(
        q=term, after=start_time,before=end_time, filter=filters,limit=limit))       
    return pd.DataFrame(comments)
    
def lambda_handler(event, context):
    df = data_prep_comments("bitcoin", start_time=int(dt.datetime(int(yesterday.strftime("%Y")),int(yesterday.strftime("%m")),int(yesterday.strftime("%d")), 0,1).timestamp()), 
                            end_time=  int(dt.datetime(int(yesterday.strftime("%Y")),int(yesterday.strftime("%m")),int(yesterday.strftime("%d")), 23,59).timestamp()),filters = [], limit = limit)
    
    bucket = 'bert-btc'
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, f'daily_comments/df{yesterday.strftime("%Y-%m-%d")}.csv').put(Body=csv_buffer.getvalue())
```

### 2.2 AWS EC2 Crontab automaticly run "main.py".
Here is the code for Crontab automation.
```bash
crontab -e
30 2 * * * python3 run main.py
```
main.py
```python
if __name__ == "__main__":
    yest = yesterday.strftime("%Y-%m-%d")

    bucket = bucket
    csv_path = os.path.join(bucket,data_key)
    df = wr.s3.read_csv(path=csv_path)
    df.to_csv("df.csv", index=False)

    BATCH_SIZE = 16
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    data_loader = create_data_loader(df, tokenizer, BATCH_SIZE, max_len=512)

    wr.s3.download(path = "s3://bucket/model.pth",local_file='model.pth')
    trained_model = SentimentClassifier(3)
    trained_model.load_state_dict(torch.load("model.pth"))

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(trained_model,data_loader)
    df['prediction'] = y_pred
    df = df[['body','created_utc','datetime','prediction']]
    df.to_csv(f"/home/ubuntu/Bert/df_predicted.csv")

    write_key =  f"daily_prediction/df{yest}.csv"
    write_path = os.path.join(bucket,write_key)
    wr.s3.to_csv(df,path=write_path)

    df2 = df[['datetime','text','prediction']]
    write_WC_key = f"daily_wordcloud/df{yest}.csv"
    write_WC_path = os.path.join(bucket,write_WC_key)
    wr.s3.to_csv(df2,path=write_WC_path)
    df_pred_count = prediction_count()
    wr.s3.to_csv(df_pred_count,path=f's3://bucket/daily_sentiCounts/sentiCount{yest}_2h.csv')
    word_cloud(df2)

    btcData2h = get_BTC_data()
    btcData2h.to_csv(f"/home/ubuntu/Bert/dateBTC2h.csv")
    wr.s3.to_csv(btcData2h,path=f's3://bucket/daily_BTCdata/date{yest}BTC2h.csv')

    dataToday = pd.merge(btcData2h,df_pred_count, on ='dateHour')
    dataToday.to_csv("/home/ubuntu/Bert/redditBTC_2h.csv")
    dataPast = pd.read_csv("/home/ubuntu/Bert/redditBTC_20210331_current.csv")
    dataToday = pd.read_csv("/home/ubuntu/Bert/redditBTC_2h.csv")
    data = pd.concat([dataPast,dataToday])
    data.to_csv("/home/ubuntu/Bert/redditBTC_20210331_current.csv",index=False)
    wr.s3.to_csv(data,path=f's3://bucket/BTCplusReddit/data{yest}.csv')
    predictPrice()
```
This python script will do the sentimental prediction of reddit comments, upload the result to S3; download the historical bitcoin price data of that day, use LSTM model to predict the price for the next day; output the predicted price plot to a [dashboard](http://18.224.251.221:8080/), as well as the wordcloud of the reddit comment that day.
