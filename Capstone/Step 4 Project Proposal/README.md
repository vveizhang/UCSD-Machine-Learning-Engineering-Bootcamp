# Step 4: Project Proposal

Bitcoin is a digital currency which operates free of any central control or the oversight of banks or governments.It has a distributed network system, where people can control their funds in a transparent way. It is the leading cryptocurrency and has the highest market capitalization among digital currencies. Unlike repeating phenomena like weather, cryptocurrency values do not follow a repeating pattern and mere past value of Bitcoin does not reveal any secret of future Bitcoin value. Humans follow general sentiments and technical analysis to invest in the market. Hence Sentiment is an important factor, considering people’s sentiment can improve the prediction of bitcoin price. 

This project will try to get sentiment data from various influencers often over social media platforms, and try to improve the prediction of bitcoin price.

The social media comments data will be from reddit. The “PushshiftAPI” will be employed to scrape the reddit comments regarding bitcoin. The historical data of bitcoin prices will come from the “cryptocompare API”. 

The approach outline of this project will be:
Web scraping historical reddit comments regarding bitcoin using the “Pushshift” API.
Manually label about 3 thousands of comments with three sentimental classes: Positive, Neutral, Negative.
Use a pre-trained BERT(Bidirectional Encoder Representations from Transformers) model and fine-tuning the model using the previously manually labeled data to do the sentiment analysis. Also try other models to compare and choose the best one. This is a classification problem.
Employ the best model to do the sentiment analysis on the historical reddit comments data.
Deploy the best model as a web API.
Get the historical bitcoin price data, combine the price data and sentiment data according to the time.
Try to build a model which uses current bitcoin price data and social media sentiment data to predict the price of bitcoin in the future. This is a regression problem.
Deploy the model to predict bitcoin price.
Create an automation cloud system that automatically walks through the web data scraping, sentiment analysis and price prediction on a daily manner.
Create a dashboard to show the daily-updating bitcoin price prediction and historical price, as well as show the word cloud of daily different sentimental reddit comments regarding bitcoin.

The final deliverable of the project will be a dashboard to show the daily-updating bitcoin price prediction and word cloud of daily reddit comments, as well as a web API for the sentiment analysis regarding bitcoin comments.

Since the NLP computing runs much faster using GPUs, the BERT model will need some GPU computing power. The rest of the computing can be done using CPU-based cloud computing.




