# Step 2: Collect Your Data

## Introduction

To obtain data for my own project, I did some google search and explored some related dataset out there:


## Three datasets that I have explored:

https://aws.amazon.com/marketplace/pp/prodview-hncbgbk6sb2qs#overview
This dataset, powered by the Twitter API , contains a curated set of cryptocurrency related Tweets from the past month. It covers a rolling 1-month window and will be updated at least once per week on Mondays to provide recent Tweets from the past week. It is limited to English language Tweets from verified accounts  and excludes Retweets for the purpose of keeping each record in the dataset unique. 

https://data.world/mercal/btc-tweets-sentiment
These are extracted bitcoin tweets from a day sample that were originally scored for sentiment. The last two columns (New_sentiment_score, New_sentiment_state) were added from a NLP trained model to compare against the original sentiments.

https://data.world/lexyr/reddit-cryptocurrency-data-for-august-2021
The collection of posts/comments on various crypto subreddits for August 2021. The cryptocurrency community is a growing force in the Internet, especially now as DeFi-based transactions gain worldwide acceptance. This dataset aims to help analyze public sentiment on various cryptocurrencies over a given month.
The dataset is a comprehensive list of all the posts and comments made on several of Reddit's cryptocurrency boards from August 1 to August 31 of 2021.
 
Most of the datasets out there didn’t cover enough long period of time for the project. So I decided to get the data myself using the PushshiftAPI. 

Here is the link to the notebook: [Step2_CollectData.ipynb](https://github.com/vveizhang/UCSD-Machine-Learning-Engineering-Bootcamp/blob/main/Capstone/Step2_Collect_Data/Step2_CollectData.ipynb)