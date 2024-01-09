# Kaggle TalkingData-AdTracking-Fraud-Detection-Challenge

Kaggle - TalkingData AdTracking Fraud Detection Challenge Solution (Public LB: 0.9810276, Private LB: 0.9819337) <br>

**Model Architecture** - Single GBM Model <br>
**Medal Won** - Bronze(Top 7%) <br>
**Leaderboard** - https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/leaderboard <br>
**Problem Category** - Highly imbalanced Dataset (Less than 1% positive samples) <br>
**Approach** - Downsampling negative saamples + LightGBM (with hyperparameter tuning) <br>

The original training set had more than 180 million rows. After downsampling, my new training set was 940K rows only which had almost equal proportion of positive and negative samples. Other than that, lots of feature engineering was done and all the features were generated using BigQuery, since it turned out to be lightning quick and moreover python gives a huge memory spike when you are merging datframes and in turn memory issues. But that doesn't affect my love for Python, not even a bit.
