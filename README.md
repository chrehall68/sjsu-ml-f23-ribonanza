# SJSU ML Club's Ribonanza Project
SJSU ML Club's attempt at the [Kaggle Ribonanza project](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/) hosted by Stanford.

To run our project, simply run `python main.py`.

## About the project
- We used a continuous transformer architecture for our model. After comparisons with
LSTM, Convolutional, and plain Feedforward models, the transformer architecture performed
the best.
We used a variety of techniques to make and improve our model.
- We used base-pair probability matrices to boost our model's performance. Base-pair
probability matrices represent the probability that a certain base in an RNA strand
will be paired with another base. Thus, it makes sense that this would help our
model predict an RNA strand's 3D structure 
- We used pseudo-labeling to train on the test dataset, boosting our leaderboard score
by around `0.002`

## Members
- Haydon Behl
- Neal Chandra
- Eliot Hall
- Atharva Jadhav

## Score
Our best score was slightly less than 0.15
