# ---------------------------------------------------------------------------------------------------------------------
# Author: Eric Wang
# Sources:
#  1. Clickstream Data: Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).
#     Link: https://link.springer.com/article/10.1007%2Fs00521-018-3523-0
# ---------------------------------------------------------------------------------------------------------------------
# The goal of this project is to classify an e-commerce website's users by their likelihood of making a purchase.
# The data contain various engagement metrics and contextual information from Google Analytics, including time spent on
# a page and the session date's proximity to a holiday. I used a logistic regression model to achieve a 70%
# classification accuracy rate, a high rate given that a user's website engagement is just one facet of their
# consumer behavior.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DATAROOT = '/Users/ericwang/Documents/GitHub datasets/Classify customers by purchasing likelihood/'
EXPORT = '/Users/ericwang/Documents/GitHub/Classify customers by purchasing likelihood/'

# Import the data and check its integrity by creating a summary dataframe and looking for missing values.
# Note that the `Revenue` variable indicates whether a user made a purchase - it is the target variable.
raw = pd.read_csv(DATAROOT + 'online_shoppers_intention.csv')
print(raw.head(5))
assert not (raw.isna().sum() > 0).any()
summary = raw.describe(include=['object', 'float', 'int'])

build = raw
build['Weekend'] = build['Weekend'].astype(int)
build['Revenue'] = build['Revenue'].astype(int)

# Filter out "Other" visitor types because of their ambiguity: they only account for 85 rows out of 12,330.
print(build.query("VisitorType == 'Other'").shape[0])
build = build.query("VisitorType != 'Other'")
build['VisitorType'] = np.where(build['VisitorType'] == 'Returning_Visitor', 1, 0)

# Create preliminary plots for the four highest correlating fields. If necessary, make sure to avoid obvious
# multicollinearity in the field choices (e.g., `ProductRelated` (number of product related pages a user visited) naturally
# has a high correlation with `ProductRelated_Duration` (total time spent on product related pages). Note that
# `PageValues` "represents the average value for a web page that a user visited before completing an e-commerce
# transaction." Since it can also be used as the label, I will not consider it.
print(build.corrwith(build['Revenue']).abs().sort_values(ascending=False).head(15))

plt.clf()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
fig.suptitle('Preliminary Scatter Plots with Pearson Correlations', size=14)

x1, x2, x3, x4 = build['ExitRates'], build['ProductRelated'], build['BounceRates'], build['Administrative']
inputs = [x1, x2, x3, x4]
y = build['Revenue']

i = 0
for ax in (ax1, ax2, ax3, ax4):
    ax.scatter(inputs[i], y)
    ax.set_xlabel(xlabel=(inputs[i].name + ": " + round(inputs[i].corr(y, method='pearson'), 2).astype(str)), fontsize = 12)
    i = i+1

plt.show()
plt.savefig(EXPORT + 'preliminary_scatter_plots.png', dpi=300)

# Note that the dataset is unbalanced. Create a balanced sample to avoid bias in the model.
print(build['Revenue'].value_counts()/build.shape[0])

sample = build.query('Revenue == 1')
sample = sample.append(build.query('Revenue != 1').sample(n=sample.shape[0]))

# Create a logistic regression model with regularization.
x_train, x_test, y_train, y_test = train_test_split(sample[['ExitRates', 'ProductRelated', 'BounceRates', \
    'Administrative', 'VisitorType', 'Informational', 'SpecialDay']],\
    sample['Revenue'], test_size=0.15)

model = LogisticRegression(penalty='l1', C=1, solver='liblinear')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
probabilities = model.predict_proba(x_test)
model.score(x_test, y_test)