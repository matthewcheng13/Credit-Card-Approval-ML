# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf

# set up data
data = pd.read_csv('/crx.data', delimiter=',')
df = pd.DataFrame(data)
df.columns = ['Male','Age','Debt','Married','BankCustomer','EducationLevel','Ethnicity',
              'YearsEmployed', 'PriorDefault','Employed','CreditScore','DriversLicense',
              'Citizen','ZipCode','Income','Approved']
df = df.replace('?','0')

dftrain = df[0:600]
dfeval = df[601:]
ytrain = dftrain.pop('Approved')
yeval = dfeval.pop('Approved')

#set up feature columns
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['Male','Married','BankCustomer','EducationLevel','Ethnicity',
                       'PriorDefault','Employed','DriversLicense','Citizen']
NUMERIC_COLUMNS = ['Age','Debt','YearsEmployed','CreditScore','ZipCode','Income']

def one_hot_cat_column(feature_name, vocab):
  return fc.indicator_column(
      fc.categorical_column_with_vocabulary_list(feature_name,
                                                 vocab))

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  # Need to one-hot encode categorical features.
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(fc.numeric_column(feature_name,
                                           dtype=tf.float64))


# create input functions for model
NUM_EXAMPLES = len(ytrain)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    dataset = (dataset
      .repeat(n_epochs)
      .batch(NUM_EXAMPLES))
    return dataset
  return input_fn

train_input_fn = make_input_fn(dftrain, ytrain)
eval_input_fn = make_input_fn(dfeval, yeval, shuffle=False, n_epochs=1)

#set parameters and then train model
params = {
  'n_trees': 60,
  'max_depth': 4,
  'n_batches_per_layer': 1,
  'center_bias': True
}

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
est.train(train_input_fn, max_steps=100)

# evaluate model
results = est.evaluate(eval_input_fn)
clear_output()
pd.Series(results).to_frame()

# set up classifier
in_memory_params = dict(params)
in_memory_params['n_batches_per_layer'] = 1

def make_inmemory_train_input_fn(X, y):
  y = np.expand_dims(y, axis=1)
  def input_fn():
    return dict(X), y
  return input_fn
train_input_fn = make_inmemory_train_input_fn(dftrain, ytrain)

#train classifier
est = tf.estimator.BoostedTreesClassifier(
    feature_columns,
    train_in_memory=True,
    **in_memory_params)

est.train(train_input_fn)
results = est.evaluate(eval_input_fn)
clear_output()
pd.Series(results).to_frame()

sns_colors = sns.color_palette('colorblind')

# see prediction results
pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))

labels = y_eval.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
df_dfc.describe().T

bias = pred_dicts[0]['bias']
dfc_prob = df_dfc.sum(axis=1) + bias
np.testing.assert_almost_equal(dfc_prob.values,
                               probs.values)

# set up functions for plotting
def _get_color(value):
    """To make positive DFCs plot green, negative DFCs plot red."""
    green, red = sns.color_palette()[2:4]
    if value >= 0: return green
    return red

def _add_feature_values(feature_values, ax):
    """Display feature's values on left of plot."""
    x_coord = ax.get_xlim()[0]
    OFFSET = 0.15
    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):
        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(feat_val), size=12)
        t.set_bbox(dict(facecolor='white', alpha=0.5))
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')
    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue',
    fontproperties=font, size=12)

def plot_example(example):
  TOP_N = 8 
  sorted_ix = example.abs().sort_values()[-TOP_N:].index 
  example = example[sorted_ix]
  colors = example.map(_get_color).tolist()
  ax = example.to_frame().plot(kind='barh',
                          color=[colors],
                          legend=None,
                          alpha=0.75,
                          figsize=(10,6))
  ax.grid(False, axis='y')
  ax.set_yticklabels(ax.get_yticklabels(), size=14)

  _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)
  return ax

# plot individual 121 example
ID = 182
example = df_dfc.iloc[ID]
TOP_N = 8 
sorted_ix = example.abs().sort_values()[-TOP_N:].index
ax = plot_example(example)
ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
ax.set_xlabel('Contribution to predicted probability', size=14)
plt.show()

importances = est.experimental_feature_importances(normalize=True)
df_imp = pd.Series(importances)

# plot gain feature importance
N = 8
ax = (df_imp.iloc[0:N][::-1]
    .plot(kind='barh',
          color=sns_colors[0],
          title='Gain feature importances',
          figsize=(10, 6)))
ax.grid(False, axis='y')

# plot mean feature contributions
dfc_mean = df_dfc.abs().mean()
N = 8
sorted_ix = dfc_mean.abs().sort_values()[-N:].index 
ax = dfc_mean[sorted_ix].plot(kind='barh',
                       color=sns_colors[1],
                       title='Mean |directional feature contributions|',
                       figsize=(10, 6))
ax.grid(False, axis='y')