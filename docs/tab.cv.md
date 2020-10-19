# Cross Validation (Intermediate)
> How to perform various Cross Validation methodologies






---
This article is also a Jupyter Notebook available to be run from the top down. There
will be code snippets that you can then run in any environment.

Below are the versions of `fastai`, `fastcore`, `scikit-learn`, and `iterative-stratification` currently running at the time of writing this:
* `fastai`: 2.0.14 
* `fastcore`: 1.0.14 
* `scikit-learn`: 0.22.2.post1 
* `iterative-stratification`: 0.1.6 
---



# Introduction

In this tutorial we will show how to use various cross validation methodologies inside of `fastai`  with the `tabular` and `vision` libraries. First, let's walk through a `tabular` example

# Tabular

## Importing the Library and the Dataset

We'll be using the `tabular` module for the first example, along with the `ADULTS` dataset. Let's grab those:

```python
from fastai.tabular.all import *
```

```python
path = untar_data(URLs.ADULT_SAMPLE)
```

Let's open it in `Pandas`:

```python
df = pd.read_csv(path/'adult.csv')
```

```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>Private</td>
      <td>101320</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Married-civ-spouse</td>
      <td>NaN</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>1902</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>Private</td>
      <td>236746</td>
      <td>Masters</td>
      <td>14.0</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>10520</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>96185</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Divorced</td>
      <td>NaN</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>Self-emp-inc</td>
      <td>112847</td>
      <td>Prof-school</td>
      <td>15.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>Self-emp-not-inc</td>
      <td>82297</td>
      <td>7th-8th</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>
</div>



Next we want to create a constant test set and declare our various variables and `procs`. We'll just be using the last 10% of the data, however figuring out how to make your test set is a very important problem. To read more, see Rachel Thomas' article on [How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/).
{% include note.html content='we call it a test set here as we make our own mini validation sets when we&#8217;re training' %}

```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
```

And now we'll split our dataset:

```python
print(f'10% of our data is {int(len(df) * .1)} rows')
```

    10% of our data is 3256 rows


```python
start_val = len(df) - 3256; start_val
```




    29305



```python
train = df.iloc[:start_val]
test = df.iloc[start_val:]
```

Now that we have the `DataFrames`, let's look into a few different CV methods:

## K-Fold

Every Cross Validation method is slightly different, and what version you should use depends on the dataset you are utilizing. The general idea of Cross Validation is we split the dataset into `n` sets (usually five is enough), train five seperate models, and then at the end we can ensemble them together. This should in theory make a group of models that performs better than one model on the entire dataset.

As we are training, there is zero overlap in the validation sets whatsoever. As a result we create five distinct validation sets.

### Introduction

Now for the `kfold`. We'll first be using `sklearn`'s `KFold` class. This method works by running through all the indicies available and seperating out the folds. For a minimum example, take the following:

```python
train_idxs = list(range(0,9))
test_idxs = [10]
```

We now have some training indicies and a test set:

```python
train_idxs, test_idxs
```




    ([0, 1, 2, 3, 4, 5, 6, 7, 8], [10])



Now we can instantiate a `KFold` object, passing in the number of splits, whether to shuffle the data before splitting into folds, and potentially a seed:

```python
from sklearn.model_selection import KFold
```

```python
dummy_kf = KFold(n_splits=5, shuffle=False); dummy_kf
```




    KFold(n_splits=5, random_state=None, shuffle=False)



And now we can run through our splits by iterating through train and valid indexes. We pass in our `x` data through `dummy_kf.split` to get the indexes 
> You could also pass in your `y`'s intead:

```python
for train_idx, valid_idx in dummy_kf.split(train_idxs):
    print(f'Train: {train_idx}, Valid: {valid_idx}')
```

    Train: [2 3 4 5 6 7 8], Valid: [0 1]
    Train: [0 1 4 5 6 7 8], Valid: [2 3]
    Train: [0 1 2 3 6 7 8], Valid: [4 5]
    Train: [0 1 2 3 4 5 8], Valid: [6 7]
    Train: [0 1 2 3 4 5 6 7], Valid: [8]


### Extra Preprocessing

Now the question is how can I use this when training on our data?

When we preprocess our tabular training dataset, we build our `procs` based upon it. When doing a CV (Cross Validation) we will often exclude some data as it gets pushed to the validation set, leading to such errors as:


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-16-59d14331bcbc> in <module>()
          1 #hide_input
    ----> 2 raise AssertionError('nan values in `education-num` but not in setup training set')
    

    AssertionError: nan values in `education-num` but not in setup training set


So how do we fix this? We should preprocess the entire training `DataFrame` into [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) first, this way we can extract all the `proc` information. Let's do that now:

```python
to_base = TabularPandas(train, procs, cat_names, cont_names, y_names='salary')
```

Next we need to extract all the information we need. This includes:
* [`Categorify`](https://docs.fast.ai/tabular.core#Categorify)'s classes
* [`Normalize`](https://docs.fast.ai/data.transforms#Normalize)'s `means` and `stds`
* [`FillMissing`](https://docs.fast.ai/tabular.core#FillMissing)'s `fill_vals` and `na_dict`

```python
classes = to_base.classes
means, stds = to_base.normalize.means, to_base.normalize.stds
fill_vals, na_dict = to_base.fill_missing.fill_vals, to_base.fill_missing.na_dict
```

Now we could generate new procs based on those and apply them to our dataset:

```python
procs = [Categorify(classes), Normalize.from_tab(means, stds), FillMissing(fill_strategy=FillStrategy.median, fill_vals=fill_vals, na_dict=na_dict)]
```

### Now Let's Train

Now that we have our adjusted `procs`, let's try training.

We'll want to make a loop that will do the following:

1. Make our `KFold` and split
2. Build a [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) object given our splits
3. Train for some training regiment
4. Get predictions on the [`test`](https://fastcore.fast.ai/test#test) set, and potentially keep track of any statistics.

Let's do so below:

```python
val_pct, tst_preds = L(), L()
kf = KFold(n_splits=5, shuffle=False)
for train_idx, valid_idx in kf.split(train.index):
    splits = (L(list(train_idx)), L(list(valid_idx)))
    procs = [Categorify(classes), Normalize.from_tab(means, stds), FillMissing(fill_strategy=FillStrategy.median, fill_vals=fill_vals, na_dict=na_dict)]
    to = TabularPandas(train, procs, cat_names, cont_names, y_names='salary',
                       splits=splits)
    dls = to.dataloaders(bs=512)
    learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)
    learn.fit(3, 1e-2)
    test_dl = learn.dls.test_dl(test)
    with learn.no_bar():
        val_pct.append(learn.validate()[-1])
        tst_preds.append(learn.get_preds(dl=test_dl))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.379017</td>
      <td>0.380708</td>
      <td>0.826992</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.364980</td>
      <td>0.359392</td>
      <td>0.832281</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.355631</td>
      <td>0.361775</td>
      <td>0.825456</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.382340</td>
      <td>0.376627</td>
      <td>0.829039</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.362212</td>
      <td>0.366542</td>
      <td>0.832111</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.355434</td>
      <td>0.372222</td>
      <td>0.830063</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.385911</td>
      <td>0.374170</td>
      <td>0.843542</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.368800</td>
      <td>0.339751</td>
      <td>0.842348</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.360772</td>
      <td>0.349895</td>
      <td>0.843542</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.377877</td>
      <td>0.358854</td>
      <td>0.835523</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.362264</td>
      <td>0.362680</td>
      <td>0.833646</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.355874</td>
      <td>0.363413</td>
      <td>0.833134</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.380469</td>
      <td>0.358595</td>
      <td>0.838423</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.363201</td>
      <td>0.352324</td>
      <td>0.837912</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.356388</td>
      <td>0.350427</td>
      <td>0.837741</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


Now let's take a look at our results:

```python
for i, (pred, truth) in enumerate(tst_preds):
    print(f'Fold {i+1}: {accuracy(pred, truth)}')
```

    Fold 1: 0.8390663266181946
    Fold 2: 0.834152340888977
    Fold 3: 0.8320024609565735
    Fold 4: 0.8356879353523254
    Fold 5: 0.8329238295555115


Let's try ensembling them and seeing what happens:

```python
sum_preds = []
for i, (pred, truth) in enumerate(tst_preds):
    sum_preds.append(pred.numpy())
avg_preds = np.sum(sum_preds, axis=0) / 5
print(f'Average Accuracy: {accuracy(tensor(avg_preds), tst_preds[0][1])}')
```

    Average Accuracy: 0.8366093635559082


As we can see, ensembling all the models together boosted our score by .1%. Not the highest of increases though! Let's try out another CV method and see if it works better

## Stratified K-Fold

While the first example simply split our dataset either randomly (if we passed `True`) or just down the indicies, there are a multitude of cases where we won't have perfectly balanced classes (where the previous example would be useful). What can we do in such a situation?

Stratified K-Fold Validation allows us to split our data while also preserving the percentage of samples inside of each class. We'll follow the same methodology as we did before with a few minor changes to have it work with Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold
```

The only difference is along with our `train.index` we also need to pass in our `y`'s so it can gather the class distributions:

```python
val_pct, tst_preds = L(), L()
skf = StratifiedKFold(n_splits=5, shuffle=False)
for train_idx, valid_idx in kf.split(train.index, train['salary']): # right here
    splits = (L(list(train_idx)), L(list(valid_idx)))
    procs = [Categorify(classes), Normalize.from_tab(means, stds), FillMissing(fill_strategy=FillStrategy.median, fill_vals=fill_vals, na_dict=na_dict)]
    to = TabularPandas(train, procs, cat_names, cont_names, y_names='salary',
                       splits=splits)
    dls = to.dataloaders(bs=512)
    learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)
    learn.fit(3, 1e-2)
    test_dl = learn.dls.test_dl(test)
    with learn.no_bar():
        val_pct.append(learn.validate()[-1])
        tst_preds.append(learn.get_preds(dl=test_dl))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.377596</td>
      <td>0.366456</td>
      <td>0.831599</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.360850</td>
      <td>0.361772</td>
      <td>0.827674</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.356481</td>
      <td>0.359992</td>
      <td>0.831257</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.377417</td>
      <td>0.388749</td>
      <td>0.822726</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.360371</td>
      <td>0.376890</td>
      <td>0.824774</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.352614</td>
      <td>0.368503</td>
      <td>0.833817</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.387596</td>
      <td>0.358673</td>
      <td>0.842177</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.368236</td>
      <td>0.347018</td>
      <td>0.844907</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.362123</td>
      <td>0.345612</td>
      <td>0.841324</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.375481</td>
      <td>0.365665</td>
      <td>0.836205</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.358180</td>
      <td>0.362090</td>
      <td>0.832111</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.351982</td>
      <td>0.360600</td>
      <td>0.830404</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.385218</td>
      <td>0.363116</td>
      <td>0.831428</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.363915</td>
      <td>0.349798</td>
      <td>0.836717</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.356412</td>
      <td>0.354061</td>
      <td>0.837400</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


Let's see how our new version fairs up:

```python
for i, (pred, truth) in enumerate(tst_preds):
    print(f'Fold {i+1}: {accuracy(pred, truth)}')
```

    Fold 1: 0.8335380554199219
    Fold 2: 0.835073709487915
    Fold 3: 0.8316953182220459
    Fold 4: 0.8412162065505981
    Fold 5: 0.8387592434883118


We can see that so far it looks a bit better (we actually have one with 84%!). 

Now let's try the ensemble:

```python
sum_preds = []
for i, (pred, truth) in enumerate(tst_preds):
    sum_preds.append(pred.numpy())
avg_preds = np.sum(sum_preds, axis=0) / 5
print(f'Average Accuracy: {accuracy(tensor(avg_preds), tst_preds[0][1])}')
```

    Average Accuracy: 0.835995078086853


Not quite as well in the ensemble (down by 0.1%), however I would trust this version much *much* more than the regular `KFold`.

Why?

Stratification ensures that we maintain the original distribution of our `y` values, ensuring that if we have rare classes they will always show up and be trained on. Now let's look at a multi-label example.

## Multi-Label Stratified K-Fold

To run Multi-Label Stratified K-Fold, I will show an example below, but we will not run it (as there currently isn't quite a close enough dataset outside of Kaggle right now).

First we'll need to import our `MultilabelStratifiedKfold` from `iterstrat`:


```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
```

Then when following our above method (ensure you have your `loss_func`, etc properly setup), we simply replace our `for train_idx, valid_idx` with:

```python
mskf = MultilabelStratifiedKFold(n_splits=5)
for train_idx, val_idx in mskf.split(X=train, y=train[y_names]):
    "blah"
```
