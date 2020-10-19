# Tabular Binary Classification (Beginner)
> Introducing the Tabular API with an example problem





---
This article is also a Jupyter Notebook available to be run from the top down. There will be code snippets that you can then run in any environment. Below are the versions of `fastai` and `fastcore` currently running at the time of writing this:
* `fastai`: 2.0.14 
* `fastcore`: 1.0.11 
---



## Binary Classification

In this example we will be walking through the `fastai` tabular API to perform binary classification on the Salary dataset.

This notebook can run along side the first tabular lesson from Walk with fastai2, shown [here](https://www.youtube.com/watch?v=liTHAhdl1cQ&list=PLFDkaGxp5BXDvj3oHoKDgEcH73Aze-eET&index=9&t=430s)

First we need to call the tabular module:

```
from fastai.tabular.all import *
```

And grab our dataset:

```
path = untar_data(URLs.ADULT_SAMPLE)
```

If we look at the contents of our folder, we will find our data lives in `adult.csv`:

```
path.ls()
```




    (#3) [Path('/home/ml1/.fastai/data/adult_sample/models'),Path('/home/ml1/.fastai/data/adult_sample/export.pkl'),Path('/home/ml1/.fastai/data/adult_sample/adult.csv')]



We'll go ahead and open it in `Pandas` and take a look:

```
df = pd.read_csv(path/'adult.csv')
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



## TabularPandas

`fastai` has a new way of dealing with tabular data by utilizing a `TabularPandas` object. It expects some dataframe, some `procs`, `cat_names`, `cont_names`, `y_names`, `y_block`, and some `splits`. We'll walk through all of them

First we need to grab our categorical and continuous variables, along with how we want to process our data.

```
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
```

When we pre-process tabular data with `fastai`, we do one or more of three transforms:

* `Categorify`
* `FillMissing`
* `Normalize`

### Categorify

`Categorify` will transform columns that are in your `cat_names` into that type, along with label encoding our categorical data. 


First we'll make an instance of it:

```
cat = Categorify()
```

And now let's try transforming a dataframe

```
to = TabularPandas(df, cat, cat_names)
```

We can then extract that transform from `to.procs.categorify`:

```
cats = to.procs.categorify
```

Let's take a look at the categories:

```
cats['relationship']
```




    (#7) ['#na#',' Husband',' Not-in-family',' Other-relative',' Own-child',' Unmarried',' Wife']



We can see that it added a `#na# `category. Let's look at the actual column:

```
to.show(max_n=3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Private</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Wife</td>
      <td>White</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>#na#</td>
      <td>Unmarried</td>
      <td>Black</td>
    </tr>
  </tbody>
</table>


We can see now, for example, that `occupation` got returned a `#na# `value (as it was missing)

If we call `to.cats` we can see our one-hot encoded variables:

```
to.cats.head()
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
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>13</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>15</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>6</td>
      <td>3</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Normalize

To properly work with our numerical columns, we need to show a relationship between them all that our model can understand. This is commonly done through Normalization, where we scale the data between -1 and 1, and compute a `z-score`

```
norm = Normalize()
```

Let's make another `to`

```
to = TabularPandas(df, norm, cont_names=cont_names)
```

```
norms = to.procs.normalize
```

And take a closer look.

We can grab the means and standard deviations like so:

```
norms.means
```




    {'age': 38.58164675532078,
     'fnlwgt': 189778.36651208502,
     'education-num': 10.079815426825466}



```
norms.stds
```




    {'age': 13.64022319230403,
     'fnlwgt': 105548.3568809906,
     'education-num': 2.5729591440613078}



And we can also call `to.conts` to take a look at our transformed data:

```
to.conts.head()
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
      <th>fnlwgt</th>
      <th>education-num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.763796</td>
      <td>-0.838084</td>
      <td>0.746294</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.397233</td>
      <td>0.444987</td>
      <td>1.523609</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.042642</td>
      <td>-0.886734</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.042642</td>
      <td>-0.728873</td>
      <td>1.912267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.250608</td>
      <td>-1.018314</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### FillMissing

Now the last thing we need to do is take care of any missing values in our **continuous** variables (we have a special `#na#` for categorical data already). We have three strategies we can use:
* `median`
* `mode`
* `constant`

By default it uses `median`:

```
fm = FillMissing(fill_strategy=FillStrategy.median)
```

We'll recreate another `TabularPandas`:

```
to = TabularPandas(df, fm, cont_names=cont_names)
```

Let's look at those missing values in the first few rows:

```
to.conts.head()
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
      <th>fnlwgt</th>
      <th>education-num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>101320</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>236746</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>96185</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>112847</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>82297</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>



**But wait!** There's more!

```
to.cat_names
```




    (#1) ['education-num_na']



We have categorical values?! Yes!

```
to.cats.head()
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
      <th>education-num_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




We now have an additional boolean value based on if the value was missing or not too!

## The DataLoaders

Now let's build our `TabularPandas` for classifying. We're also going to want to split our data and declare our `y_names` too:

```
splits = RandomSplitter()(range_of(df))
splits
```




    ((#26049) [18724,19703,4062,9102,28824,4054,5833,16188,2731,28161...],
     (#6512) [24465,976,1726,10178,4740,3920,32288,26018,20274,9660...])



What is `range_of`?

```
range_of(df)[:5], len(df)
```




    ([0, 1, 2, 3, 4], 32561)



It's a list of total index's in our `DataFrame`

We'll use all our `cat` and `cont` names, the `procs`, declare a `y_name`, and finally specify a single-label classification problem with `CategoryBlock`

```
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
y_names = 'salary'
y_block = CategoryBlock()
```

Now that we have everything declared, let's build our `TabularPandas`:

```
to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names,
                   y_names=y_names, y_block=y_block, splits=splits)
```

And now we can build the `DataLoaders`. We can do this one of two ways, first just calling `to.dataloaders()` on our data:

```
dls = to.dataloaders()
```

Or we can create the `DataLoaders` ourselves (a train and valid). One great reason to do this this way is we can pass in different batch sizes into each `TabDataLoader`, along with changing options like `shuffle` and `drop_last`

So how do we use it? Our train and validation data live in to.train and to.valid right now, so we specify that along with our options. When you make a training DataLoader, you want `shuffle` to be `True` and `drop_last` to be `True` (so we drop the last incomplete batch)


```
trn_dl = TabDataLoader(to.train, bs=64, shuffle=True, drop_last=True)
val_dl = TabDataLoader(to.valid, bs=128)
```

Now we can make some `DataLoaders`:

```
dls = DataLoaders(trn_dl, val_dl)
```

And show a batch of data:

```
dls.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Private</td>
      <td>11th</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>17.0</td>
      <td>200199.000263</td>
      <td>7.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>36.0</td>
      <td>256635.997971</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Transport-moving</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>44.0</td>
      <td>172032.000108</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Self-emp-not-inc</td>
      <td>5th-6th</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>44.0</td>
      <td>112506.998184</td>
      <td>3.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Local-gov</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>40.0</td>
      <td>74949.002528</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>?</td>
      <td>10th</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>17.0</td>
      <td>138506.999270</td>
      <td>6.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Private</td>
      <td>Assoc-voc</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>White</td>
      <td>False</td>
      <td>28.0</td>
      <td>247819.002902</td>
      <td>11.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>37.0</td>
      <td>112496.998941</td>
      <td>14.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Separated</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>False</td>
      <td>41.0</td>
      <td>215479.000444</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Private</td>
      <td>Assoc-voc</td>
      <td>Divorced</td>
      <td>Sales</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>31.0</td>
      <td>163302.999641</td>
      <td>11.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>


> Why can we do the `.dataloaders()`? Because `TabularPandas` itself is actually a set of `TabDataLoaders`! See below for a comparison test:

```
to._dbunch_type == dls._dbunch_type
```




    True



## Tabular Learner and Training a Model

Now we can build our `Learner`! But what's special about a tabular neural network?

### Categorical Variables

When dealing with our categorical data, we create what is called an **embedding matrix**. This allows for a higher dimentionality for relationships between the different categorical cardinalities. Finding the best size ratio was done through experiments by Jeremy on the Rossmann dataset

This "rule of thumb" is to use either a maximum embedding space of 600, or 1.6 times the cardinality raised to the 0.56, or written out as:

{% raw %}
$$min(600, (1.6 * {var.nunique)}^{0.56})$$
{% endraw %}

Let's calculate these embedding sizes for our model to take a look-see:

```
emb_szs = get_emb_sz(to); emb_szs
```




    [(10, 6), (17, 8), (8, 5), (16, 8), (7, 5), (6, 4), (3, 3)]




If we want to see what each one aligns to, let's look at the order of `cat_names`

```
to.cat_names
```




    (#7) ['workclass','education','marital-status','occupation','relationship','race','education-num_na']



Let's specifically look at `workclass`:

```
to['workclass'].nunique()
```




    9



If you notice, we had `10` there, this is to take one more column for any missing categorical values that may show

### Numerical Variables

Numericals we can simply pass in how many there are to the model:

```
cont_len = len(to.cont_names); cont_len
```




    3



And now we have all the pieces we need to build a `TabularModel`!

### TabularModel

What makes this model a little different is our batches is actually two inputs:

```
batch = dls.one_batch(); len(batch)
```




    3



```

batch[0][0], batch[1][0]
```




    (tensor([ 5, 12,  5,  5,  2,  5,  1]), tensor([-0.1858, -0.4134, -0.4253]))




With the first being our categorical variables and the second being our numericals.

Now let's make our model. We'll want our size of our embeddings, the number of continuous variables, the number of outputs, and how large and how many fully connected layers we want to use:

```
net = TabularModel(emb_szs, cont_len, 2, [200,100])
```

Let's see it's architecture:

```
net
```




    TabularModel(
      (embeds): ModuleList(
        (0): Embedding(10, 6)
        (1): Embedding(17, 8)
        (2): Embedding(8, 5)
        (3): Embedding(16, 8)
        (4): Embedding(7, 5)
        (5): Embedding(6, 4)
        (6): Embedding(3, 3)
      )
      (emb_drop): Dropout(p=0.0, inplace=False)
      (bn_cont): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (layers): Sequential(
        (0): LinBnDrop(
          (0): BatchNorm1d(42, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): Linear(in_features=42, out_features=200, bias=False)
          (2): ReLU(inplace=True)
        )
        (1): LinBnDrop(
          (0): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): Linear(in_features=200, out_features=100, bias=False)
          (2): ReLU(inplace=True)
        )
        (2): LinBnDrop(
          (0): Linear(in_features=100, out_features=2, bias=True)
        )
      )
    )



### tabular_learner


Now that we know the background, let's build our model a little bit faster and generate a `Learner` too:

```
learn = tabular_learner(dls, [200,100], metrics=accuracy)
```

And now we can fit!

```
learn.lr_find()
```








    SuggestedLRs(lr_min=0.0013182567432522773, lr_steep=1.3182567358016968)




![png](03_tab.clas.binary_files/output_88_2.png)


```
learn.fit(3, 1e-2)
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
      <td>0.373488</td>
      <td>0.362226</td>
      <td>0.841523</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.367182</td>
      <td>0.354484</td>
      <td>0.838759</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.368737</td>
      <td>0.355205</td>
      <td>0.834306</td>
      <td>00:03</td>
    </tr>
  </tbody>
</table>



Can we speed this up a little? Yes we can! The more you can load into a batch, the faster you can process the data. This is a careful balance, for tabular data I go to a maximum of 4096 rows per batch if the dataset is large enough for a decent number of batches:

```
dls = to.dataloaders(bs=1024)
learn = tabular_learner(dls, [200,100], metrics=accuracy)
learn.fit(3, 1e-2)
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
      <td>0.389950</td>
      <td>0.407716</td>
      <td>0.812500</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.366711</td>
      <td>0.350281</td>
      <td>0.839681</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.357688</td>
      <td>0.355638</td>
      <td>0.836763</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



We can see we fit very quickly, but it didn't fit quite as well (there is a trade-off):

```
dls = to.dataloaders(bs=4096)
learn = tabular_learner(dls, [200,100], metrics=accuracy)
learn.fit(3, 1e-2)
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
      <td>0.449081</td>
      <td>0.503046</td>
      <td>0.772267</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.407749</td>
      <td>0.465814</td>
      <td>0.758446</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.387042</td>
      <td>0.445514</td>
      <td>0.776720</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


## Inference

Now let's look at how we can perform inference. To do predictions we can use `fastai`'s in-house `learn.predict` for individual rows, and `get_preds` + `test_dl` for batches of predictions:

```
row, cls, probs = learn.predict(df.iloc[0])
```





```
row.show()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Private</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Wife</td>
      <td>White</td>
      <td>False</td>
      <td>49.0</td>
      <td>101320.00007</td>
      <td>12.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>


Now let's try `test_dl`. There's something special we can do here too:


```
dl = learn.dls.test_dl(df.iloc[:100])
```

Let's look at a batch:

```
dl.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Private</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Wife</td>
      <td>White</td>
      <td>False</td>
      <td>49.0</td>
      <td>101320.000070</td>
      <td>12.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>44.0</td>
      <td>236745.999937</td>
      <td>14.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>#na#</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>True</td>
      <td>38.0</td>
      <td>96185.001851</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Self-emp-inc</td>
      <td>Prof-school</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>False</td>
      <td>38.0</td>
      <td>112846.997614</td>
      <td>15.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Self-emp-not-inc</td>
      <td>7th-8th</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Wife</td>
      <td>Black</td>
      <td>True</td>
      <td>42.0</td>
      <td>82296.994924</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Handlers-cleaners</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>20.0</td>
      <td>63210.003417</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Divorced</td>
      <td>#na#</td>
      <td>Other-relative</td>
      <td>White</td>
      <td>False</td>
      <td>49.0</td>
      <td>44434.000334</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Private</td>
      <td>11th</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>37.0</td>
      <td>138939.999209</td>
      <td>7.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>46.0</td>
      <td>328215.996478</td>
      <td>9.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Self-emp-inc</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Husband</td>
      <td>White</td>
      <td>True</td>
      <td>36.0</td>
      <td>216711.000298</td>
      <td>10.0</td>
      <td>&gt;=50k</td>
    </tr>
  </tbody>
</table>


We have our labels! It'll grab them if possible by default! 

What does that mean? Well, besides simply calling `get_preds`, we can also run `validate` to see how a model performs. This is nice as it can allow for efficient methods when calculating something like permutation importance:

```
learn.validate(dl=dl)
```








    (#2) [0.4870152175426483,0.7699999809265137]



We'll also show an example of `get_preds`:

```
preds = learn.get_preds(dl=dl)
```





```
preds[0][0]
```




    tensor([0.5952, 0.4048])



What would happen if I accidently passed in an unlablled dataset to `learn.validate` though? Let's find out:

```
df2 = df.iloc[:100].drop('salary', axis=1)
df2.head()
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
    </tr>
  </tbody>
</table>
</div>



```
dl = learn.dls.test_dl(df2)
learn.validate(dl=dl)
```








    (#2) [None,None]



We can see it will simply just return `None`!
