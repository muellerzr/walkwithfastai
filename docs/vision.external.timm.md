# Utilizing the `timm` Library Inside of `fastai`
> How to bring the power of Transfer Learning with new architectures






---
This article is also a Jupyter Notebook available to be run from the top down. There
will be code snippets that you can then run in any environment.

Below are the versions of `fastai`, `fastcore`, and `timm` currently running at the time of writing this:
* `fastai`: 2.0.14 
* `fastcore`: 1.0.11 
* `timm`: 0.2.1 
---



## Bringing in External Models into the Framework

As we are well aware, `fastai` models deep down are just `PyTorch` models. However as the field of Machine Learning keeps going, new and fresh architectures are introduced. Wouldn't it be nice if it were easy to integrate them into the `fastai` framework and play with them?

## Using Ross Wightman's `timm` Library

[Ross Wightman](https://twitter.com/wightmanr) has been on a mission to get pretrained weights for the newest Computer Vision models that come out of papers, and compare his results what the papers state themselves. The fantastic results live in his repository [here](https://github.com/rwightman/pytorch-image-models)

For users of the `fastai` library, it is a goldmine of models to play with! But how do we use it? Let's set up a basic `PETs` problem following the [tutorial](https://walkwithfastai.com/vision.clas.single_label):

```python
path = untar_data(URLs.PETS)
pat = r'/([^/]+)_\d+.*'
item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.))
batch_tfms = [*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)]
bs=16
pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(0.2),
                 get_y=RegexLabeller(pat = pat),
                 item_tfms=item_tfms,
                 batch_tfms=batch_tfms)
dls = pets.dataloaders(path/'images', bs=bs)
```

From here we would normally do something like `cnn_learner(dls, arch, metrics)`, however we need to do a few things special to work with Ross' framework.

`fastai` has a [`create_body`](https://docs.fast.ai/vision.learner#create_body) function, whcih is called during [`cnn_learner`](https://docs.fast.ai/vision.learner#cnn_learner), that will take a model architecuture and slice off the last Linear layer (resulting in a "body" that outputs unpooled features). This function looks like:

```python
def create_body(arch, n_in=3, pretrained=True, cut=None):
    "Cut off the body of a typically pretrained `arch` as determined by `cut`"
    model = arch(pretrained=pretrained)
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if   isinstance(cut, int):      return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else:                           raise NamedError("cut must be either integer or a function")
```

We're going to create our own that plays well
> Also:notebooks like this are exported as external modules inside of the `wwf` library! This one can be found in `vision.timm` to be used with your projects!


<h4 id="create_timm_body" class="doc_header"><code>create_timm_body</code><a href="wwf/vision/timm.py#L13" class="source_link" style="float:right">[source]</a></h4>

> <code>create_timm_body</code>(**`arch`**:`str`, **`pretrained`**=*`True`*, **`cut`**=*`None`*, **`n_in`**=*`3`*)

Creates a body from any model in the `timm` library.


```python
def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")
```

How do we use it? Let's try it out on an `efficientnet_b3` architecture (the entire list of supported architectures is found [here](https://github.com/rwightman/pytorch-image-models#models)

```python
body = create_timm_body('efficientnet_b3a', pretrained=True)
```

From here we can calculate the number input features our head needs to have with [`num_features_model`](https://docs.fast.ai/callback.hook#num_features_model).  We'll mutliply this by two since we have two pooling layers, [`AdaptiveConcatPool2d`](https://docs.fast.ai/layers#AdaptiveConcatPool2d) and `nn.AdaptiveAvgPool2d`

```python
nf = num_features_model(body)*2; nf
```




    3072



And now we can create a head!


```python
head = create_head(nf, dls.c)
```

To mix them together, we just wrap the two in a `nn.Sequential` and we now have a `PyTorch` model ready to be trained on:

```python
net = nn.Sequential(body, head)
```

From here we would pass it onto [`Learner`](https://docs.fast.ai/learner#Learner), specifying our `splitter` to be the [`default_split`](https://docs.fast.ai/vision.learner#default_split)
> `default_splitter` expects the body in `model[0]` and the head in `model[1]` to split our layer groups

```python
learn = Learner(dls, net, splitter=default_split)
```

To know this all worked properly, we should be able to call `learn.freeze()` and check the number of frozen parameters. (You can also call `learn.summary` but we are not since it has a lengthy output):

```python
learn.freeze()
unfrozen_params = filter(lambda p: not p.requires_grad, learn.model.parameters())
unfrozen_params = sum([np.prod(p.size()) for p in unfrozen_params])
model_parameters = filter(lambda p: p.requires_grad, learn.model.parameters())
frozen_params = sum([np.prod(p.size()) for p in model_parameters])
```

```python
unfrozen_params, frozen_params
```




    (1686272, 10608936)



Which we can see that only 1.6 million of the 10 million parameters are trainable, so our model is ready for transfer learning!

## Turning it all into a function

Let's make this a bit easier and create something like [`cnn_learner`](https://docs.fast.ai/vision.learner#cnn_learner), but for `timm`! We'll call it a [`timm_learner`](/vision.external.timm.html#timm_learner). First let's look at and compare what [`cnn_learner`](https://docs.fast.ai/vision.learner#cnn_learner) does internally:

```python
def cnn_learner(dls, arch, loss_func=None, pretrained=True, cut=None, splitter=None,
                y_range=None, config=None, n_out=None, normalize=True, **kwargs):
    "Build a convnet style learner from `dls` and `arch`"
    if config is None: config = {}
    meta = model_meta.get(arch, _default_meta)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if normalize: _add_norm(dls, meta, pretrained)
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = create_cnn_model(arch, n_out, ifnone(cut, meta['cut']), pretrained, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=ifnone(splitter, meta['split']), **kwargs)
    if pretrained: learn.freeze()
    return learn
```

At first it looks scary, but let's try and read it as best we can:
1. Grab potential private meta about an architecture we're using
2. Grab the number of expected outputs
3. Potentially normalize
4. Add a `y_range`
5. Create a `cnn_model` and [`Learner`](https://docs.fast.ai/learner#Learner)
6. Freeze our model

We're going to make a custom [`create_timm_model`](/vision.external.timm.html#create_timm_model) and [`timm_learner`](/vision.external.timm.html#timm_learner) function to do what we just did above. First, [`create_timm_model`](/vision.external.timm.html#create_timm_model) will model after [`create_cnn_model`](https://docs.fast.ai/vision.learner#create_cnn_model):


<h4 id="create_timm_model" class="doc_header"><code>create_timm_model</code><a href="wwf/vision/timm.py#L25" class="source_link" style="float:right">[source]</a></h4>

> <code>create_timm_model</code>(**`arch`**:`str`, **`n_out`**, **`cut`**=*`None`*, **`pretrained`**=*`True`*, **`n_in`**=*`3`*, **`init`**=*`kaiming_normal_`*, **`custom_head`**=*`None`*, **`concat_pool`**=*`True`*, **\*\*`kwargs`**)

Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library


```python
def create_timm_model(arch:str, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=True, **kwargs):
    "Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library"
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else: head = custom_head
    model = nn.Sequential(body, head)
    if init is not None: apply_init(model[1], init)
    return model
```

And now for our [`timm_learner`](/vision.external.timm.html#timm_learner):


<h4 id="timm_learner" class="doc_header"><code>timm_learner</code><a href="wwf/vision/timm.py#L41" class="source_link" style="float:right">[source]</a></h4>

> <code>timm_learner</code>(**`dls`**, **`arch`**:`str`, **`loss_func`**=*`None`*, **`pretrained`**=*`True`*, **`cut`**=*`None`*, **`splitter`**=*`None`*, **`y_range`**=*`None`*, **`config`**=*`None`*, **`n_out`**=*`None`*, **`normalize`**=*`True`*, **\*\*`kwargs`**)

Build a convnet style learner from `dls` and `arch` using the `timm` library


```python
def timm_learner(dls, arch:str, loss_func=None, pretrained=True, cut=None, splitter=None,
                y_range=None, config=None, n_out=None, normalize=True, **kwargs):
    "Build a convnet style learner from `dls` and `arch` using the `timm` library"
    if config is None: config = {}
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = create_timm_model(arch, n_out, default_split, pretrained, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=default_split, **kwargs)
    if pretrained: learn.freeze()
    return learn
```

Let's try it out by making the same model we did a moment ago:

```python
learn = timm_learner(dls, 'efficientnet_b3a')
```

And to verify let's look at those parameters one more time:

```python
unfrozen_params = filter(lambda p: not p.requires_grad, learn.model.parameters())
unfrozen_params = sum([np.prod(p.size()) for p in unfrozen_params])
model_parameters = filter(lambda p: p.requires_grad, learn.model.parameters())
frozen_params = sum([np.prod(p.size()) for p in model_parameters])
```

```python
unfrozen_params, frozen_params
```




    (1686272, 10608936)



They're exactly the same! So now we can utilize any architecture found inside of `timm` right away, and we built it in a structure very similar to how native `fastai` does it. 

To use this module in your own work, simply do:
```python
from wwf.vision.timm import *
learn = timm_learner(dls, 'efficientnet_b3a', metrics=[error_rate, accuracy])
```
{% include note.html content='`timm` needs to be installed beforehand' %}

## Model Lookup

To query various models to see what is available, you should directly use the `timm` library.


```python
import timm
```

### Listing all models available

One option is to list every model possible:

```python
timm.list_models()[:10]
```




    ['adv_inception_v3',
     'cspdarknet53',
     'cspdarknet53_iabn',
     'cspresnet50',
     'cspresnet50d',
     'cspresnet50w',
     'cspresnext50',
     'cspresnext50_iabn',
     'darknet53',
     'densenet121']



### Searching for models

You can also query the names of what is available as well, denoted as below:

```python
timm.list_models('*efficientnet*')[:10]
```




    ['efficientnet_b0',
     'efficientnet_b1',
     'efficientnet_b1_pruned',
     'efficientnet_b2',
     'efficientnet_b2_pruned',
     'efficientnet_b2a',
     'efficientnet_b3',
     'efficientnet_b3_pruned',
     'efficientnet_b3a',
     'efficientnet_b4']



```python
timm.list_models('*b3a')[:10]
```




    ['efficientnet_b3a']



```python
timm.list_models('resne*t*', pretrained=True)[:10]
```




    ['resnest14d',
     'resnest26d',
     'resnest50d',
     'resnest50d_1s4x24d',
     'resnest50d_4s2x40d',
     'resnest101e',
     'resnest200e',
     'resnest269e',
     'resnet18',
     'resnet26']



## Some Warnings

* Watch for anything with a `tf_` prefix. This means the original weights were ported from Google, so it uses manual padding to match TensorFlow's "same" padding, which adds GPU overhead and a general slowdown. If possible try to use the non-TF versions of models

* HRNet is a bit of a problem-child, so it is the only one not straight-forward to use
