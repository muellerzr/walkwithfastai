# Contribution Template
> A brief sentence describing the goal of the article


Notebooks should be named by their sections as well as a topic. Such as how this one is `01_intro` (for introduction section) `.contribute` (as it's the contribution guide). Another example would be `02_vision.classification.single_label`

In the very first part of your notebook you should denote the `fastai` and `fastcore` versions, as well as any other library used. I have a cookie-cutter cell below. Simply call [`state_versions`](/utils.html#state_versions) with a list of your libraries being used and it will display them in `Markdown`. 

Ensure you have a `#hide_input` comment above the function call so the documentation looks clean.




---
This article is also a Jupyter Notebook available to be run from the top down. There will be code snippets that you can then run in any environment. Below are the versions of `fastai`  `fastcore`  currently running at the time of writing this:
* `fastai`: 2.0.14 
* `fastcore`: 1.0.11 
---



From here you should denote sections with two ##, walking users through your problem or example you are showing:

## My Problem

My problem is `x`, below we will try to solve `x` using `y`. (No need for verbatum, be as creative as you would like to be!)

Ideally if you are walking through an implementation, at the end show an example of how to perform both batch and single input predictions. (see `vision.classification.single_label` for an example)

## Submitting a PR

Once you've written your notebook, simply submit your PR to [the repo](https://github.com/walkwithfastai/walkwithfastai.github.io) and I will review it!
