# Utility Functions
> Various utilities that function as the backbone for the website


## Declaring Version Information

Simple methods to grab and state libraries you are using for a tutorial


<h4 id="get_version" class="doc_header"><code>get_version</code><a href="__main__.py#L3" class="source_link" style="float:right">[source]</a></h4>

> <code>get_version</code>(**`lib`**:`str`)

Returns version of `lib`



<h4 id="state_versions" class="doc_header"><code>state_versions</code><a href="__main__.py#L3" class="source_link" style="float:right">[source]</a></h4>

> <code>state_versions</code>(**`libs`**:`list`=*`[]`*)

State all the versions currently installed from `libs` in Markdown


Example usage:

```python
state_versions(['fastai', 'fastcore'])
```





---
This article is also a Jupyter Notebook available to be run from the top down. There
will be code snippets that you can then run in any environment.

Below are the versions of `fastai` and `fastcore` currently running at the time of writing this:
* `fastai`: 2.0.14 
* `fastcore`: 1.0.11 
---


