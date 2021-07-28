Tuning your machine learning model's hyperparameters is a great way to tailor your model's performance. Your model keeps over-fitting to your training data? Tuning your hyperparameters can help! Whenever you tune a model's hyperparameters you need to select some objective to quantify desirable model performance. For some problems, one of which will be discussed in this post, you may want to examine how model performance changes across *several* objectives. We will see that this process allows us to make a more-informed choice of hyperparameters! 

## What are Hyperparameter and Why Should We Tune Them?

The specification of hyperparameters values (whether optimal or default) is essential to many of the common machine learning algorithms because they specify model topology and ultimately model performance. For example, the number of trees in a random forest or sizes of the hidden layers in a neural network would both be be specified by hyperparameters. If you are looking for more explanation of what hyperparameters are and how they differ from regular parameters, I suggest you read [this](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/#:~:text=In\%20summary\%2C\%20model\%20parameters\%20are,be\%20set\%20manually\%20and\%20tuned) article or even just browse through the [Wikipedia Page](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) on hyperparameters.

Now I want to be clear, tuning hyperparameters is generally a "finishing touches" step when developing a machine learning model. Shahul ES said it best in [this](https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020) article by stating that hyperparameter tuning is a great way to "extract the last juice out of your models". Processes like model or feature selection cannot be overlooked. Moral of the story: if your model is severely under-performing or just flat-out broken, don't look to hyperparameter tuning to fix all your problems.

## Why Even Consider Multiple Objectives?

For many problems a single objective effectively captures desirable model performance. For example, consider the classic [Iris Classification Problem](https://archive.ics.uci.edu/ml/datasets/iris) where a model classifies types or iris flowers. You are probably thinking that for this task you want a model that is as accurate as possible, and I would agree! For the Iris problem, I would definetly pick the set of hyperparameters that maximizes model accuracy. 
 
 But now think about other classic problem of breast cancer diagnosis based on the [Breast Cancer Wisconsin dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\%28Diagnostic\%29). In this case, *how* your model classifies patients has very different implications. This is best visualized with its confusion matrix:

![Figure 1](https://github.com/kravitsjacob/multiobjective-hyperparameter/blob/gh-pages/Figures/Cancer%20Confusion%20Matrix%20Figure.pdf?raw=true)

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/kravitsjacob/multiobjective-hyperparmater/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
