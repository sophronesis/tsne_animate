# tsne_animate
Automated animation for scikit-learn's t-sne algorithm



Requirements:
```
ffmpeg or mencoder  to save gifs
```


Easy to install:  
```
$ pip install tsne_animate
```

Easy to use:
```python
from sklearn import datasets, manifold
from tsne_animate import tsneAnimate
digits = datasets.load_digits()
tsne = tsneAnimate(manifold.TSNE(learning_rate=1000))
tsne.animate(digits.data,digits.target)
```

![digits](https://github.com/hardkun/tsne_animate/blob/master/examples/digits.gif)

```python
from sklearn import datasets, manifold
from tsne_animate import tsneAnimate
iris = datasets.load_iris()
tsne = tsneAnimate(manifold.TSNE(learning_rate=50))
tsne.animate(iris.data,iris.target,0,'iris.gif')
```

![iris](https://github.com/hardkun/tsne_animate/blob/master/examples/iris.gif)
