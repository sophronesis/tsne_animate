# tsne_animate
Automated animation for scikit-learn's t-sne algorithm

Easy to use:  
```
$pip install tsne_animate
```

```
from sklearn import datasets, manifold
from tsne_animate import tsneAnimate
digits = datasets.load_digits()
tsne = tsneAnimate(manifold.TSNE(learning_rate=1000))
tsne.animate(digits.data,digits.target,'digits.gif')
```

![digits](https://github.com/hardkun/tsne_animate/blob/master/examples/digits.gif)

```
from sklearn import datasets, manifold
from tsne_animate import tsneAnimate
iris = datasets.load_iris()
tsne = tsneAnimate(manifold.TSNE(learning_rate=50))
tsne.animate(iris.data,iris.target,'iris.gif')
```

![iris](https://github.com/hardkun/tsne_animate/blob/master/examples/iris.gif)
