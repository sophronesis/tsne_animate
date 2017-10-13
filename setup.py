from setuptools import setup

setup(name='tsne_animate',
      version='0.1.4',
      description='Automated animation for scikit-learn\'s t-sne algorithm',
      url='https://github.com/hardkun/tsne_animate',
      author='Oleksandr Buzynnyi',
      author_email='oleksandr.buzynnyi@gmail.com',
      license='MIT',
      packages=['tsne_animate'],
      install_requires=['numpy',"scipy",'scikit-learn','matplotlib'],
      zip_safe=False)