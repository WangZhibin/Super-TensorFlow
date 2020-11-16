# Super-TensorFlow

This is the my own TensorFlow Edition (1.15.2) that is optimized and built from source.

## Install on macOS

1. Intell Python Environment

```
$ conda create --name tensorflow python=3.7 numpy scipy
$ conda activate tensorflow
```

2. Install dependencies

```
python -m pip install --upgrade pip --user
pip install --upgrade cython --user
pip install --upgrade textblob --user
pip install --upgrade nltk --user
pip install --upgrade jieba --user
pip install --upgrade visdom --user
pip install --upgrade keras --user
pip install --upgrade pillow --user
pip install --upgrade opencv-python --user
pip install --upgrade opencv-contrib-python --user
pip install --upgrade theano --user
pip install --upgrade scipy --user
pip install --upgrade pandas --user

pip install --upgrade xlrd --user
pip install --upgrade statsmodels --user
pip install --upgrade seaborn --user
pip install --upgrade plotly --user
pip install --upgrade bokeh --user
pip install --upgrade pydot --user
pip install --upgrade scikit-learn --user
pip install --upgrade scikit-image --user
pip install --upgrade sklearn --user
pip install --upgrade pyedflib --user
pip install --upgrade bs4 --user
pip install --upgrade ignite --user
pip install --upgrade cython --user
pip install --upgrade keras --user

pip install --upgrade opencv-python --user
pip install --upgrade opencv-contrib-python --user
pip install --upgrade theano --user
pip install --upgrade numpy --user
pip install --upgrade scipy --user
pip install --upgrade pandas --user
pip install --upgrade xlrd --user
pip install --upgrade statsmodels --user
pip install --upgrade matplotlib --user
pip install --upgrade seaborn --user
pip install --upgrade plotly --user
pip install --upgrade bokeh --user
pip install --upgrade pydot --user

pip install --upgrade scikit-learn --user
pip install --upgrade scikit-image --user
pip install --upgrade sklearn --user
pip install --upgrade XGBoost --user
pip install --upgrade LightGBM --user
pip install --upgrade CatBoost --user
pip install --upgrade Eli5 --user
pip install --upgrade NLTK --user
pip install --upgrade Gensim --user
pip install --upgrade jupyter --user
pip install --upgrade notebook --user
pip install --upgrade pyedflib --user
pip install --upgrade bs4 --user
pip install --upgrade pandas-profiling --user
pip install --upgrade Django --user
pip install --upgrade gym --user

pip uninstall protobuf
pip uninstall google
pip install google
pip install protobuf
pip install google-cloud
pip install keras_preprocessing

```

3. Install Bazel

https://github.com/bazelbuild/bazel/releases/tag/0.26.1

bazel-0.26.1-installer-darwin-x86_64.sh

```
$ chmod +x bazel-0.26.1-installer-darwin-x86_64.sh
$ ./bazel-0.26.1-installer-darwin-x86_64.sh --user
```

4. Configurations

```
./configure
```

Input the paths of Python and choose No for other installation choices

5. Build

```
$ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```

6. Build pip package

```
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

7. Install pip package

```
$ pip install /tmp/tensorflow_pkg/tensorflow-1.15.2-cp37-cp37m-macosx_10_15_x86_64.whl
```

Have fun!
