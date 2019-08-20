### tutorials
1. https://www.tensorflow.org/tutorials/layers
https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd
2. https://docs.google.com/presentation/d/1TVixw6ItiZ8igjp6U17tcgoFrLSaHWQmMOwjlgQY9co/pub?slide=id.g110257a6da_0_598


#### Stacked denoising auto encoder:
1. https://github.com/xiaohu2015/DeepLearning_tutorials
2. https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/models/sda.py


### kernel
https://ipython.readthedocs.io/en/stable/install/kernel_install.html
https://nteract.io/kernels

show kernels
```
jupyter kernelspec list --json
```

remove kernels:
```
jupyter kernelspec uninstall unwanted-kernel
```


In a certain python env, run 
```
pyenv local anaconda3-5.2.0
python -m ipykernel install --user --name anaconda3-5.2.0
```

or create a virtual env
```
pyenv virtualenv anaconda3-5.2.0 tf2.0-py3.6.5
pyenv local tf2.0-py3.6.5
python -m ipykernel install --user --name tf2.0-py3.6.5
```

if there is not ipykernel
```
pip install ipykernel
```

in `Hydrogen`, choose `Hydrogen: Start Local Kernel`
