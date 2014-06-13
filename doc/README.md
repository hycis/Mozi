
# Procedures for Reconstructing Spec Files with AutoEncoder #

In order to use this package, user should install Anaconda(a super package that includes 
numpy, matplotlib and others), Theano and sqlite3.

Steps from data preparation to training model to generating specs from model

__1. Generate datafile from spec files__

In order to feed the data into the training framework, 
the first step is to merge all the spec files into a numpy data 
file that is readable by AutoEncoder by using the script
[specs2data.py](../scripts/specs2data.py)

In order to know all the options available for the script, use 

```bash
$ python specs2data.py -h
```

For example, in order to merge p276 spec files into one npy file (splits = 1), issue

```bash
$ python specs2data.py --spec_files /path/to/p276/*.spec --splits 1 --input_spec_dtype f4 
--feature_size 2049 --output_dir /path/to/output_dir/
```

__2. Setting Environment Variables__

In smartNN, there are three environment variables.

```python
smartNN_DATA_PATH   # the directory for all the datasets
smartNN_SAVE_PATH   # the directory to save the best models, the outputs logs and the hyperparams 
smartNN_DATABASE_PATH # the directory to save the database which contains the stats from 
                      # all the experiments which is used for picking the best model
``` 




