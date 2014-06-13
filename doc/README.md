
# Procedures for Reconstructing Spec Files #

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
--feature_size 2048 --output_dir /path/to/output_dir/
```

__2. Setup AutoEncoder for training__



