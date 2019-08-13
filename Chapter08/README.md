To run this example, you will need to download the Kaggle Diabetic Retinopathy dataset from https://www.kaggle.com/c/diabetic-retinopathy-detection. In order to download this dataset, you will need to register with Kaggle. The Kaggle command-line API is needed for the download as well: https://github.com/Kaggle/kaggle-api. See directions there on how to set up the command-line API and get it working.

Once you've downloaded the file, you will need to set up the following directory structure

```bash
 DR/
   train/
   trainLabels.csv
```

You can do this with a couple mkdir commands:

```bash
mkdir DR
mkdir DR/train
mv (location of downloaded trainLabels.csv) DR/
```

Finally, you will need to set the DEEPCHEM_DATA_DIR environment variable to your current directory

```bash
export DEEPCHEM_DATA_DIR=`pwd`
```

If you'd like to get a pre-trained model, you run the following commands:

```bash
wget https://s3-us-west-1.amazonaws.com/deepchem.io/featurized_datasets/DR_model.tar.gz
mv DR_model.tar.gz test_model/
cd test_model
tar -zxvf DR_model.tar.gz
cd ..
```

Now, you call `run.py` to run the model

```bash
python run.py
```
