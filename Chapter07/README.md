To run this example, you will need to download the Broad BBBC005 dataset from https://data.broadinstitute.org/bbbc/BBBC005/.  
No login or registration is needed to download this dataset, so the raw images can simply be fetched with

```bash
wget https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip
unzip BBBC005_v1_images.zip
```

The ground-truth segmentation masks can be fetched as follows

```bash
wget https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip
unzip BBBC005_v1_ground_truth.zip
```

To obtain the pretrained model you do

```bash
mkdir models
cd models
wget https://s3-us-west-1.amazonaws.com/deepchem.io/featurized_datasets/microscopy_models.zip
unzip microscopy_models.zip
```
