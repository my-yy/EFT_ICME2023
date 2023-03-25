Code for *EFT: Expert Fusion Transformer for Voice-Face Representation Learning,ICME,2023*

## Requirements

```
pytorch==1.8.1
wandb==0.12.10
```



## Dataset

Download `dataset.zip` from [GoogleDrive](https://drive.google.com/drive/folders/1vher2RbPzh388p2_ZWgSxw3Dr8QEFgmz?usp=sharing) (4GB) and unzip it to the project root. 
The folder structure is shown below:

```
dataset/
├── evals
│   ├── test_matching_1N.pkl
│   ├── test_matching_g.pkl
│   ├── test_matching.pkl
│   ├── test_retrieval.pkl
│   ├── test_verification_gn.pkl
│   ├── test_verification_g.pkl
│   ├── test_verification_n.pkl
│   ├── test_verification.pkl
│   └── valid_verification.pkl
├── features
└── info
    ├── name2movies.pkl
    ├── name2tracks.pkl
    └── train_valid_test_names.pkl
```
The `dataset/features` folder contains vast number of small files. 
We suggest placing the project on an SSD disk to prevent IO bottlenecks.

# Train

``python main.py``  for the default setting in our paper

``python main.py --face_features=f_2plus1D_512 ``  for only use the R2+1D expert

``python main.py --bc_mode=sbc_3.0 ``  for change the SBC std threshold to 3.0


---

*use [wandb](https://wandb.ai) to view the training process:*

1. Create  `.wb_config.json`  file in the project root, using the following content:

   ```
   {
     "WB_KEY": "Your wandb auth key"
   }
   ```

   

2. add `--dryrun=False` to the training command, for example:   `python main.py --dryrun=False`



## Model Checkpoint

You can get the final model checkpoint in `eft_checkpoint.zip`.




