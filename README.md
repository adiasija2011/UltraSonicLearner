# UltraSonicLearner

## TransUnet (Breast Ultrasound Segmentation)
Implements TransUnet for Breast Ultrasound Segmentation

|     Network     |                        Original code                         |                          Reference                           |
| :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    TransUnet    |      [Pytorch](https://github.com/Beckschen/TransUNet)       |       [Arxiv'21](https://arxiv.org/pdf/2102.04306.pdf)       |

## Datasets

Download the required dataset from [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) and [OASBUD](https://zenodo.org/record/545928#.Y_TIs4DP20n) and put it under ```data/```. For instance, the combined dataset should look like this:

```
├── root
    ├── data
        ├── combined
            ├── images
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── benign (10).png
                |   ├── malignant (17).png
                |   ├── ...
            ├── combined_train.txt
            ├── combined_val.txt
            ├── combined_test.txt
            ├── busi_test.txt
            └── oasbud_test.txt
    ├── src
    ├── main.py
    ├── split.py
```

## Environments

- Pytorch: 1.13.0 cuda 11.7
- cudatoolkit: 11.7.1
- scikit-learn: 1.0.2

## Training

For training, run the following command:

```python
python main.py --model TransUnet --base_dir ./data/combined --train_file_dir combined_train.txt --val_file_dir combined_val.txt --base_lr 0.01 --epoch 300 --batch_size 8
```

## Testing

```python
python test.py --model TransUnet --base_dir ./data/busi  --batch_size 1 --test_file_dir busi_test.txt --ckpt ./checkpoint/[checkpoint]
```
Download checkpoints from [here](https://drive.google.com/drive/folders/1hh24cJbeBt5yol8UwkjxyI1iyAfUVrcM?usp=sharing).
Predictions will be stored in ```results/```.
## Bar Plots

```python
python bar_plot.py
```

Metrics need to be manually updated in ```bar_plot.py```. Bar plot will be saved as ```All.png```. ```bar_plot.py``` also opens an interactive bar plot.

## Acknowledgements:

This code borrows from [Medical-Image_Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks). This code-base uses helper functions from [CMU-Net](https://github.com/FengheTan9/CMU-Net) and [Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation).

## Other QS:

If you have any questions or suggestions about this project, please contact us through email: sudarshanrajagopalan2002@gmail.com, adiasija@gmail.com, dhananjayagrad@gmail.com.

