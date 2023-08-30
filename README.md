# MixupE
Codes for "MixupE: Understanding and Improving Mixup from Directional Derivative Perspective" UAI 2023 Best Student Paper

## Dataset

For CIFAR-10/100 and SVHN, these datasets can be downloaded automatically with torchvision.

Download [Tiny-Imgane-Net](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4)

## Environments
Environment details used for our study.

Python: 3.8.6
PyTorch: 1.11.0+cu10.2
Torchvision: 0.12.0+cu10.2
CUDA: 10.2
NumPy: 1.22.3

## Train with MixupE

```
python main.py --dataset <Dataset_Name> --data_dir <Dataset_Path> --root_dir <Exp_Path> --train mixupe --mixupe_version 3
```

Take the CIFAR-100 for example,
```
python main.py --dataset cifar100 --data_dir ~/data/CIFAR100/ \
    --root_dir ./experiments/ --labels_per_class 5000 \
    --arch preactresnet50  --epochs 800 --schedule 10 300 600 --gammas 0.01 0.1 0.1 \
    --train mixupe --adv_gen none --adv_train none  --mixup_alpha 2.0 \
    --decay 0.0001 --mixupe_version 3 --mixup_eta 0.001 --learning_rate 0.1 --threshold 1.0 
```
where the hyper-parameters in "--gammas" are "initial learning rate", "learning rate decay factors".

## Citation
```
@inproceedings{zou2023mixupe,
  title={Mixupe: Understanding and improving mixup from directional derivative perspective},
  author={Zou, Yingtian and Verma, Vikas and Mittal, Sarthak and Tang, Wai Hoh and Pham, Hieu and Kannala, Juho and Bengio, Yoshua and Solin, Arno and Kawaguchi, Kenji},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={2597--2607},
  year={2023},
  organization={PMLR}
}
```
