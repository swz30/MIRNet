# Learning Enriched Features for Real Image Restoration and Enhancement (ECCV 2020)

[Syed Waqas Zamir](https://scholar.google.ae/citations?hl=en&user=POoai-QAAAAJ), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en)


[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2003.06792)
[![supplement](https://img.shields.io/badge/Supplementary-Material-B85252)](https://drive.google.com/file/d/1QIKp7h7Rd85odaS6bDoeDGXb0VLKo8I9/view?usp=sharing)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://www.youtube.com/watch?v=6xSzRjAodv4)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1hnhqSrjqQQiYn7XPAGpFgMBTfBlb1QAy/view?usp=sharing)

<hr />

### News
- A lightweight, fast and extended version of MIRNet is accepted in **TPAMI**. [Paper](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/) | [Code](https://github.com/swz30/MIRNetv2)

- Keras Tutorial on MIRNet is available at https://keras.io/examples/vision/mirnet/ 

- Video on Tensorflow Youtube channel https://youtu.be/BMza5yrwZ9s

- Links to (unofficial) implementations are added [here](#other-implementations)

<hr />

> **Abstract:** *With the goal of recovering high-quality image content from its degraded version, image restoration enjoys numerous applications, such as in surveillance, computational photography, medical imaging, and remote sensing.  Recently, convolutional neural networks (CNNs) have achieved dramatic improvements over conventional approaches for image restoration task. Existing CNN-based methods typically operate either on full-resolution or on progressively low-resolution representations. In the former case, spatially precise but contextually less robust results are achieved, while in the latter case, semantically reliable but spatially less accurate outputs are generated. In this paper, we present a novel architecture with the collective goals of maintaining spatially-precise high-resolution representations through the entire network, and receiving strong contextual information from the low-resolution representations.  The core of our approach is a multi-scale residual block containing several key elements: (a) parallel multi-resolution convolution streams for extracting multi-scale features, (b) information exchange across the multi-resolution streams, (c) spatial and channel attention mechanisms for capturing contextual information, and (d) attention based multi-scale feature aggregation. In the nutshell, our approach learns an enriched set of features that combines contextual information from multiple scales, while simultaneously preserving the high-resolution spatial details. Extensive experiments on five real image benchmark datasets demonstrate that our method, named as MIRNet, achieves state-of-the-art results for a variety of image processing tasks, including image denoising, super-resolution and image enhancement.* 

<details>
  <summary> <strong>Network Architecture</strong> (click to expand) </summary>
 
<p align="center">
  <img src = "https://i.imgur.com/vmywppl.png" width="700">
  <br/>
  <b> Overall Framework of MIRNet </b>
</p>

<table>
  <tr>
    <td> <img src = "https://i.imgur.com/tqpje3M.png" width="600"> </td>
    <td> <img src = "https://i.imgur.com/DQ6SYaH.png" width="300"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Selective Kernel Feature Fusion (SKFF)</b></p></td>
    <td><p align="center"> <b>Downsampling Module</b></p></td>
  </tr>
</table>

<table>
<tr>
    <td> <img src = "https://i.imgur.com/FmHQ0VD.png" width="600"> </td>
    <td> <img src = "https://i.imgur.com/aOAFSkq.png" width="300"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Dual Attention Unit (DAU)</b></p></td>
    <td><p align="center"><b>Upsampling Module</b></p></td>
  </tr>
</table>

</details>

## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
sudo apt-get install cmake build-essential libjpeg-dev libpng-dev
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```


## Training
1. Download the SIDD-Medium dataset from [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)
2. Generate image patches
```
python generate_patches_SIDD.py --ps 256 --num_patches 300 --num_cores 10
```
3. Download validation images of SIDD and place them in `../SIDD_patches/val`
 
4. Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

5. Train your model with default arguments by running

```
python train_denoising.py
```

**Note:** Our model is trained with 2 Nvidia Tesla-V100 GPUs. See [#5](https://github.com/swz30/MIRNet/issues/5) for changing the model parameters.  

## Evaluation
You can download, at once, the complete repository of MIRNet (including pre-trained models, datasets, results, etc) from this Google Drive  [link](https://drive.google.com/drive/folders/1C2XCufoxxckQ29EkxERFPxL8R3Kx68ZG?usp=sharing), or evaluate individual tasks with the following instructions:

### Image Denoising 
- Download the [model](https://drive.google.com/file/d/13PGkg3yaFQCvz6ytN99Heh_yyvfxRCdG/view?usp=sharing) and place it in ./pretrained_models/denoising/

#### Testing on SIDD dataset
- Download sRGB [images](https://drive.google.com/drive/folders/1j5ESMU0HJGD-wU6qbEdnt569z7sM3479?usp=sharing) of SIDD and place them in ./datasets/sidd/
- Run
```
python test_sidd_rgb.py --save_images
```
#### Testing on DND dataset
- Download sRGB [images](https://drive.google.com/drive/folders/1-IBw_J0gdlM6AlqSm3Z7XWTXR-So4xzp?usp=sharing) of DND and place them in ./datasets/dnd/
- Run
```
python test_dnd_rgb.py --save_images
```
### Image Super-resolution
- Download the [models](https://drive.google.com/drive/folders/1yMtXbk6RXoFfmeRRGu1XfNFSHH6bSUoR?usp=sharing) and place them in ./pretrained_models/super_resolution/
- Download [images](https://drive.google.com/drive/folders/1mAr0YCqBJFXsnOnOp0WWxkAiGF9DQAe8?usp=sharing) of different scaling factor and place them in ./datasets/super_resolution/
- Run
```
python test_super_resolution.py --save_images --scale 3
python test_super_resolution.py --save_images --scale 4
```

### Image Enhancement 
#### Testing on LOL dataset
- Download the LOL [model](https://drive.google.com/file/d/1t_FcBuMZD5th2KWVVNXYGJ7bMz5ZAWvF/view?usp=sharing) and place it in ./pretrained_models/enhancement/
- Download [images](https://drive.google.com/drive/folders/1LR6J4tkG6DLHqsipsMgHgU_p1xOZjdAA?usp=sharing) of LOL dataset and place them in ./datasets/lol/
- Run
```
python test_enhancement.py --save_images --input_dir ./datasets/lol/ --result_dir ./results/enhancement/lol/ --weights ./pretrained_models/enhancement/model_lol.pth
```
#### Testing on Adobe-MIT FiveK dataset
- Download the FiveK [model](https://drive.google.com/file/d/1BsXOvhMz2z80E_V93dgD6QaEspZE0w-u/view?usp=sharing) and place it in ./pretrained_models/enhancement/
- Download some sample [images](https://drive.google.com/drive/folders/1tyrELge59GdhZ18VR6yFwVb5Kenq2hSd?usp=sharing) of fiveK dataset and place them in ./datasets/fivek_sample_images/
- Run
```
python test_enhancement.py --save_images --input_dir ./datasets/fivek_sample_images/ --result_dir ./results/enhancement/fivek/ --weights ./pretrained_models/enhancement/model_fivek.pth
```


## Results

Experiments are performed on five real image datasets for different image processing tasks including, image denoising, super-resolution and image enhancement. Images produced by MIRNet can be downloaded from Google Drive [link](https://drive.google.com/drive/folders/1z6bFP7ydBaQOPmk8n1byYY0xcLx7aBHp?usp=sharing).

<details>
  <summary> <strong>Image Denoising</strong> (click to expand) </summary>
<img src = "https://i.imgur.com/te123qk.png" ></details>

<details>
  <summary> <strong>Image Super-resolution </strong> (click to expand) </summary>
<img src = "https://i.imgur.com/pBdUPXa.png" ></details>

<details>
  <summary> <strong>Image Enhancement</strong> (click to expand) </summary>
<img src = "https://i.imgur.com/TZRBlux.png" ></details>

## Other Implementations
- [Tensorflow](https://github.com/soumik12345/MIRNet) (Soumik Rakshit)
- [Tensorflow-JS](https://github.com/Rishit-dagli/MIRNet-TFJS) (Rishit Dagli) 
- [Tensorflow-TFLite](https://github.com/sayakpaul/MIRNet-TFLite-TRT) (Sayak Paul)


## Citation
If you use MIRNet, please consider citing:

    @inproceedings{Zamir2020MIRNet,
        title={Learning Enriched Features for Real Image Restoration and Enhancement},
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
                and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
        booktitle={ECCV},
        year={2020}
    }

## Contact
Should you have any question, please contact waqas.zamir@inceptioniai.org

## Our Related Works
- Learning Enriched Features for Fast Image Restoration and Enhancement, TPAMI 2022. [Paper](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/) | [Code](https://github.com/swz30/MIRNetv2)
- Restormer: Efficient Transformer for High-Resolution Image Restoration, CVPR 2022. [Paper](https://arxiv.org/abs/2111.09881) | [Code](https://github.com/swz30/Restormer)
- Multi-Stage Progressive Image Restoration, CVPR 2021. [Paper](https://arxiv.org/abs/2102.02808) | [Code](https://github.com/swz30/MPRNet)
- CycleISP: Real Image Restoration via Improved Data Synthesis, CVPR 2020. [Paper](https://arxiv.org/abs/2003.07761) | [Code](https://github.com/swz30/CycleISP)
