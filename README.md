# Learning Enriched Features for Real Image Restoration and Enhancement

This repository is for MIRNet introduced in the following paper

[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), and [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en) "Learning Enriched Features for Real Image Restoration and Enhancement"

**Paper**: https://arxiv.org/abs/2003.06792

**Supplementary**: [pdf](https://drive.google.com/file/d/1QIKp7h7Rd85odaS6bDoeDGXb0VLKo8I9/view?usp=sharing)

## Codes and Pre-trained Models Releasing Soon! 

## Abstract

With the goal of recovering high-quality image content from its degraded version, image restoration enjoys numerous applications, such as in surveillance, computational photography, medical imaging, and remote sensing.  Recently, convolutional neural networks (CNNs) have achieved dramatic improvements over conventional approaches for image restoration task. Existing CNN-based methods typically operate either on full-resolution or on progressively low-resolution representations. In the former case, spatially precise but contextually less robust results are achieved, while in the latter case, semantically reliable but spatially less accurate outputs are generated. In this paper, we present a novel architecture with the collective goals of maintaining spatially-precise high-resolution representations through the entire network, and receiving strong contextual information from the low-resolution representations.  The core of our approach is a multi-scale residual block containing several key elements: (a) parallel multi-resolution convolution streams for extracting multi-scale features, (b) information exchange across the multi-resolution streams, (c) spatial and channel attention mechanisms for capturing contextual information, and (d) attention based multi-scale feature aggregation. In the nutshell, our approach learns an enriched set of features that combines contextual information from multiple scales, while simultaneously preserving the high-resolution spatial details. Extensive experiments on five real image benchmark datasets demonstrate that our method, named as MIRNet, achieves state-of-the-art results for a variety of image processing tasks, including image denoising, super-resolution and image enhancement. 

## Network Architecture
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


## Results
Experiments are performed on five real image datasets for different image processing tasks including, image denoising, super-resolution and image enhancement.

### Image Denoising

### Image Super-resolution 

### Image Enhancement



## Citation
If you use MIRNet, please consider citing:

    @article{Zamir2020MIRNet,
        title={Learning Enriched Features for Real Image Restoration and Enhancement},
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
                and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
        journal={arXiv preprint arXiv:2003.06792},
        year={2020}
    }

## Contact
Should you have any question, please contact waqas.zamir@inceptioniai.org
