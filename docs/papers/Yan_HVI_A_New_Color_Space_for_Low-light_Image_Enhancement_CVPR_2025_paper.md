# HVI: A New Color Space for Low-light Image Enhancement

**Qingsen Yan**$^{1*}$, **Yixu Feng**$^{1*}$, **Cheng Zhang**$^1$, **Guansong Pang**$^{2\dagger}$, **Kangbiao Shi**$^1$, **Peng Wu**$^1$  
$^1$ Northwestern Polytechnical University, China  
$^2$ Singapore Management University, Singapore  
$^3$ Xi’an University of Architecture and Technology, China  
Code: https://github.com/Fediory/HVI-CIDNet | $^*$ Equal contribution, $^\dagger$ Corresponding author

---

## Abstract

Low-Light Image Enhancement (LLIE) is a crucial computer
vision task that aims to restore detailed visual information
from corrupted low-light images. Many existing LLIE methods are based on standard RGB (sRGB) space, which often
produce color bias and brightness artifacts due to inherent
high color sensitivity in sRGB. While converting the images
using Hue, Saturation and Value (HSV) color space helps
resolve the brightness issue, it introduces significant red and
black noise artifacts. To address this issue, we propose a new
color space for LLIE, namely Horizontal/Vertical-Intensity
(HVI), defined by polarized HS maps and learnable intensity. The former enforces small distances for red coordinates
to remove the red artifacts, while the latter compresses the
low-light regions to remove the black artifacts. To fully leverage the chromatic and intensity information, a novel Color
and Intensity Decoupling Network (CIDNet) is further introduced to learn accurate photometric mapping function
under different lighting conditions in the HVI space. Comprehensive results from benchmark and ablation experiments
show that the proposed HVI color space with CIDNet outperforms the state-of-the-art methods on 10 datasets. The
code is available at https://github.com/Fediory/HVI-CIDNet.

## 1. Introduction

Under low-light conditions, imaging sensors often capture weak light signals with severe noise, resulting in poor
visual quality for low-light images. Obtaining high-quality
images from such degraded images necessitates Low-Light
Image Enhancement (LLIE) that aims at improving the image brightness while simultaneously reducing the impact of
noise and color bias [24].

## 2. Related Work

The majority of existing LLIE approaches [10, 15, 18), 35,
51, 55] focus on finding an appropriate image brightness,
which is typically done by employing deep neural networks
to learn a mapping relationship between low-light images

and normal-light images within the standard RGB (sRGB)
space. However, the image brightness exhibits a strong
coupling with the color from the three sRGB channels, $a . k . a .$
high color sensitivity in [9, 23], causing an obvious color
distortion of the restored image in these LLIE methods [3,
18], as shown in Fig. 1 (a).

Inspired by Kubelka-Munk theory [9], recent methods
[25, 52, 56] have sought to transform images from the sRGB
color space to the Hue, Saturation and Value (HSV) color
space. These methods help achieve the brightness enhancement more accurately, but they amplify local color space
noise [9], introducing severe artifacts in the results. As illustrated in Fig. 1 (b), the transformation from sRGB to the
HSV disrupts the continuity of the red (① Red Discontinuity
Noise) and black (② Black Plane Noise) color, resulting in
increased Euclidean distances for similar color and the introduction of artifacts in the final images (see the zoomed
in images of $^D$ and $\Theta$ ). These two types of noise can cause
severe artifacts in the enhancement of red-dominated or extremely dark images.

To address these issues, we introduce a new color space
named Horizontal/Vertical-Intensity (HVI), designed specifically for the LLIE task. The key intuition is that minimizing
color space noise, by reducing Euclidean distances in similar
colors. To this end, we polarize in the Hue and Saturation
(HS) plane to enforce smaller distances for similar red point
coordinates, which eliminates the red discontinuity noise
in the primary HSV space (see Fig. 1 (c)). For the black
plane noise issue, we introduce a trainable darkness density
parameter $k$ and its corresponding adaptive intensity collapse function $C _ { k } ,$ which compresses the radius of low-light
regions to zero, with the flexibility to gradually expand to
the value of one as the intensity increases, as illustrated in
Fig. 1 (d). This helps remove the black noise artifacts while
maintaining the primary image appearance.

We further propose an LLIE network, named Color and
Intensity Decoupling Network (CIDNet), to leverage the
chromatic and intensity information for optimizing the color
and intensity in the HVI space. After transforming the image into the HVI space, CIDNet leverages two network

*These authors contributed equally to this work.

$^ \dagger \mathrm { C o r r e s p o n d i n g }$ authors.

ecoupling
SRGB
HSV
 Horizontal/Vertical-Intensity
Brightnessland Color
larization
nable
Intensity
High Color Sensitivity
(a)
(b)
(d)

> **Figure 1. The top row illustrates the process of transforming images from the sRGB color space, via HSV, to the HVI color space. The bottom row presents the corresponding test results. The sRGB color space is known for its high color sensitivity, often causing color distortions in test images. By decoupling brightness and color to obtain the HSV color space, the illumination enhancement appears normalized. However, this transformation introduces varying levels of red discontinuities and black regions, which subsequently cause artifacts in the enhanced images. Introducing polarization to HSV ensures continuity in the red regions. Introducing a learnable intensity function Ck, on the other hand, helps collapse the black regions, yielding the HVI color space with optimal image enhancement.**

Red and Black Artifacts
Black Artifacts
Color Distortion
Well-enhanced
Figure 1. The top row illustrates the process of transforming images from the sRGB color space, via HSV, to the HVI color space. The
bottom row presents the corresponding test results. The sRGB color space is known for its high color sensitivity, often causing color
distortions in test images. By decoupling brightness and color to obtain the HSV color space, the illumination enhancement appears
normalized. However, this transformation introduces varying levels of red discontinuities and black regions, which subsequently cause
artifacts in the enhanced images. Introducing polarization to HSV ensures continuity in the red regions. Introducing a learnable intensity
function Ck, on the other hand, helps collapse the black regions, yielding the HVI color space with optimal image enhancement.

branches, namely HV-branch and intensity-branch, to respectively model decoupled color and brightness information
for learning accurate photometric mapping under different
lighting conditions and restoring more natural colors.

Our contributions can be summarized as follows:
• We introduce a new HVI color space for the LLIE task,
which is uniquely defined by polarized HS and trainable
intensity. This offers an effective tool that eliminates the
color space noise arising from the HSV space, largely
enhancing the brightness of the low-light images.
• We further propose a novel LLIE-baseline network, CID-
Net, to concurrently model the intensity and chromatic
of low-light images in the HVI space. Despite being lightweight and computationally-efficient with relatively small parameters (1.88M) and computational loads
(7.57GFLOPs), it enables the learning of accurate photometric mapping under different lighting conditions.
• Comprehensive results from quantitative and qualitative
experiments show that our approach outperforms various
types of state-of-the-art (SOTA) methods on different metrics across 10 datasets.

Single-stage Methods. Single-stage deep learning approaches [3, 5, 13, 18, 26, 51] has been widely used in LLIE.
Existing methods propose distinct solutions to address the

aforementioned issues. For instance, RetinexNet [43] enhances images by decoupling illumination and reflectance
based on Retinex theory. Bread [11] decouples the entanglement of noise and color distortion by using YCbCr color
space. Furthermore, they designed a color adaption network
to tackle the color distortion issue left in light-enhanced
images. Still, RetinexNet and Bread can show inaccurate
control in terms of brightness and biased color in black areas.

Diffusion-based Methods. With the advancement of
Denoising Diffusion Probabilistic Models (DDPMs) [14],
diffusion-based generative models have achieved remarkable
results in the LLIE task. It has shown the capability to
generate more accurate and appropriate images. However,
they still exhibit issues such as local overexposure or color
shift. Recent LLIE diffusion methods have attempted to
address these challenges by incorporating global supervised
brightness correction or employing local color correctors [6,
16, 17, 40, 45, 54]. Other methods such as Diff-Retinex [48]
rethink the retinex theory with a diffusion-based model in the
LLIE task, aiming to decomposed an image to illumination
and reflectance from sRGB color space. However, these
diffusion models often fail to fully decouple brightness and
color information.

RGB. Currently, the most commonly used is the sRGB
color space. For the same principle as visual recognition

by the human eye, sRGB is widely used in digital imaging
devices [32]. Nevertheless, image brightness and color exhibit a strong interdependence with the three channels in
sRGB [9]. A slight disturbance in the color space will cause
an obvious variation in both the brightness and color of the
generated image. Thus, sRGB is not a desired color space
for low-light enhancement.

HSV and YCbCr. The HSV color space represents points
in an RGB color model with a cylindrical coordinate system
[7]. Indeed, it does decouple brightness and color of the
image from RGB channels. However, the red color discontinuity and black plane noise pose significant challenges when
we enhance the images in HSV color space, resulting in the
emergence of various obvious artifacts. To circumvent issues
related to HSV, some methods [1, 11] also transform sRGB
images to the YCbCr color space, which has an illumination axis (Y) and reflect-color-plane (CbCr). Although it
solved the hue dimension discontinuity problem of HSV, the
Y axis is still coupled with the CbCr plane partially, leading
to severe color shifts.

The HVI space is built upon the HSV color space, which
is proposed to address the color space noise issues arising
from the HSV space. The key intuition in HVI is that the
restored images should have good perceptual quality for
respective colors, i.e., similar colors have small Euclidean
distances. Below we introduce the HVI transformation in
detail, where the HSV color space is first applied to decouple
the brightness and color information of input images, which
could cause color space noise (e.g., red discontinuity and
black plane). We then introduce our proposed polarized HS
operations and learnable intensity collapse function in HVI
to effectively address these issues.

Intensity Map. In the task of LLIE, one crucial aspect
is accurately estimating the illumination intensity map of
the scene from a sRGB input image. Previous methods
[8, 43, 51] largely rely on the Retinex theory [21], using deep
learning to directly generate the corresponding normal-light
map. While this approach aligns with statistical principles,
it often struggles to fit physical laws and human perception
[9], resulting in limited generalizability [40]. Therefore, we
instead refer to the Max-RGB theory [21] to estimate the
intensity map, a.k.a., Value in HSV, rather than using neural
networks to generate it. According to Max-RGB, for each
individual pixel $x ,$ we can estimate the intensity map of an
image $\mathbf { I } _ { m a x } \in \mathbb { R } ^ { \mathrm { H \times W } }$ as follows:

\mathbf {I}_{max}(x) =\max _{\mathbf {c}\in \{R,G,B\}} (\mathbf {I_{c}}(x)). \label {eq:1}

transformation that can lead to various red and black noises.
We introduce these separately as follows.

Hue/Saturation Plane. Real-world low-light images often contain significant noise, making its identification and
removal a key challenge in LLIE. Recent studies [43, 48] indicate that the noise in low-light images is a primary cause to
shifts in Hue and Saturation, a.k.a. a general case of Reintex
theory [40], while having minimal impact on light intensity.
Therefore, decoupling the sRGB color space, known for its
high color sensitivity, can be advantageous for the LLIE task.
By leveraging pixel-based photometric invariance [9] and
dichromatic reflection modeling [34], sRGB can be decoupled into illuminance and chromatic components, yielding
the HSV (Hue/Saturation-Value) color space. In this representation, the Value map (V) component corresponds to
light intensity map $( \mathbf { V } = \mathbf { I } _ { m a x } ) ,$ while the HS plane forms a
chromaticity plane independent of illuminance constraints.
Specifically, the transformation of sRGB image to Saturation
map (S) is defined as follows

\mathbf {s} = \begin {cases} 0, & \mathbf {I}_{max} = \mathbf {0} \\ \frac {\Delta }{\mathbf {I}_{max} }, & \mathbf {I}_{max} \ne \mathbf {0} \\ \end {cases} \\ ,
(2)

where $\Delta = \mathbf { I } _ { m a x } - m i n ( \mathbf { I } _ { c } )$ and $\mathbf { s }$ is any pixel in $\mathbf { S } .$ The
Hue map (H) is formulated as
(0.
if s = 0
(3)
\mathbf {h}= \begin {cases} 0, &\text {if } \mathbf {s} = 0 \\ \frac {\mathbf {I_{G}} - \mathbf {I_{B}}}{\Delta } \mod {6}, &\text {if } \mathbf {I}_{max} = \mathbf {I_{R}} \\ 2+\frac {\mathbf {I_{B}} - \mathbf {I_{R}}}{\Delta }, &\text {if } \mathbf {I}_{max} = \mathbf {I_{G}} \\ 4+\frac {\mathbf {I_{R}} - \mathbf {I_{G}}}{\Delta }, &\text {if } \mathbf {I}_{max} = \mathbf {I_{B}} \end {cases} , \label {eq:hue}

where h is any pixel in H.

Color Space Noises. Converting an sRGB image to HSV
space effectively decouples brightness from color, enabling
more accurate color denoising and more natural illuminance
recovery. However, this transformation also amplifies noise
in the red and dark regions [9], which are critical for LLIE
tasks. As illustrated in Fig. 1 (b), enhancing the image
brightness within the HSV color space yields a more balanced brightness level. However, excessive noise in the red
discontinuities and the black plane introduces significant
artifacts, particularly in red and dark areas of output image,
which greatly degrade the perceptual quality.

To address this issue, we propose the HVI color space
as follows, which effectively preserves the decoupling of
brightness and color while minimizing these artifacts.

Our primary approach to addressing the color space noise
(1) issue is to ensure that more similar colors exhibit smaller
Euclidean distances. Along the Hue axis, the color red
appears identically at both $\mathbf { h } = 0$ and $\mathbf { h } = \, 6 , \,$ due to the

modular arithmetic of Hue-axis, which splits the same color
across two ends of the spectrum. In particular, to address
the red discontinuity issue, we apply polarization to the
Hue axis (h) at each pixel in $\mathbf { H } ,$ obtaining orthogonal $h =$
$\cos ( \frac { \pi \mathrm { h } } { 3 } )$ and $v = \sin ( \frac { \pi \mathrm { h } } { 3 } ) .$ When the Hue axis is polarized,
it forms an angle within the orthogonalized $h - v$ plane, with
s representing the distance from the origin.

For the black plane noise issue, we aim to collapse regions
of low light intensity while preserving those with higher intensity. However, the optimal extent of collapse varies across
different datasets and networks. Therefore, it is important to
make this region adaptively collapsible through a learning
process. To achieve this, we introduce an adaptive intensity
collapse function $\mathbf { C } _ { k }$ as follows

$\mathbf{C}_k(x) = \left(\sqrt{\sin\left(\frac{\pi I_{max}(x)}{2}\right)}\right)^k + \varepsilon$

where $k \in \mathbb { Q } ^ { + }$ is a trainable parameter to control the dark
color point density, and a small $\varepsilon = 1 \! \times \! 1 0 ^ { - 8 }$ is used to avoid
gradient explosion. Essentially, $\mathbf { C } _ { k }$ serves a radius mapping
function, with smaller $\mathbf { C } _ { k }$ corresponding to smaller radius
or lower intensity values. Thus, black points are clustered
together as $\mathbf { C } _ { k }$ decreases. We then formalize the Horizontal
(H) map and Vertical (V) map as

\begin {split} \mathbf {\hat {H}} &= \mathbf {C}_k \odot \mathbf {S} \odot H,\\ \mathbf {\hat {V}} &= \mathbf {C}_k \odot \mathbf {S} \odot V, \end {split} \label {eq:7}

where $h \, \in \, H , \, v \, \in \, V , \,$ and $\odot$ denotes the element-wise
multiplication. $\mathbf { \hat { H } } ,$ $\mathbf { \hat { V } } ,$ and $\mathbf { I } _ { m a x }$ can be concatenated to form

an HVI image.

Thanks to these operations, the HVI space builds a strong
color space that maintains the advantages of HSV while removing the red and black noises. More importantly, since
HVI is trainable due to $k$ and $\mathbf { C } _ { k } ,$ specifically designed neural networks, as discussed in the following, can be created to
optimize LLIE upon HVI under various lighting conditions.

> **Figure 2. The overview of the proposed CIDNet. (a) HVI Color Transformation (HVIT) takes an sRGB image as input and generates HV color map and intensity map as outputs. (b) Enhancement Network performs the main processing, utilizing a dual-branch UNet architecture with six Lighten Cross-Attention (LCA) blocks. Lastly, we apply Perceptual-inverse HVI Transform (PHVIT) to take a light-up HVI map as input and transform it into an sRGB-enhanced image.**

To more effectively utilize chromatic and brightness information in the HVI space, we introduce a novel dual-branch
LLIE network, named Color and Intensity Decoupling Network (CIDNet), to separately model the HV-plane and I-axis
information in the HVI space, as shown in Fig. 2. CIDNet
employs HV-branch to suppress the noise and chromaticity in the dark regions and utilizes I-branch to estimate the
illuminance of the whole images.

The overall framework of CIDNet can be divided into
three consecutive main steps. There is an HVI transformation
applied before the dual-branch enhancement network. After
the enhancement, CIDNet performs perceptual-inverse HVI
transformation to map the image to the sRGB space. Below
we introduce each step in detail.

As described in Sec. 3.2, the HVI transformation decomposes the sRGB image into two components: an intensity
map containing scene illuminance information and an HV
color map containing scene color and structure information.
Specifically, we first calculate the intensity map using Eq. 1,
which is $\mathbf { I _ { I } } = \mathbf { I } _ { m a x } .$ Subsequently, we utilize the intensity
map and the original image to generate HV color map using
Eq. 5. Furthermore, a trainable density-k is employed to
adjust the color point density of the low-intensity color plane,
as shown in Fig. 2(a).

As illustrated in Fig. 2 (b), the dual-branch network is
built upon the UNet architecture, involving an encoder and
an decoder with respective three Lighten Cross-Attention
(LCA) modules, and multiple skip connections. Two key
designs in the network are the dual-branch structure and the
LCA module. We explain the key intuition behind these
designs as follows.

The LLIE task can be decomposed into two sub-tasks:
noise removal in low-light regions and brightness enhancement. By converting the image to the HVI color space, where
luminance and color are decoupled, we can apply brightness
mapping to the Intensity map and denoising to the HV color
(5) map. Since these two sub-tasks follow distinct statistical
patterns [24], inspired by Retinex-based methods [8, 43, 44],
we use separate branches, the I-branch and HV-branch, to
address each sub-task individually. Additionally, the input
to the HV branch is formed by concatenating the Intensity
map with the HV color map, as we observed that severely
low-light images contain a small amount of noise in the
luminance component as well.

The cross-attention between the two branches is used,
rather than using self-attention individually to each branch.
One main reason is that the illumination intensity is inversely
proportional to image noise intensity. A low-light image
may also contain high-illumination regions requiring only
minimal denoising and enhancement. Therefore, using the
intensity features to guide the HV-branch in denoising can
reduce global color shifts and achieve more effective noise
suppression. For another reason, the noisy intensity information, after being denoised in the HV-branch, is transferred to
the I-branch through cross-attention, resulting in smoother
enhancement outcomes.

To convert HVI back to the HSV color space, we perform
a Perceptual-inverse HVI Transformation (PHVIT), which
is a surjective mapping while allowing for the independent
adjustment of the image’s saturation and brightness.

The PHVIT sets $\hbar$ and $\hat { v }$ as an intermediate variable as

$\hat{h} = \frac{H}{\mathbf{C}_k + \varepsilon}, \quad \hat{v} = \frac{V}{\mathbf{C}_k + \varepsilon}$ where $\varepsilon = 1 \times 1 0 ^ { - 8 }$ is used to avoid

(a) HVI C
ol
(b) Enhan
nt Network
Transfor
ation
In
Density-k
dan

> **Figure 2. The overview of the proposed CIDNet. (a) HVI Color Transformation (HVIT) takes an sRGB image as input and generates HV color map and intensity map as outputs. (b) Enhancement Network performs the main processing, utilizing a dual-branch UNet architecture with six Lighten Cross-Attention (LCA) blocks. Lastly, we apply Perceptual-inverse HVI Transform (PHVIT) to take a light-up HVI map as input and transform it into an sRGB-enhanced image.**

Figure 2. The overview of the proposed CIDNet. (a) HVI Color Transformation (HVIT) takes an sRGB image as input and generates HV
color map and intensity map as outputs. (b) Enhancement Network performs the main processing, utilizing a dual-branch UNet architecture
with six Lighten Cross-Attention (LCA) blocks. Lastly, we apply Perceptual-inverse HVI Transform (PHVIT) to take a light-up HVI map as
input and transform it into an sRGB-enhanced image.

gradient explosion. Then, we convert $\hbar$ and $\hat { v }$ to HSV color
space. The Hue (\mathbf{H}), Saturation (S) and Value $( \mathbf { V } )$ map can
be estimated as

\begin {split} \mathbf {H} &= \arctan (\frac {\hat {v}}{\hat {h}})\mod 1,\\ \mathbf {S} &= \alpha _{S}\sqrt {\hat {h}^{2}+\hat {v}^{2}},\\ \mathbf {V} &= \alpha _{I}\hat {\mathbf {I}}_{\mathbf {I}}, \end {split}
$\mathbf{S} = \alpha_S \sqrt{\hat{h}^2 + \hat{v}^2}$
$\mathbf{V} = \alpha_I \mathbf{\hat{I}}_I$

where $\; \alpha _ { S } , \alpha _ { I } \;$ are the ${ }$ customizing $\mathrm { \ l i n e a r ~ }$ parameters $\mathrm { ~ t o }$

change the image color saturation and brightness. Finally,
we will obtain the sRGB image with HSV image [7].

To provide comprehensive supervision for training CID-
Net, we guide the enhancement from two key perspectives,
including the GroundTruth in the sRGB space and the HVI
map in the HVI space. Specifically, given the enhanced
HVI map $\mathbf { I } _ { \mathrm { H V I } }$ and the restored sRGB image $\mathbf { I }$ output from
CIDNet, we aim to minimize their difference to the primary
sRGB GroundTruth I and its corresponding HVI map IHVI:
$( 7 )$
$\mathcal{L} = \lambda \cdot l(\mathbf{\hat{I}}_{HVI}, \mathbf{I}_{HVI}) + l(\mathbf{\hat{I}}, \mathbf{I})$

where λ is a weighting hyperparameter to balance the losses
in the two different color spaces. This helps achieve not only
more closely with the probabilistic distribution of sRGB in
the HVI space, especially the red and black ones, due to
the optimized $k$ and $\mathbf { C } _ { k } ,$ but also inheritance of pixel-level
structure detail in the sRGB space.

We employ seven commonly-used LLIE benchmark
datasets for evaluation, including LOLv1 [43], LOLv2 [47],
DICM [22], LIME [12], MEF [28], NPE [38], and VV [36].

We also conduct further experiments on two extreme datasets,
SICE [2] (containing mix and grad test sets [53]) and SID
(Sony-Total-Dark) [4].

### 5.1. Datasets and Implementation Details

LOL. The LOL dataset has v1 [43] and v2 [47] versions.
Compared to LOL-v1 that contains both real and synthetic
(6) data, LOL-v2 is divided into real and synthetic subsets. For
LOLv1 and LOLv2-Real, we crop the training images into
$4 0 0 \times 4 0 0$ patches and train CIDNet for 1,500 epochs with a
batch size of 8. For LOLv2-Synthetic, we set the batch size
to 1 and train CIDNet for 500 epochs without cropping.

SICE. The original SICE dataset [2] contains 589 lowlight and overexposed images, with the training, validation,
and test sets divided into three groups according to 7:1:2.
CIDNet is optimized using $1 6 0 \times 1 6 0$ cropped images from
the training set for 1,000 epochs with a batch size of 10 and
tested on the datasets SICE-Mix and SICE-Grad [53].

Sony-Total-Dark. This dataset is a customized version
of a SID subset [4]. To make this dataset more challenging,
we convert the raw format images to sRGB images with no
gamma correction, resulting in images with extreme darkness. We crop the training images into $2 5 6 \times 2 5 6$ patches
and train CIDNet for 1,000 epochs with a batch size of 4.

Experiment Settings. We implement our CIDNet by
PyTorch. The model is trained with the Adam [19] optimizer
$(\beta_1 = 0.9,$ and $\beta_2 = 0.999)$ by using a single NIVIDA 2080Ti
or $3 0 9 0$ GPU. The learning rate is initially set to $1 \! \times \! 1 0 ^ { - 4 }$ and
then steadily decreased to $1 \times 1 0 ^ { - 7 }$ by the cosine annealing
scheme [27] during the training process.

Evaluation Metrics. For the paired datasets, we adopt
the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM) [42] as the distortion metrics. To comprehensively evaluate the perceptual quality of the restored
images, we report Learned Perceptual Image Patch Similarity (LPIPS) [50] with AlexNet [20] as the reference. For
the unpaired datasets, we evaluate single recovered images
using BRISQUE [29] and NIQE [30] perceptually.

**Table 1. Quantitative results of PSNR/SSIM↑ and LPIPS↓ on the LOL (v1 and v2) datasets. Due to the limited number of test set in LOLv1,
we use GT mean method during testing to minimize errors. This approach will be explained in supplementary. The FLOPs is tested on a
single $2 5 6 \times 2 5 6$ image. The best performance is in red color and the second best is in cyan color.**

| Method (Methods) | Color Model | Params (M) | FLOPs (G) | LOLv1 PSNR↑ | SSIM↑ | LPIPS↓ | LOLv2-Real PSNR↑ | SSIM↑ | LPIPS↓ | LOLv2-Synthetic PSNR↑ | SSIM↑ | LPIPS↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RetinexNet [43] | Retinex | 0.84 | 584.47 | 18.915 | 0.427 | $0 . 4 7 0$ | 16.097 | 0.401 | 0.543 | 17.137 | $0 . 7 6 2$ | 0.255 |
| KinD [51] | Retinex | 8.02 | 34.99 | 23.018 | 0.843 | 0.156 | 17.544 | 0.669 | 0.375 | 18.320 | 0.796 | 0.252 |
| ZeroDCE [10] | RGB | 0.075 | 4.83 | 21.880 | $0 . 6 4 0$ | 0.335 | 16.059 | 0.580 | 0.313 | 17.712 | 0.815 | 0.169 |
| RUAS [33] | Retinex | $0 . 0 0 3$ | $0 . 8 3$ | 18.654 | $0 . 5 1 8$ | $0 . 2 7 0$ | 15.326 | 0.488 | $0 . 3 1 0$ | 13.765 | 0.638 | 0.305 |
| LLFlow [41] | RGB | 17.42 | 358.40 | 24.998 | 0.871 | 0.117 | 17.433 | 0.831 | 0.176 | 24.807 | 0.919 | 0.067 |
| EnlightenGAN [18] | RGB | 61.01 | 114.35 | 20.003 | 0.691 | 0.317 | 18.230 | 0.617 | 0.309 | 16.570 | 0.734 | 0.220 |
| SNR-Aware [46] | SNR+RGB | 4.01 | 26.35 | 26.716 | $0 . 8 5 1$ | 0.152 | 21.480 | 0.849 | 0.163 | 24.140 | 0.928 | $0 . 0 5 6$ |
| Bread [11] | YCbCr | $2 . 0 2$ | 19.85 | 25.299 | 0.847 | 0.155 | 20.830 | $0 . 8 4 7$ | 0.174 | 17.630 | 0.919 | $0 . 0 9 1$ |
| PairLIE [8] | Retinex | 0.33 | 20.81 | 23.526 | 0.755 | 0.248 | 19.885 | 0.778 | $0 . 3 1 7$ | 19.074 | 0.794 | 0.230 |
| LLFormer [39] | RGB | 24.55 | 22.52 | 25.758 | 0.823 | 0.167 | 20.056 | $0 . 7 9 2$ | 0.211 | 24.038 | 0.909 | 0.066 |
| RetinexFormer [3] | Retinex | $1 . 5 3$ | 15.85 | 27.140 | 0.850 | 0.129 | 22.794 | $0 . 8 4 0$ | 0.171 | 25.670 | 0.930 | 0.059 |
| GSAD [16] | RGB | 17.36 | 442.02 | 27.605 | 0.876 | 0.092 | 20.153 | 0.846 | 0.113 | 24.472 | 0.929 | 0.051 |
| QuadPrior [40] | Kubelka-Munk | 1252.75 | 1103.20 | 22.849 | 0.800 | $0 . 2 0 1$ | 20.592 | 0.811 | 0.202 | 16.108 | $0 . 7 5 8$ | 0.114 |
| **CIDNet (Ours)** | **HVI** | **$1 . 8 8$** | **7.57** | **28.201** | **0.889** | **0.079** | **24.111** | **0.871** | **0.108** | **25.705** | **0.942** | **0.045** |

**Table 2. Quantitative result on SICE, Sont-Total-Dark, and the five

unpaired datasets (DICM [22], LIME [12], MEF [28], NPE [38],
and VV [36]). The top-ranking score is in red.**

| Method (Methods) | SICE PSNR↑ | SSIM↑ | Sony-Total-Dark PSNR↑ | SSIM↑ | Unpaired BRISQUE↓ | NIQE↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RetinexNet [43] | 12.424 | 0.613 | 15.695 | 0.395 | 23.286 | 4.558 |
| ZeroDCE [10] | 12.452 | 0.639 | 14.087 | 0.090 | 26.343 | 4.763 |
| URetinexNet [44] | 10.899 | 0.605 | 15.519 | $0 . 3 2 3$ | $3 . 8 2 9$ | 26.359 |
| RUAS [33] | 8.656 | 0.494 | 12.622 | 0.081 | 36.372 | 4.800 |
| LLFlow [41] | 12.737 | 0.617 | $1 6 . 2 2 6$ | 0.367 | 28.087 | 4.221 |
| **CIDNet (Ours)** | **13.435** | **0.642** | **22.904** | **$0 . 6 7 6$** | **23.521** | **3.523** |

**Table 3. Results of applying HVI transformation as a plug-in to various LLIE methods on LOLv2-Real. Values in brackets represent the

absolute improved performance gain. The best PSNR/SSIM↑, LPIPS↓, and inference time↓ are in bolded.**

| Method (Methods) | Model Type | PSNR↑ | SSIM↑ | LPIPS↓ | Time GPU (s)↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FourLLIE [37] | CNN | 22.730 (+0.381) | 0.856 (+0.009) | 0.125 (+0.011) | 0.075 |
| LEDNet [57] | CNN | 23.394 (+3.456) | 0.837 (+0.010) | 0.115 (-0.005) | 0.054 |
| SNR-Aware [46] | Transformer | 22.251 (+0.771) | 0.840 (-0.009) | 0.117 (-0.054) | 0.070 |
| LLFormer [39] | Transformer | 22.671 (+2.615) | 0.852 (+0.060) | 0.117 (-0.094) | 0.139 |
| GSAD [15] | Diffusion | 23.715 (+3.562) | 0.876 (+0.030) | 0.103 (-0.010) | 0.315 |
| DiffLight [6] | CNN+Diffusion | 23.969 (+1.364) | 0.859 (+0.003) | 0.109 (-0.012) | 0.578 |
| **CIDNet (Ours)** | **Transformer** | **24.111** | **$0 . 8 7 1$** | **0.108** | **0.053** |

**Table 4. Model ablation. Dual denotes the dual-branch network.**

| Evaluation Aspect | Model Variant | PSNR↑ | SSIM↑ | LPIPS↓ |
| :--- | :--- | :---: | :---: | :---: |
| **Color Space** | sRGB | 20.062 | 0.825 | 0.137 |
| | $\mathrm { H S V }$ | 21.349 | 0.801 | 0.167 |
| | HVI (Chỉ Phân cực - w/ Polarization Only) | 21.558 | 0.821 | 0.149 |
| | HVI (Chỉ $\mathbf { C } _ { k }$ - w/ $C_k$ Only) | 21.536 | 0.825 | 0.179 |
| **Cấu trúc Mạng (Structure)** | UNet Baseline [31] | 19.306 | 0.778 | 0.222 |
| | SelfAttn [49] | 22.313 | $0 . 8 3 5$ | 0.126 |
| | Dual + SelfAttn [49] | 23.159 | 0.856 | 0.116 |
| **Hàm Mất mát (Loss)** | Chỉ HVI (HVI Only) | 23.221 | 0.854 | 0.132 |
| | Chỉ sRGB (sRGB Only) | 23.319 | 0.857 | 0.123 |
| **Mô hình Đầy đủ** | **Full Model (HVI-CIDNet)** | **24.111** | **0.871** | **0.108** |

Dual-branch Network Structure. In Tab. 4, adding
self-attention to the baseline noticeably improves all three
metrics, indicating that transformer-based models hold potential for application in the HVI color space. We then modified
the architecture from a single-branch to a dual-branch structure without cross-attention, resulting in a PSNR increase of
0.846 dB, while SSIM and LPIPS showed minimal change.
Further incorporating the cross-attention into the I-branch
and HV-branch (Full Model) obtains the best color restoration, light enhancement, and optimal metric performance, as
shown in Tab. 4. This can also be observed in Fig. 6.

Loss Function. As shown in Tab. 4, compared to using
both HVI and sRGB losses, relying solely on the HVI loss
lacks pixel-level spatial consistency constraints, leading to
a loss of structural detail in the image and thus lower performance across the three metrics, especially in the LPIPS
metric. On the other hand, using only sRGB loss is focused on pixel-space enhancement, neglecting the low-light

SelfAttn
Dual + Self
$\mathrm { \ D u a l } + \mathrm { C r o s s }$
GroundTruth
Figure 6. Results of using different structures on LOLv2-Real.
probability distribution in the HVI color space, resulting in
undesired color imbalance.

In this work, we introduce the HVI color space and the
CIDNet approach to address the color bias and brightness
artifact issues that occur to current sRGB-based LLIE approaches. By encapsulating polarized HS maps and a learnable intensity component, HVI shows strong robustness to
both issues. To further enhance LLIE, CIDNet is designed to
model decoupled chromatic and intensity information in the
HVI space for achieving precise photometric adjustments
under varying lighting conditions. Experimental results on
10 datasets demonstrate that the HVI color space, combined
with CIDNet, outperforms SOTA LLIE methods, establishing it as a robust solution for low-light enhancement.

## Acknowledgments

This work was supported by NSFC of China under Grant 62301432 and Grant
6230624, the Natural Science Basic Research Program
of Shaanxi under Grant 2023-JC-QN-0685 and Grant
QCYRCXM-2023-057, the Fundamental Research Funds

## References

[1] A. Brateanu, R. Balmez, A. Avram, and C. Orhei. Lyt-net: Lightweight yuv transformer-based network for low-light image enhancement, 2024.

[2] Jianrui Cai, Shuhang Gu, and Lei Zhang. Learning a deep single image contrast enhancer from multi-exposure images. IEEE Transactions on Image Processing, 27(4):2049–2062, 2018.

[3] Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, and Yulun Zhang. Retinexformer: One-stage retinex-based transformer for low-light image enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12504–12513, 2023.

[4] Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. Learning to see in the dark. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3291–3300, 2018.

[5] Liangyu Chen, Jiawei Zhang, Xiaojie Guo, and Jinshan Pan. Retinex-based deep illumination estimation for low-light image enhancement. IEEE Transactions on Multimedia, 2023.

[6] Yixu Feng, Shuo Hou, Haotian Lin, Yu Zhu, Peng Wu, Wei Dong, Jinqiu Sun, Yanning Zhang, and Qingsen Yan. Difflight: Integrating content and detail for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6580–6589, 2024.

[7] Alvy Ray Smith. Color gamut transform pairs. ACM Siggraph Computer Graphics, 12(3):12–19, 1978.

[8] Zhenqi Fu, Yang Yue, Shengyan Shen, Yan Zhang, and Huixing Lv. Pairlie: Large-scale pair-free low-light image enhancement, 2024.

[9] Theo Gevers and Arnold WM Smeulders. Color-based object recognition. Pattern recognition, 32(3):453–464, 1999.

[10] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. Zero-reference deep curve estimation for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1780–1789, 2020.

[11] Yuekun Guo, Yuxuan Liu, Guanqing Shen, Zhiwen Shao, and Lizhuang Ma. Bread: Branch-complementary and dual-carrier network for arbitrary-scale low-light image enhancement. IEEE Transactions on Multimedia, 2024.

[12] Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light image enhancement via illumination map estimation. IEEE Transactions on image processing, 26(2):982–993, 2016.

[13] Jie Hao, Han Xu, Xiaojie Guo, Yanwei Pang, and Jichang Guo. Scl-liew: Self-supervised contrastive learning for low-light image enhancement in the wild. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[15] Shuo Hou, Yixu Feng, Haotian Lin, Peng Wu, Wei Dong, Jinqiu Sun, Yanning Zhang, and Qingsen Yan. Frequency-guided low-light image enhancement. In ACM Multimedia 2024, 2024.

[16] Shuo Hou, Yixu Feng, Peng Wu, Haotian Lin, Wei Dong, Jinqiu Sun, Yanning Zhang, and Qingsen Yan. Global structure-aware diffusion model for low-light image enhancement. arXiv preprint arXiv:2407.03964, 2024.

[17] Zhiying Jiang, Zhangkai Ni, Wei Dong, Shiqi Wang, and Sam Kwong. A coarse-to-fine framework with diffusion-based detail complement for low-light image enhancement. IEEE Transactions on Multimedia, 2024.

[18] Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, and Zhangyang Wang. Enlightengan: Deep light enhancement without paired supervision. IEEE Transactions on Image Processing, 30:2340–2349, 2021.

[19] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[20] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

[21] Edwin H Land. The retinex theory of color vision. Scientific American, 237(6):108–129, 1977.

[22] Chulwoo Lee, Chulsu Lee, and Chang-Su Kim. Contrast enhancement based on layered difference representation of 2d histograms. IEEE Transactions on Image Processing, 22(12):5372–5384, 2013.

[23] Seung-Hwan Lee, Hyeongmin Lee, Sunghyun Cho, Kwang-Hoon Sohn, and Kyunghyun Lee. Color balance: An exploratory study on the role of color in image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25963–25973, 2024.

[24] Chongyi Li, Chunle Guo, Linghao Han, Junxu Jiang, Ming-Ming Cheng, Jiange Gu, and Chen Change Loy. Low-light image and video enhancement using deep learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12):9396–9416, 2021.

[25] Jiewen Li, Xiaofeng Cong, and Jinhui Tang. Decoupled color-illumination network for low-light image enhancement. IEEE Transactions on Multimedia, 2024.

[26] Risheng Liu, Long Ma, Jiaao Zhang, Xin Fan, and Zhongxuan Luo. Retinex-inspired unrolling with architecture search for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2501–2510, 2021.

[27] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

[28] Kede Ma, Kai Zeng, and Zhou Wang. Perceptual evaluation of multi-exposure image fusion algorithms. IEEE Transactions on Image Processing, 26(10):4945–4956, 2017.

[29] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing, 21(12):4695–4708, 2012.

[30] Anish Mittal, Rajiv Soundararajan, and Alan C. Bovik. Making a “completely blind” image quality analyzer. IEEE Signal Processing Letters, 20(3):209–212, 2013.

[31] Olivier Petit, Nicolas Thome, Clément Rambour, and Luc Soler. U-net transformer: Self and cross attention for medical image segmentation. In International Workshop on Machine Learning in Medical Imaging, pages 267–276. Springer, 2021.

[32] David Pascale. A review of rgb color spaces... from srgb to dci-p3. The BabelColor Company, 2003.

[33] Risheng Liu, Long Ma, Zhiying Jiang, Jinyuan Liu, and Xin Fan. Learning to enhance low-light images via retrospective unrolling and adaptive search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(12):14264–14280, 2023.

[34] Sha Shafer. Using color to separate reflection components. COLOR research and application, 10(4):210–218, 1985.

[35] Liang Shen, Zihan Yue, Fan Feng, Quan Chen, Shan Liu, and Jie Zhou. Msr-net: Low-light image enhancement using deep convolutional network. arXiv preprint arXiv:1711.02488, 2017.

[36] Vassilios Vonikakis. Vassilios vonikakis. https://vvonikakis.com/, 2024.

[37] Cong Wang, Jinshan Pan, and Wei-Shi Zheng. Fourllie: Towards real-time low-light image enhancement with fourier transform. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8585–8593, 2023.

[38] Shuhang Wang, Jin Zheng, Hai-Miao Hu, and Bo Li. Naturalness preserved enhancement algorithm for non-uniform illumination images. IEEE Transactions on Image Processing, 22(9):3538–3548, 2013.

[39] Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, and Ming-Hsuan Yang. Ultra-high-definition low-light image enhancement: A large-scale dataset and a transformer baseline. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4601–4616, 2022.

[40] Tian Ye, Sixiang Chen, Wenhao Chai, Zhaopeng Tu, and Ernie Chu. Physics-guided generative diffusion model for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25985–25995, 2024.

[41] Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-Pui Chau, and Alex C Kot. Low-light image enhancement with normalizing flow. In Proceedings of the AAAI conference on artificial intelligence, pages 2604–2612, 2022.

[42] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

[43] Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. In Proceedings of the British Machine Vision Conference (BMVC), 2018.

[44] Wenhui Wu, Jian Weng, Pingping Zhang, Xu Wang, Wenhan Yang, and Jianmin Jiang. Uretinex-net: Retinex-based deep unfolding network for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5901–5910, 2022.

[45] Yuanzhi Zhu, Zhaofei Yu, and Tiejun Huang. Denoising diffusion models for low-light image enhancement. IEEE Transactions on Multimedia, 2024.

[46] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. Snr-aware low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17711–17720, 2022.

[47] Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, and Jiaying Liu. Sparse gradient regularized deep retinex network for robust low-light image enhancement. IEEE Transactions on Image Processing, 30:2072–2086, 2021.

[48] Hao Gao, Feifan Shao, and Sibei Yang. Diff-retinex: Generative diffusion retinex for low-light image enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12514–12523, 2023.

[49] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5728–5739, 2022.

[50] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.

[51] Yonghua Zhang, Jiawan Zhang, and Xiaojie Guo. Kindling the darkness: A practical low-light image enhancer. In Proceedings of the 27th ACM International Conference on Multimedia, pages 1632–1640, 2019.

[52] Yonghua Zhang, Xiaojie Guo, Jiayi Ma, Wei Liu, and Jiawan Zhang. Beyond brightening low-light images. International Journal of Computer Vision, 129(4):1013–1037, 2021.

[53] Zhao Zhang, Huan Zheng, Suiyi Zhao, Richang Hong, Mingliang Xu, and Shuicheng Yan. Deep color consistent network for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1899–1908, 2022.

[54] Dewei Zheng, Zhendong Mao, and Yongdong Zhang. Progressive feature-guided diffusion models for low-light image enhancement. In Proceedings of the 31st ACM International Conference on Multimedia, pages 4467–4476, 2023.

[55] Minfeng Zhu, Pingbo Pan, Wei Chen, and Yi Yang. Eemefn: Low-light image enhancement via edge-enhanced multi-exposure fusion network. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 13106–13113, 2020.

[56] Yongmin Zhuang, Chen Chen, and Ling Shao. Multi-branch network for low-light image enhancement in hsv color space. IEEE Transactions on Multimedia, 2023.

[57] Shangchen Zhou, Chongyi Li, and Chen Change Loy. Lednet: Joint low-light enhancement and deblurring in the dark. In ECCV, 2022.
