# DarkIR: Robust Low-Light Image Restoration

**Daniel Feijoo** $^{1\dagger}$, **Juan C. Benito** $^1$, **Alvaro Garcia** $^1$, **Marcos V. Conde** $^{1,2\dagger}$  
$^1$ Cidaut AI, Valladolid, Spain  
$^2$ Computer Vision Lab, University of Würzburg, Germany  
Code: https://github.com/cidautai/DarkIR | $^\dagger$ Corresponding Authors: {danfei, marcos.conde}@cidaut.es

---

> **Figure 1.** Previous Low-light Image Enhancement (LLIE) and restoration methods are not robust to blur and illumination changes. Our multi-task model is able to restore real low-light images under varying illumination, noise and blur conditions. Zoom-in to see details.

Photography during night or in dark conditions typically suffers from noise, low-light and blurring issues due
to the dim environment and the common use of long exposure. Although Deblurring and Low-light Image Enhancement (LLIE) are related under these conditions, most approaches to image restoration solve these tasks separately.
In this paper, we present an efficient and robust neural network for multi-task low-light image restoration. Instead of
following the current tendency of Transformer-based models, we propose new attention mechanisms to enhance the
receptive field of efficient CNNs. Our method reduces the
computational costs in terms of parameters and MAC operations compared to previous methods. Our model, DarkIR,
achieves new state-of-the-art results on the popular LOL-
Blur, LOLv2 and Real-LOLBlur datasets, being able to generalize on real-world night and dark images.

During night or low-light conditions, we can define the image formation process as follows

$$\mathbf{y} = \gamma(\mathbf{x} \otimes \mathbf{k}) + \mathbf{n}, \qquad (1)$$

where y is the observed dim image, x the unperturbed
captured scene, k represents the lens (point-spread function) PSF blurring kernel, n is the additive sensor noise,
and $\gamma$ is a function to control the dynamic range and pixel
saturation. We use $\otimes$ to represent the convolution operator.

In comparison with daytime conditions, at low-light, the
noise (shot and read) is substantially higher. During night
photography, cameras usually use long exposure (slow shutter speed) to allow more available light to illuminate the
image. However, long exposure could lead to ghosting and
blurring artifacts. These image degradations are more notable on smartphones, since these have a fixed aperture and
limited optics. For these reasons, joint low-light enhance-

ment and deblurring is paramount for mobile computational
photography [12, 65] — see Figure 1 samples.

Under any circumstances, the captured image will
present certain levels of noise and blur [13]. Using a tripod
to capture steady images, we can notably reduce the blur.
Moreover, if we use proper illumination, noise can also be
reduced [1, 20].

However, nowadays most photographs are captured using (handheld) smartphones. Due to the limited sensor size
and optics, these employ more complex Image Signal Processors (ISPs) [10, 12] in comparison to DSLM (Digital
Single-Lens Mirrorless) cameras. Smartphone photography is still far from DSLM quality standards, however, recent research in low-level computer vision and computational photography allows to close the gap. Low-light image enhancement (LLIE) [3, 14, 30, 31, 45, 64], image deblurring [21, 25, 38, 40] and night photography enhancement [42, 65] are popular tasks.

For instance, avoiding the presence of blur due to hand
tremor (hand shaking) has been well-studied, even in lowlight scenarios [29, 62]. However, in most cases, these tasks
are solved individually, thus, the state-of-the-art methods
for image deblurring do not generalize to nighttime images,
and the best methods for low-light enhancement cannot reduce the notable blur. This presents a clear limitation, since
multiple task-specific models need to be fine-tuned, stored
and applied in sequence, which limits their applications in
real-world cases.

To the best of our knowledge, very few works aim to
solve these tasks (denoising, deblurring and LLIE) in a
jointly end-to-end manner [6, 33, 63, 65], being LED-
Net [65] the most notable work. We focus on this research
direction since exploiting the correlation between the degradations allows us to achieve the best performance in terms
of image reconstruction, usability and efficiency.

Our contribution We propose a convolutional neural network (CNN) that operates in both the spatial and frequency
domains. In the spatial domain, we focus on solving the
noise n and the non-uniform blur k, we achieve this by using large receptive field spatial attention. On the other hand,
in the Fourier domain, we are able to enhance the low-light
conditions easily [28, 44], due to the global nature of the
task. We can summarize our contributions as:

1. We design a lightweight neural network with frequency
attention, and large receptive field attention, combining
spatial and frequency information.
2. Our model, DarkIR, achieves state-of-the-art results on
the popular LOLBlur and Real-LOLBlur datasets [65],
improving +1dB in PSNR over LEDNet [65], while having less computational cost.
3. DarkIR represents a new baseline for multi-task
night/dark image enhancement.

## 2. Related Work

### Image Deblurring

We can see decades of research on reconstructing sharp scenes. Reducing the blur in an image is
divided into blind and non-blind methods. While the nonblind methods consider the blurring kernel k (or PSF) to
process the image, the blind methods do not have any prior
knowledge about the blur degradation.

In recent years, multiple deep learning-based approaches
have been proposed for blind and non-blind deblurring [25,
38], surpassing traditional methods in both scenarios. The
non-blind approaches offer a great solution, considering
that only blurry-sharp image pairs are required for training
such models, and we do not require PSF estimation or any
information about the sensor. Most of these approaches are
sensor-agnostic i.e., they can enhance sRGB images captured from different cameras.

Nowadays, a big part of these methods are based on convolutional neural networks (CNNs) [4, 25, 37, 57]. In DeblurGAN [25], the authors use Generative Adversarial Networks (GANs) to solve this problem. More recently, the
authors of [24] implemented an efficient frequency domain
based transformer for deblurring. We also find iterative
methods and diffusion models [33, 49].

### Low-Light Image Enhancement (LLIE)

The first
methods used to consider image statistics or prior information [2, 18], being most of them based on the well-known
Retinex Theory [26].

Following the deep learning tendency, nowadays LLIE
methods are based on Convolutional Neural Networks
(CNNs) such as RetinexNet [48] (and the corresponding
LOL dataset), ZeroDCE [16] and SCI [35]. Recent methods explore the power of transformers in this task, such as
in the case of RetinexFormer [3], or use Fourier frequency
information to enhance the amplitude of the image, like in
FourLLIE [44].

### Low-Light Blur Enhancement

Even though image deblurring and low-light enhancement are tasks that capture
great attention, solving both tasks at the same time is a
challenging task, and very few works in the literature tackle
it [6, 33, 63, 65]. NBDN [6] proposes a non-blind network
to enhance night saturated images. When deconvolving the
image to its sharp version, the presence of noise or saturated
regions needs to be kept in mind by the algorithm. With this
work, the authors proved that previous methods had issues
in solving this specific task.

On the other hand, with LEDNet [65], the authors try
to solve the problem of low-light enhancement considering
that the images are also blurry. This is a realistic assumption, since smartphones need long exposure times for lowlight environments. They developed an encoder-decoder
network to solve this problem and introduced the popular
LOLBlur and Real-LOLBlur datasets.

> **Figure 2.** General diagram of DarkIR. The neural network follows an encoder-decoder architecture. We use different blocks for encoding and decoding that follow the Metaformer structure [56]. The encoder focuses on the low-light illumination issues using Fourier information. Thus, the encoder produces a low-resolution reconstructed image $\hat{x}_{\downarrow 8}$ with corrected illumination. The decoder focuses on upscaling and reducing the blur using the prior illumination-enhanced encoded features. To achieve this, the decoder uses large receptive field spatial attention. This design allows our lightweight model to have less parameters and FLOPs than previous methods.

## 3. DarkIR Architecture

We follow Metaformer [56] to design our neural network.
This design simplifies transformer-based architectures into
simple blocks with 2 components: global attention (e.g., token mixer), and a feed-forward network (FFN, MLP). The
typical formulation for these blocks is:

$$z_1 = \text{Attention}(\text{LayerNorm}(z)) + z \qquad (2)$$
$$z_2 = \text{FFN}(\text{LayerNorm}(z_1)) + z_1 \qquad (3)$$

where $z$ are the input features and $z_2$ the output features
of the block. Most popular image restoration models, such
as NAFNet [7], use the same structure and adapt the Attention module to different tasks.

We improve the metaformer structure by developing
LLIE and deblurring specific blocks. Furthermore, we use
simple gates (SG) and simplified channel attention (SCA)
instead of activations as [7].

Low-light Enhancement can be solved efficiently in the
frequency domain.

Many works [28, 44] proved that the low-light conditions
are highly correlated with the amplitude of the image in the
Fourier domain. Thus, by enhancing just the amplitude of
an image (without touching the phase), we can substantially
correct the illumination of the image. Moreover, this property stands at different resolutions [28]. Therefore, we can
estimate an illumination-enhanced image at low-resolution
and upscale it.

Sharpening and Reducing Blur usually require large receptive fields, this could be achieved by extracting deep
features while downsampling the image – NAFNet’s approach [7].
An alternative would be to use large kernels [32], however, this could lead to more computational
complexity and memory requirements.

DarkIR Model In Figure 2, we illustrate our model. Unlike most previous methods, we use two different blocks for
(3) the encoder and decoder. The idea behind this asymmetry is to perform low-light enhancement at low-resolution
in the encoder, and reduce the blur in the decoder, following a similar strategy as LEDNet [65]. The decoder will
use illumination-enhanced features from the encoder, and
shall focus on upsampling and improving the sharpness of
the already enhanced low-resolution reconstruction $\hat{x}_{\downarrow 8}$

For the encoder block, and to restore the low-light conditions, we work in the Fourier domain [28, 44]. The decoding block focuses on the spatial domain by incorporating dilated convolutions with large receptive field. By using task-specific blocks, we are able to use fewer blocks.
This design allows to reduce notably the number of parameters and computational cost in terms of MACs (Multiply-
Accumulate Operations) and FLOPs (Floating-point Operations per second).

How can we make sure the Encoder is performing lowlight enhancement? As shown in Figure 2, the encoded
features are linearly combined using a convolutional layer
to produce an intermediate image representation $\hat{x}_{\downarrow 8}$
will use this to regularize our model using an additional
loss function. By producing a good low-resolution repre-

sentation, we can ensure that the amplitude in the Fourier
domain has been properly enhanced.

### 3.1. Encoder Blocks (EBlock)

We design the encoder blocks (EBlock) to enhance the lowlight conditions of the image using Fourier information, and
following the Metaformer [56] (and NAFBlock [7]) structure. The block has two components: the spatial attention module (SpAM) and a feed-forward network in the frequency domain (FreMLP).

### 3.2. Decoder Blocks (DBlock)

The spatial attention module resembles the NAF-
Block [7] with an inverted residual block followed by simplified channel attention (SCA). Instead of using activations, we use a simple gating mechanism, allowing our
model to extract meaningful spatial information for enhancement in the frequency domain.

As suggested by other works [33, 44] the information related to the light conditions of the image depends mainly on
the amplitude in the frequency domain. To enhance this, we
apply the Fast Fourier Transform (FFT) and operate only
over its amplitude. After this operation, we transform again
to the space domain with the Inverse Fast Fourier Transform (IFFT). The Fre-MLP serves as an additional attention mechanism. In this context, the MLP operating in the
amplitude has better benefits than operating in the spatial
domain (e.g., channel MLP [56]).

The encoder uses strided convolutions to downsample
the features. After each level, the features have half of their
original spatial resolution, which implies that more encoder
blocks can be used in the deep levels without significantly
increasing the number of operations.

Finally, the encoder will provide illumination-enhanced
features to the decoder, and already a low-resolution estimation of the clean image x. This low-resolution image is xˆ↓8
estimated as a combination of the encoded deep features, it
has a resolution 8× lower than the original one. Although
it is a small estimation, the illumination (and amplitude) are
consistent across scales [28].

The decoder block (DBlock) focuses on spatial transformations. The input of the decoder is a deep representation
of $\hat{x}_{\downarrow 8}$ thus we can assume: (1) the decoder should focus
on upsampling such an initial estimation, (2) the decoder
should focus on reducing the blur and improving sharpness,
since the illumination has been corrected by the encoder. In
this block, we also maintain the metaformer structure [56]:

$$z_1 = \text{Di-SpAM}(\text{LayerNorm}(z)) + z \qquad (4)$$
$$z_2 = \text{GatedFFN}(\text{LayerNorm}(z_1)) + z_1 \qquad (5)$$

Inspired by Large Kernel Attention (LKA) [17], we created the Dilated-Spatial Attention Module (Di-SpAM). Un-

like LKA [17], we use features at 3 different levels, using three dilated depth-wise convolutions with expand (dilation) factors 1,4,9. The attributes from the three branches
are combined together, then we apply simplified channel attention to further enhance the features. Finally, we use an
MLP with simple gates instead of activations [7].

Besides the new block designs, the loss function helped to
maximize the potential of our approach. We use a combination of distortion losses and perceptual losses to optimize our model $f$ First, to ensure high-fidelity (low distortion), we use $L_{pixel}$ defined as: $L_{pixel} = \|x - \hat{x}\|_1$ where
$f(y) = \hat{x}$ and $x$ are, respectively, the enhanced and the
ground-truth (clean) images. Thus, $L_{pixel}$ is the $L_1$ loss.

To ensure high-fidelity, we use l1 norm loss and for perceptual similarity we incorporate loss $L_{percep}$ For the last
one, we use LPIPS [60] based on VGG19 [43] to calculate
the distance between features of our images:

$$L_{percep} = \text{LPIPS}(x, \hat{x}) \qquad (6)$$

Using this loss, we make sure that the network will produce a pleasant image, close to the clean reference. Following [41], we also incorporate the gradient (∇) edge loss:

$$L_{edge} = \|\nabla x - \nabla \hat{x}\|_2^2 \qquad (7)$$

that enforces consistency and accuracy in the reconstruction of the edges (high-frequencies).

Finally, similar to LEDNet [65] we included $L_{lol}$ an architecture guiding loss to assert that the encoder focuses on
the low-light enhancing. This loss works over the upsampled output of the encoder $\hat{x}_{\uparrow 8}$

$$L_{lol} = \|x - \hat{x}_{\uparrow 8}\|_1 + \text{LPIPS}(x, \hat{x}_{\uparrow 8}) \qquad (8)$$

comparing it with the reference $x$ Note that $\uparrow 8$ indicates an 8× resolution upsampling of the encoder output,
obtained by bilinear interpolation.

The complete loss function is then:

$$\mathcal{L} = \lambda_p \cdot L_{l1} + \lambda_{pe} \cdot L_{percep} + \lambda_{ed} \cdot L_{edge} + L_{lol} \qquad (9)$$
The constants $\lambda_p$ and $\lambda_{pe}, \text{ và } \lambda_{ed}$ are loss weights empirically set to 1, $1e^{-2}$ and 50 respectively.

(4) To train our model and evaluate its ability to reconstruct
(5) low-light blurred images, we use the LOLBlur dataset [65].
Although there are other datasets for this task such as
NBDN [6], LOLBlur offers a large-scale synthetic dataset
produced using a sophisticated pipeline.

**Table 1.** Quantitative evaluation on the LOLBlur dataset. DarkIR achieves new state-of-the-art results in distortion and perceptual metrics. Best and second best results are bolded and underlined, respectively.

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Params (M) ↓ | MACs (G) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| KinD++ [61] | 21.26 | 0.753 | 0.359 | 1.20 | 34.99 |
| DRBN [54] | 21.78 | 0.768 | 0.325 | 0.60 | 48.61 |
| DeblurGAN-v2 [25] | 22.30 | 0.745 | 0.356 | 60.90 | - |
| MIMO [8] | 22.41 | 0.835 | 0.262 | 6.80 | 67.25 |
| NAFNet [9] | 25.36 | 0.882 | 0.158 | 12.05 | 12.30 |
| LEDNet [65] | 25.74 | 0.850 | 0.224 | 7.40 | 38.65 |
| RetinexFormer [3] | 26.02 | 0.887 | 0.181 | 1.61 | 15.57 |
| Restormer [57] | 26.72 | **0.902** | **0.133** | 26.13 | 144.25 |
| **DarkIR-m (Ours)** | <u>27.00</u> | 0.883 | 0.162 | 3.31 | **7.25** |
| **DarkIR-l (Ours)** | **27.30** | <u>0.898</u> | <u>0.137</u> | 12.96 | <u>27.19</u> |

has 10200 training pairs, and 1800 testing
pairs. Note that this is a synthetic dataset, although generated in a realistic manner [65]: the data is generated by averaging frames to synthesize blur and darkening the normallight images with EC-Zero-DCE (a variant of Zero-DCE
[16]). We use the dataset variant that includes real sensor
noise, which makes our model more effective in challenging conditions and real scenarios.

Real-LOLBlur is a real-world test dataset, that contains 482 real-world night blurry images selected from RealBlur [40] to verify the generalization of the proposed
method. Note that these images do not have a ground-truth
since these were captured in the wild.

LOLv2 (real) is a real-world dataset that includes 689
low/high paired images for training and 100 low/high paired
images for testing. Note that LOLv2-Real [55] is the extended version of LOL[47], thus we use the v2 version directly. We also use LOLv2-Synthetic that includes 900 pairs
of low/high images for training and 100 validation ones.

LSRW includes images from a DSLM Nikon camera and
a Huawei smartphone. The LSRW-Nikon dataset is composed of 3150 training image pairs and 20 testing image
pairs. The LSRW-Huawei dataset contains 2450 pairs of
images and 30 pairs for training and validation, respectively.

We quantitatively and qualitatively evaluate the proposed
DarkIR on the LOLBlur Dataset [65]. Implementations de-

tails can be found in the supplementary. We retrained some
general purpose state-of-the-art methods in LOLBlur and
compare them with LEDNet [65] and our proposed network.We follow the baseline methods analyzed in previous
works: (1,2) zero-shot methods trained for real-world cases,
and (3) fine-tuned methods for this particular task.

1. LLIE $\to$ Deblurring. Chúng tôi consider Zero-DCE [16],
RUAS [30] and RetinexFormer [3] as LLIE models, followed by a deblurring network like MIMO-UNet [8] or
NAFNet [7].

2. Deblurring → LLIE. For deblurring, we include popular baselines such as DeblurGAN-v2 [25] trained on the
RealBlur [40] dataset, and MIMO-UNet [8] or NAFNet [7]
trained on the GoPro [38] dataset. We employ Zero-
DCE [16] and RetinexFormer [3] for light enhancement.

3. End-to-end training on the LOLBlur dataset.
We consider the following LLIE models retrained on
the LOLBlur dataset:
KinD++ [61], DRBN [54] and

RetinexFormer[3]. In addition, we consider four deblurring networks: DeblurGAN-v2 [25], NAFNet[9], MIMO-
UNet [8] and Restormer[57].

Evaluation Metrics. We employ traditional quality (distortion) metrics PSNR and SSIM for evaluation on the synthetic LOLBlur dataset. To evaluate the perceptual quality of the restored images, we use the perceptual metric
LPIPS [60] between the reference and reconstructed image.

### 4.2. Quantitative and Qualitative Results

 In Table 1 we
compare with fine-tuned methods for this task. We improve

**Table 2.** Additional quantitative evaluation on LOLBlur dataset with enhancement pipelines (Deblurring + LLIE).

| Pipeline Category | Model Combination | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :--- | :---: | :---: | :---: |
| **1. LLIE $\to$ Deblurring** | Zero-DCE [16] $\to$ MIMO [8] | 17.68 | 0.542 | 0.510 |
| | RUAS [30] $\to$ MIMO [8] | 17.81 | 0.569 | 0.523 |
| | RetinexFormer [3] $\to$ NAFNet [7] | 17.16 | 0.673 | 0.392 |
| **2. Deblurring $\to$ LLIE** | Chen [5] $\to$ Zero-DCE [16] | 17.02 | 0.502 | 0.516 |
| | DeblurGAN-v2 [25] $\to$ Zero-DCE [16] | 18.33 | 0.589 | 0.476 |
| | MIMO [8] $\to$ Zero-DCE [16] | 17.52 | 0.570 | 0.498 |
| | NAFNet [7] $\to$ RetinexFormer [3] | 14.66 | 0.500 | 0.465 |
| **Đầu-cuối (End-to-End)** | **DarkIR-m (Ours)** | **27.00** | **0.883** | **0.162** |

**Table 3.** Metrics on the LSRW dataset (50 test images from Huawei and Nikon) [19]. All the values are adopted from [35, 53].

| Method | PSNR (dB) ↑ | SSIM ↑ |
| :--- | :---: | :---: |
| RetinexNet [47] | 15.906 | 0.3725 |
| FIDE [51] | 17.669 | 0.5485 |
| DRBN [54] | 16.149 | 0.5422 |
| KinD [61] | 16.472 | 0.4929 |
| STAR [50] | 14.608 | 0.5039 |
| EnGAN [22] | 16.311 | 0.4697 |
| ZDCE [16] | 15.834 | 0.4664 |
| RUAS [30] | 14.437 | 0.4276 |
| SCI [35] | 15.017 | 0.4846 |
| **DarkIR-mt (Ours)** | **18.930** | **0.5830** |

**Table 4.** Results on LOLv2-Real [55] and LOLv2-Synthetic [55]. Our multi-task model (DarkIR-mt) obtains new SOTA results by leveraging all-in-one training.

| Method | MACs (G) ↓ | Params (M) ↓ | LOLv2-Real PSNR ↑ | LOLv2-Real SSIM ↑ | LOLv2-Syn PSNR ↑ | LOLv2-Syn SSIM ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| UFormer [46] | 12.00 | 5.29 | 18.82 | 0.771 | 19.66 | 0.871 |
| RetinexNet [47] | 587.47 | 0.84 | 15.47 | 0.567 | 17.13 | 0.798 |
| EnGAN [22] | 61.01 | 114.35 | 18.23 | 0.617 | 16.57 | 0.734 |
| RUAS [30] | 0.83 | 0.003 | 18.37 | 0.723 | 16.55 | 0.652 |
| FIDE [51] | 28.51 | 8.62 | 16.85 | 0.678 | 15.20 | 0.612 |
| DRBN [54] | 48.61 | 5.27 | 20.29 | 0.831 | 23.22 | 0.927 |
| KinD [61] | 34.99 | 8.02 | 14.74 | 0.641 | 13.29 | 0.578 |
| Restormer [57] | 144.25 | 26.13 | 19.94 | 0.827 | 21.41 | 0.830 |
| MIRNet [58] | 785.00 | 31.76 | 20.02 | 0.820 | 21.94 | 0.876 |
| SNR-Net [52] | 26.35 | 4.01 | 21.48 | 0.849 | 24.14 | 0.928 |
| FourLLIE [44] | 5.80 | 0.120 | 21.60 | 0.847 | 24.17 | 0.917 |
| Retinexformer [3] | 15.57 | 1.61 | 22.80 | 0.840 | **25.67** | 0.930 |
| **DarkIR-mt (Ours)** | **7.25** | **3.31** | **23.87** | **0.880** | 25.54 | **0.934** |

**Table 5.** Perceptual quality metrics on Real-LOLBlur [65].

| Method | MUSIQ ↑ | NRQM ↑ | NIQE ↓ |
| :--- | :---: | :---: | :---: |
| RUAS $\to$ MIMO | 34.39 | 3.322 | 6.812 |
| MIMO $\to$ Zero-DCE | 28.36 | 3.697 | 6.892 |
| RetinexFormer [3] | 45.30 | 5.281 | **4.576** |
| NAFNet [7] | **50.22** | 4.940 | 5.123 |
| LEDNet [65] | 39.11 | **5.643** | 4.764 |
| Restormer [57] | 46.60 | 4.627 | 5.268 |
| **DarkIR-m (Ours)** | 48.36 | 4.983 | 4.998 |
| **DarkIR-l (Ours)** | **48.79** | 4.917 | 5.051 |

In Table 6 we compare results obtained using different block configurations, such as NAFBlock, EBlock or
DBlock. The description states which changes are applied
to the network, and everything else is kept exactly as in
the proposed model. We can clearly see how our model
achieves the best results. Unlike the final proposed model,
all the models in this study were trained using crops of
256px instead of 384px, which explains the lower results in
general. We also studied the decoder block spatial attention
in Table 7, where the well-known Large Kernel Attention
(LKA) mechanism is compared with our Dilated-spatial Attention Module (Di-SpAM). Our approach performs better
and requires less parameters and operations.

In Table 8 we also explore the scalability of our model
by changing its channel embedding. As expected, by increasing the embedding size, the performance raises. In ascending order of parameters, we have channel embeddings
of 16, 32, and 64. We find the best efficiency/performance
balance in the 32 channels embedding (DarkIR-m).

In the supplementary material, we present additional
studies on the development of the architecture.

Input
LEDNet
Retinexformer
Nafnet
Restormer
Ours-m
Ours-l

> **Figure 7. Qualitative comparison on real night scenes from the RealBlurLOL dataset. (Zoom in for best view).**

Figure 7. Qualitative comparison on real night scenes from the RealBlurLOL dataset. (Zoom in for best view).

**Table 6.** Network blocks ablation study. Combination of EBlocks and DBlocks achieves the best performance.

| Network Block Configuration | Params (M) ↓ | MACs ↓ (G) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| EBlock replaced by NAFBlock | 3.12 | 7.69 | 26.51 | 0.8764 | 0.169 |
| EBlock with Biến đổi Pha | 4.08 | 8.38 | 26.30 | 0.8710 | 0.179 |
| All dùng DBlock | 3.20 | 8.19 | 26.68 | 0.8760 | 0.170 |
| All dùng EBlock | 3.44 | 6.46 | 26.17 | 0.8630 | 0.187 |
| All dùng NAFBlock [7] | 3.04 | 7.29 | 26.24 | 0.8680 | 0.178 |
| DBlock replaced by NAFBlock | 3.24 | 6.84 | 26.23 | 0.8640 | 0.185 |
| DBlock không có Depthwise phụ | 3.27 | 6.99 | 26.63 | 0.8750 | 0.174 |
| **DarkIR-m (Đề xuất)** | **3.31** | **7.25** | **26.90** | **0.8740** | **0.175** |

**Table 7.** Spatial Attention ablation study. We compare LKA (large kernel attention) [17] with our proposed Di-SpAM for the decoder blocks.

| Spatial Attention Mechanism | Params (M) ↓ | MACs ↓ (G) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LKA [17] | 4.06 | 9.14 | 26.45 | 0.876 | 0.172 |
| **Di-SpAM (DarkIR-m)** | **3.31** | **7.25** | **27.00** | **0.883** | **0.162** |

**Table 8.** Ablation study scaling the channel depth dimensions. Adaptation depending on memory or runtime requirements.

| Model Variant (Channels) | Params (M) ↓ | MACs ↓ (G) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DarkIR-s (16 channels) | 0.872 | 2.04 | 26.15 | 0.857 | 0.206 |
| **DarkIR-m (32 channels)** | **3.31** | **7.25** | **27.00** | **0.883** | **0.162** |
| DarkIR-l (64 channels) | 12.96 | 27.19 | 27.30 | 0.898 | 0.137 |

Using EBlock and DBlock we are able to greatly reduce
the number of parameters of the network, getting an outstanding 55% less parameters than LEDNet [65] (previous state-of-the-art) and 88% less than Restormer [57] (second best method). This reduction is also accompanied by
a reduction in the number of operations needed to enhance
the input image. Considering an image of 256px -as previous works [7]-, LEDNet uses 33.74 GMACs and Restormer
141.24 GMACs, while DarkIR only uses 7.25 GMACs –

note that 1 MAC is roughly 2 FLOPs. This means a reduction of 4× the number of operations with the previous
state-of-the-art method and almost 20× to the second best
one.

Therefore, DarkIR, while being state-of-the-art in lowlight deblurring, is also lighter in all aspects, representing
an advancement towards deploying this kind of models on
devices with low computational power.

Although we are able to reduce the computational requirements of the target device for running our model, we have
done this by using depth-wise convolutions, which are not
necessarily optimal in certain GPU architectures, as they
lack of arithmetic intensity [15]. Due to this, the model’s inference times are not drastically and proportionally reduced
with the reduction in operations. In future work, we will
propose new methods that can combine low computational
requirements with notably faster inference times.

We propose a robust model for multi-task low-light enhancement and restoration. Our model, DarkIR, is an efficient and robust neural network that performs denoising,
deblurring and low-light enhancement on dark and night
scenes. DarkIR achieves new state-of-the-art results on the
popular LOLBlur, LOLv2 and Real-LOLBlur datasets, being able to generalize on real-world night blurry images
while being more efficient than previous methods.

## Acknowledgements

This work was supported by Spanish funds through Regional Funding Agency Institute for
Business Competitiveness of Castile Leon (MACS.2 project
“Investigacion en tecnolog ´
´ıas del ambito de la movili- ´
dad autonoma, conectada, segura y sostenible”). The au- ´
thors thank Supercomputing of Castile and Leon (SCAYLE.
Leon, Spain) for assistance with the model training and
GPU resources.

## References

1. A. Abdelhamed, S. Lin, and M. S. Brown. A high-quality denoising dataset for smartphone cameras. In *CVPR*, 2018.
2. M. Abdullah-Al-Wadud, M. H. Kabir, M. A. A. Dewan, and O. Chae. A dynamic histogram equalization for image contrast enhancement. *IEEE Trans. Consumer Electron.*, 2007.
3. Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, and Y. Zhang. Retinexformer: One-stage retinex-based transformer for low-light image enhancement. In *ICCV*, 2023.
4. A. Chakrabarti. A neural approach to blind motion deblurring. In *ECCV*, 2016.
5. L. Chen, J. Zhang, S. Lin, F. Fang, and J. S. Ren. Blind deblurring for saturated images. In *CVPR*, 2021.
6. L. Chen, J. Zhang, J. Pan, S. Lin, F. Fang, and J. S. Ren. Learning a non-blind deblurring network for night blurry images. In *CVPR*, 2021.
7. L. Chen, X. Chu, X. Zhang, and J. Sun. Simple baselines for image restoration. In *ECCV*, 2022.
8. S.-J. Cho, S.-W. Ji, J.-P. Hong, S.-W. Jung, and S.-J. Ko. Rethinking coarse-to-fine approach in single image deblurring. In *ICCV*, 2021.
9. X. Chu, L. Chen, and W. Yu. Nafssr: Stereo image super-resolution using nafnet. In *CVPRW*, 2022.
10. M. V. Conde, S. McDonagh, M. Maggioni, A. Leonardis, and E. Pérez-Pellitero. Model-based image signal processors via learnable dictionaries. In *AAAI*, 2022.
11. M. V. Conde, G. Geigle, and R. Timofte. High-quality image restoration following human instructions. *arXiv:2401.16468*, 2024.
12. M. Delbracio, D. Kelly, M. S. Brown, and P. Milanfar. Mobile computational photography: A tour. *arXiv:2102.09000*, 2021.
13. M. Elad and A. Feuer. Restoration of a single super-resolution image from several blurred, noisy, and undersampled measured images. *IEEE TIP*, 1997.
14. Y. Feng, C. Zhang, P. Wang, P. Wu, Q. Yan, and Y. Zhang. You only need one color space: An efficient network for low-light image enhancement. *arXiv:2402.05809*, 2024.
15. A. Gholami, K. Kwon, B. Wu, Z. Tai, X. Yue, P. Jin, S. Zhao, and K. Keutzer. Squeezenext: Hardware-aware neural network design. In *CVPRW*, 2018.
16. C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong. Zero-reference deep curve estimation for low-light image enhancement. In *CVPR*, 2020.
17. M.-H. Guo, C.-Z. Lu, Z.-N. Liu, M.-M. Cheng, and S.-M. Hu. Visual attention network. *Computational Visual Media*, 2023.
18. X. Guo, Y. Li, and H. Ling. Lime: Low-light image enhancement via illumination map estimation. *IEEE TIP*, 2016.
19. J. Hai, Z. Xuan, R. Yang, Y. Hao, F. Zou, F. Lin, and S. Han. R2rnet: Low-light image enhancement via real-low to real-normal network. *JVCI*, 2023.
20. S. W. Hasinoff. Photon, poisson noise. *Computer Vision, A Reference Guide*, 2014.
21. Z. Hu, S. Cho, J. Wang, and M.-H. Yang. Deblurring low-light images with light streaks. In *CVPR*, 2014.
22. Y. Jiang, X. Gong, D. Liu, Y. Cheng, C. Fang, X. Shen, J. Yang, P. Zhou, and Z. Wang. Enlightengan: Deep light enhancement without paired supervision. *IEEE TIP*, 2021.
23. J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang. Musiq: Multi-scale image quality transformer. In *ICCV*, 2021.
24. L. Kong, J. Dong, J. Ge, M. Li, and J. Pan. Efficient frequency domain-based transformers for high-quality image deblurring. In *CVPR*, 2023.
25. O. Kupyn, T. Martyniuk, J. Wu, and Z. Wang. Deblurgan-v2: Deblurring (orders-of-magnitude) faster and better. In *ICCV*, 2019.
26. E. H. Land. The retinex theory of color vision. *Scientific American*, 1977.
27. B. Li, X. Liu, P. Hu, Z. Wu, J. Lv, and X. Peng. All-in-one image restoration for unknown corruption. In *CVPR*, 2022.
28. C. Li, C.-L. Guo, M. Zhou, Z. Liang, S. Zhou, R. Feng, and C. C. Loy. Embedding fourier for ultra-high-definition low-light image enhancement. *arXiv:2302.11831*, 2023.
29. O. Liba, K. Murthy, Y.-T. Tsai, et al. Handheld mobile photography in very low light. *ACM TOG*, 2019.
30. R. Liu, L. Ma, J. Zhang, X. Fan, and Z. Luo. Retinex-inspired unrolling with cooperative prior architecture search for low-light image enhancement. In *CVPR*, 2021.
31. X. Liu, Z. Wu, A. Li, et al. Ntire 2024 challenge on low light image enhancement: Methods and results. *arXiv:2404.14248*, 2024.
32. W. Luo, Y. Li, R. Urtasun, and R. Zemel. Understanding the effective receptive field in deep convolutional neural networks. In *NeurIPS*, 2016.
33. X. Lv, S. Zhang, C. Wang, Y. Zheng, B. Zhong, C. Li, and L. Nie. Fourier priors-guided diffusion for zero-shot joint low-light enhancement and deblurring. In *CVPR*, 2024.
34. C. Ma, C.-Y. Yang, X. Yang, and M.-H. Yang. Learning a no-reference quality metric for single-image super-resolution. *CVIU*, 2017.
35. L. Ma, T. Ma, R. Liu, X. Fan, and Z. Luo. Toward fast, flexible, and robust low-light image enhancement. In *CVPR*, 2022.
36. A. Mittal, R. Soundararajan, and A. C. Bovik. Making a “completely blind” image quality analyzer. *IEEE SPL*, 2012.
37. S. Nah, T. H. Kim, and K. M. Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In *CVPR*, 2017.
38. S. Nah, T. H. Kim, and K. M. Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In *CVPR*, 2017.
39. V. Potlapalli, S. W. Zamir, S. Khan, and F. S. Khan. Promptir: Prompting for all-in-one blind image restoration. *arXiv:2306.13090*, 2023.
40. J. Rim, H. Lee, J. Won, and S. Cho. Real-world blur dataset for learning and benchmarking deblurring algorithms. In *ECCV*, 2020.
41. G. Seif and D. Androutsos. Edge-based loss function for single image super-resolution. In *ICASSP*, 2018.
42. A. Shutova, E. Ershov, G. Perevozchikov, et al. Ntire 2023 challenge on night photography rendering. In *CVPRW*, 2023.
43. K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. *arXiv:1409.1556*, 2014.
44. C. Wang, H. Wu, and Z. Jin. Fourllie: Boosting low-light image enhancement by fourier frequency information. In *ACM MM*, 2023.
45. Y. Wang, Y. Yu, W. Yang, L. Guo, L.-P. Chau, A. C. Kot, and B. Wen. Exposurediffusion: Learning to expose for low-light image enhancement. In *ICCV*, 2023.
46. Z. Wang, X. Cun, J. Bao, and J. Liu. Uformer: A general u-shaped transformer for image restoration. In *CVPR*, 2022.
47. C. Wei, W. Wang, W. Yang, and J. Liu. Deep retinex decomposition for low-light enhancement. In *BMVC*, 2018.
48. C. Wei, W. Wang, W. Yang, and J. Liu. Deep retinex decomposition for low-light enhancement. *arXiv:1808.04560*, 2018.
49. J. Whang, M. Delbracio, H. Talebi, C. Saharia, A. G. Dimakis, and P. Milanfar. Deblurring via stochastic refinement. In *CVPR*, 2022.
50. J. Xu, Y. Hou, D. Ren, L. Liu, F. Zhu, M. Yu, H. Wang, and L. Shao. Star: A structure and texture aware retinex model. *IEEE TIP*, 2020.
51. K. Xu, X. Yang, B. Yin, and R. W. Lau. Learning to restore low-light images via decomposition-and-enhancement. In *CVPR*, 2020.
52. X. Xu, R. Wang, C.-W. Fu, and J. Jia. Snr-aware low-light image enhancement. In *CVPR*, 2022.
53. S. Yang, M. Ding, Y. Wu, Z. Li, and J. Zhang. Implicit neural representation for cooperative low-light image enhancement. In *ICCV*, 2023.
54. W. Yang, S. Wang, Y. Fang, Y. Wang, and J. Liu. From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement. In *CVPR*, 2020.
55. W. Yang, W. Wang, H. Huang, S. Wang, and J. Liu. Sparse gradient regularized deep retinex network for robust low-light image enhancement. *IEEE TIP*, 2021.
56. W. Yu, M. Luo, P. Zhou, C. Si, Y. Zhou, X. Wang, J. Feng, and S. Yan. Metaformer is actually what you need for vision. In *CVPR*, 2022.
57. S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang. Restormer: Efficient transformer for high-resolution image restoration. In *CVPR*, 2022.
58. S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao. Learning enriched features for fast image restoration and enhancement. *IEEE TPAMI*, 2022.
59. J. Zhang, J. Huang, M. Yao, Z. Yang, H. Yu, M. Zhou, and F. Zhao. Ingredient-oriented multi-degradation learning for image restoration. In *CVPR*, 2023.
60. R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In *CVPR*, 2018.
61. Y. Zhang, X. Guo, J. Ma, W. Liu, and J. Zhang. Beyond brightening low-light images. *IJCV*, 2021.
62. J. Zhao, J. Ma, B. Fang, S. Quan, and F. Hu. Fast motion deblurring using gyroscopes and strong edge prediction. In *ICPR*, 2016.
63. Y. Zhao, Y. Xu, Q. Yan, D. Yang, X. Wang, and L.-M. Po. D2hnet: Joint denoising and deblurring with hierarchical network for robust night image restoration. In *ECCV*, 2022.
64. H. Zhou, W. Dong, X. Liu, S. Liu, X. Min, G. Zhai, and J. Chen. Glare: Low light image enhancement via generative latent feature based codebook retrieval. In *ECCV*, 2025.
65. S. Zhou, C. Li, and C. C. Loy. Lednet: Joint low-light enhancement and deblurring in the dark. In *ECCV*, 2022.
