# Learning Optimized Low-Light Image Enhancement for Edge Vision Tasks

**S M A Sharif**, **Azamat Myrzabekov**, **Nodirkhuja Khujaev**, **Roman Tsoy**, **Seongwan Kim**, **Jaeho Lee**$^*$  
Opt-AI Inc., LG Sciencepark, Seoul, South Korea  
Email: {sharif, azamat, nodir, roma, swan.kim, jaeho.lee}@opt-ai.kr | $^*$ Corresponding Author

---

## Abstract

Low-light image enhancement (LLIE) has a significant role in edge vision applications (EVA). Despite its
widespread practicability, the existing LLIE methods are
impractical due to their high computational costs. This
study proposed a framework to learn optimized low-light
image enhancement to tackle the limitations of existing enhancement methods for accelerating EVA. The proposed
framework incorporates a lightweight and mobile-friendly
deep network. We optimized our proposed model with INT8
precision with a post-training quantization strategy and deployed it on an edge device. The LLIE model has achieved
over 199 frames per second (FPS) on a low-power edge
board. Additionally, we evaluated the practicability of an
optimized model for accelerating the vision application of
an edge environment. The experimental results illustrate
that our optimized method can significantly accelerate the
performance of SOTA vision algorithms in challenging lowlight conditions for numerous everyday vision tasks, including object detection and image registration.

## 1. Introduction

LLIE refers to enhancing the quality of images captured
under challenging low-light conditions. Enhancing poorly
lightened images is typically considered among the most
complex image restoration tasks due to the lack of visible details and aggressive sensor noise [1, 38]. Therefore,
enhancing low-light images requires comprehensive reconstruction techniques such as contrast enhancement, noise
reduction, and detail restoration. Apart from its aesthetic
importance, LLIE has numerous real-world applications,
including accelerating generic vision tasks on low-power
edge devices.

LLIE on-edge devices incorporate widespread applications, ranging from surveillance and security systems to
autonomous vehicles and industrial monitoring, often op-

erating in real time and relying heavily on visual information for decision-making. Regrettably, the performance of
edge applications such as object detection (OD), segmentation, stereo vision, etc., when camera sensors suffer from
quantum inefficiency and produce darker images [38]. Improving the quality of such low-quality images can enhance
their ability to detect and recognize objects accurately, even
in challenging lighting conditions. Overall, strengthening
the low-quality images can improve safety and efficiency in
real-world scenarios.

Recent literature has seen a surge in efforts to tackle
LLIE challenges through innovative learning-based approaches [18]. These approaches primarily fall into two
categories. The first category intends to learn LLIE as
an image-to-image translation task [3, 8, 9, 17, 23, 28,
40, 44, 44, 52]. Typically, these methods leverage convolutional neural networks (CNN) for end-to-end learning. Also, a few recent methods from the same category
[6, 51] combined their CNN architectures with transformers [42], similar to the recent image restoration methods
[48, 55, 56]. The second category of LLIE incorporates
the Retinex theory [21] into CNN architecture for enhancing low-light images[4, 41, 47, 49, 50, 54, 57, 59, 60].
However, most existing LLIE methods are developed without considering the feasibility of deploying their novel networks on edge devices. Retinex-based and attention-based
methods’ complex nature incorporates high computational
cost edge unfriendly convolution maneuvers. Consequently,
these SOTA LLIE solutions could be more practical for deploying on-edge devices.

To address the limitation and take LLIE research in a
realistic direction, we propose a novel edge-friendly LLIE
learning and deployment framework. Our proposed framework incorporates a lightweight U-shape deep model [35]
and optimization strategy. We optimized our model by
leveraging Post-training quantization (PTQ)[25] and downgraded precision from the floating weight into INT8 for
faster inference in an edge environment. Later, we deployed
our method to real-edge devices and perceived an FPS of
199. The practicability of such a low-precision LLIE model

*Corresponding author

(b)

(d)

> **Figure 1. Real-world OD enhancement leveraging proposed optimization framework. (a) Low-light image with Yolo-V3. (b) Our model (INT8) with Yolo-V3. (c) Our model (full precision) with Yolo-V3. (d) Ground-truth (for detection).**

Figure 1. Real-world OD enhancement leveraging proposed optimization framework. (a) Low-light image with Yolo-V3. (b) Our model
(INT8) with Yolo-V3. (c) Our model (full precision) with Yolo-V3. (d) Ground-truth (for detection).

> **Figure 1. Real-world OD enhancement leveraging proposed optimization framework. (a) Low-light image with Yolo-V3. (b) Our model (INT8) with Yolo-V3. (c) Our model (full precision) with Yolo-V3. (d) Ground-truth (for detection).**

is extensively evaluated by incorporating it into generic vision tasks (e.g., OD and image registration) [50]. The experimental results illustrate that our optimized model can
significantly improve the performance of such vision applications even in stochastic lighting conditions. To the best of
concern, this is the first open literature exploring quantized
models’ practicability in edge vision tasks. Our framework
can help with everyday vision tasks like OD, even in very
dark scenarios, as shown in Fig. 1. Our contributions are as
follows:

• We propose a LLIE learning framework for edge devices.
Our proposed framework incorporates a lightweight deep
model and deployment strategy in an edge environment.
• We optimized our model and achieved an FPS of 299 on
real-edge hardware by sacrificing minor fidelity scores.
• We incorporate our optimized model for accelerating the
performance of generic vision tasks.

This section briefly describes the related works, including
methods for LLIE and Quantization methods on image enhancement.

Researchers have developed various algorithms for lowlight image enhancement to improve the quality of images captured in low, insufficient-light environments. The
most straightforward approach used for this task is a Histogram equalization-based approach[15, 27]. These methods are computationally effective and only sometimes produce well-defined results. Meanwhile, Retinex-based [21]
enhancement algorithms work much better due to leveraging the human visual system’s ability to perceive colors and
brightness relative to surrounding regions, resulting in visually pleasing enhancements. With the development of deep
learning methods, current deep learning methods were inspired by Retinex theory, such as Retinex-Net [49], which
shows promising results for image enhancement. Besides
CNN models, transformer architecture was used to learn or
adjust the parameters of Retinex and produce promising results. One such model is RetinexFormer[4], where a one-

stage transformer was used to learn illumination information from low-light images. Moreover, several diffusionbased approaches, such as Diff-Retinex[54], were made,
where generative diffusion models were used to learn lowlight image features. Although all these methods show excellent results on low-light image enhancement tasks, they
need more interpretability. Notably, the complex nature of
this SOTA method makes them edge unfriendly and impractical for edge deployment.

Model quantization is a crucial step for deploying networks
on mobile devices. Unfortunately, only a few recent methods from the image enhancement domain leverage the capability of quantization for edge optimization. Among them is
a recent method, Fully Quantized Image Super-Resolution
Framework (FQSR)[46], which aims at obtaining end-toend quantized models for all layers, mainly including skip
connections. PAMS [22] applied layer-wise symmetric
quantization to extract high-level features. A channel-wise
distribution-aware quantization scheme for image enhancement was introduced by DAQ[12]. In contrast, DDTB[61]
introduced the design of an asymmetric activation quantizer,
which compresses to 2-4 bit by introducing dynamic dual
trainable clip values. Apart from that, a recent method applied PTQ for Image Super Resolution[45], authors introduce the density-based dual clipping to cut off the outliers
by analyzing the asymmetric bounds of activations.

It is worth noting that the existing quantitative enhancement methods have yet to explore the challenges and practicability of quantization on LLIE. However, a few novel
quantized methods [2, 7, 29] are introduced for image
restoration, like super-resolution. In contrast, such restoration tasks have different characteristics from those of the
LLIE. Image enhancement techniques like super-resolution
focus on improving the details, whereas LLIE incorporates
more challenging tasks due to low visibility and missing details in the darker regions. In addition to that, these methods
mainly conclude their study without exploiting real-world
edge hardware. Thus, the practicability of quantization on
image enhancement, particularly for LLIE, is still undis-

closed. This study explored the practicability of quantized
LLIE on EVA utilizing actual edge hardware.

This section describes the proposed LLIE edge optimization framework and its learning strategies. Fig. 2 illustrates
the overview of the proposed framework. We develop a
lightweight deep model with hardware-friendly vanilla convolution operation. We quantized our trained network into
INT8 precision for faster inference. Later, we deployed our
network on a real-edge device to evaluate its practicability
on real-world edge vision applications (EVA).

We consider LLIE for EVA to be an image-to-image translation task. Thus, it aims to map a low-light image $( I _ { L } )$ to
a perceptually enhanced image $( I _ { H } )$ using a mapping function M : $I _ { L } \, \rightarrow \, I _ { H } . \,$ Here, $I _ { L }$ is constrained to the range
$[ 0 , 1 ] ^ { H \times W \times 3 } ,$ where $H$ and $W$ denote the height and width
of both input and output images, respectively. Fig. 2 provides an overview of the proposed deep model architecture.

The proposed deep network is structured as a fully convolutional encoder-decoder architecture [5, 10], featuring convolutional skip connections. The initial layer of the generator transforms the input image $( I _ { L } )$ into a 64-depth feature
map. This input convolutional layer employs a kernel size
of 3×3, padding of 1, and stride of 1. Following this, an encoder consists of four consecutive feature levels, with alternating feature depths of $d = 6 4 , 9 6 , 1 2 8 , 1 9 6 .$ Each feature
level integrates a residual block, which processes an input
feature X using the equation:

$F ^ { \prime } = C ( X ) + X \qquad \qquad$

$\mathrm { ~ ( 1 ) }$

Here, C(·) comprises two convolution layers, each with
a kernel size of $3 \times$ 3, padding of 1, and stride of 1, with
the first layer activated using PReLU activation. The output
feature dimension of a residual block remains unchanged
from its input (i.e., 64 in the initial feature level). Notably,
the residual blocks are well-known for achieving better denoising performance. Such architectural choices help us in
handling sensor noise in low-light conditions.

Following each residual block in the encoder, a convolutional downsampling layer is applied as follows:

Here, $C _ { \downarrow }$ represents a $3 \times 3$ convolution operation with a
stride of 2.

The decoder section of the network mirrors the encoder
in terms of the number of feature levels, with an upsampling layer following each residual block. The upsampling
operation is implemented as follows:

Here, $~ F _ { \uparrow } ~$ involves $\mathrm { ~ a ~ }$ transpose ${ }$ convolution operation
[38]. The transpose convolution helps us make our model
fully convolutional with effective restoration performance.

In addition to the basic indoor-decoder design, we incorporated a gated skip connection to propagate refined lowlevel features. It is worth noting that the U-shaped encoder
encodes the noise along with salient information of lowlight images. We found that propagating such sensor noises
in the decoder pushes the deep model to achieve desired enhanced images [39]. Therefore, we leverage a convolution
residual-gating mechanism as follows:

Here, $\, C _ { 1 \times 1 }$ represents point-wise convolution that re-

fines the spatial information of a given tensor $X _ { R } .$

Finally, the decoder portion culminates in a final convolutional layer, which produces the three-channel enhanced
image using a convolutional kernel size of $3 \times 3 ,$ padding of
1, and a stride of 1. This final output layer is activated with
a tanh function to yield the final images within the range of
[0, 1].

We leverage a pixel-wise reconstruction loss to guide our
deep model as a coarse-to-refine reconstruction process. An
L1 or L2 distance typically serves as the pixel-wise loss
function. While both options are standard, the L2-loss is
directly related to the Peak Signal-to-Noise Ratio (PSNR)
metric and tends to yield smoother images [20, 38]. Notably, low-light images comprise a substantial sensor noise.
Thus, we utilized an L1 objective function as the reconstruction loss. Our reconstruction loss can be represented
as follows:

Here, $\textit { I } _ { H }$ represents $\mathrm { ~ t h e ~ }$ output $\mathrm { ~ o b t a i n e d ~ }$ through $\mathrm { ~ M , ~ }$
while $I _ { G }$ denotes the reference high-light image. This loss
function quantifies the absolute differences between corresponding pixels in the output and reference images, thereby
driving the network to minimize these differences during
training.

PTQ is a technique used to reduce the memory footprint and improve the inference speed of deep neural

> **Figure 2. Overview of the proposed LLIE edge optimization framework. Our framework incorporates a lightweight deep model, a PTQ, and a real-hardware deployment strategy for faster inference speed in an edge environment. We deployed our model on actual low-power hardware to find its practicability in the later phase.**

Figure 2. Overview of the proposed LLIE edge optimization framework. Our framework incorporates a lightweight deep model, a PTQ,
and a real-hardware deployment strategy for faster inference speed in an edge environment. We deployed our model on actual low-power
hardware to find its practicability in the later phase.

networks after training [25, 45]. It involves converting
the network’s weights and activations from high-precision
floating-point representations (e.g., 32-bit floating point) to
lower-precision fixed-point representations (e.g., 8-bit integers). This process significantly reduces the network’s
memory requirements and computational complexity, making it more efficient to deploy on resource-constrained devices such as edge devices. It is worth noting that PTQ can
achieve even lower bit rates for faster optimization. However, we found that most edge boards available for realworld use (e.g., autonomous driving and security) support
8-bit inference [62]. Therefore, considering the actual usage, we quantized our model into INT8 precision for realedge deployment.

Model calibration is crucial for determining appropriate
scaling factors for lower bit precision when performing
PTQ on deep networks. The calibration process is typically perceived using numerous techniques (e.g., histogram
and max calibration) [14]. We found the histogram calibration process fits our application well. Notably, the low-light
images often have a compressed histogram towards darker
values. Thus, such a calibration process helps us expand
the distribution, revealing details obscured in the shadows
by brightening them proportionally.

Here, we obtained histogram calibration $Q _ { h }$ with maximum histogram values (Mh) and maximum quantization
value (Qh). We tailor the quantization process to the specific distribution of activation values encountered during
inference. This meticulous calibration technique ensures
that the quantized model maintains accuracy and performance levels commensurate with the original floating-point

model, thus facilitating seamless deployment on resourceconstrained devices.

INT8 quantization for neural networks involves scaling and
rounding the original floating-point values (like weights) to
a lower precision integer format (typically 8 bits for INT8).
This is achieved by considering the minimum and maximum values of the original data and then rescaling the data
to a range suitable for the 8-bit integer representation. We
perceive INT8 quantization as follows:

Here, R presents The round function that ensures the
rounded to the nearest integer. P is presents the original
value to be quantized, and S is the scaling factor determined
during calibration.

We leverage three low-light enhancement datasets for learning real-world LLIE, including a dataset from the New
Trends in Image Restoration and Enhancement’24 (NTIRE)
low-light enhancement challenge [24], LOL-v1 [49], and
LOL-v2 (real data) [53]. NTIRE challenge provides 230
large-dimension images for training purposes. We extracted
4,500 non-overlapping patches from the dataset and combined them with LOL-v1 (485 training samples) and LOL-
(6) v2 (689 training samples). In the testing phase, we leverage LOL-V1 and LOL-v2 for quantatitive evaluation and
the NTIRE dataset of qualitative assessment.

The proposed u-shape lightweight model has been implemented with the PyTorch framework [33] and optimized
with the Pytorch Quantization library [31]. We optimized
our method in the learning phase with an Adam optimizer
[19]. We tuned its hyperparameters as β1 = 0.9, β2 =
0.99. Initially, we set the learning rate for the model as

$l r = 1 e - 4 .$ We trained our LLIE method for 50,000 steps
with a batch size 64. For the training phase, we utilized
image patches of dimension $1 2 8 \times 1 2 8 \times$ 3. The machine
learning phase took around 6 hours, comprising an NVIDIA
A6000 GPU with 32GB RAM and a 16-core CPU. Later, we
deployed our model on the NVIDIA Jetson Orin edge board
for real-world evaluation.

This section compares the proposed deep models with
SOTA lightweight deep models, real-world quantized performance, and performance on the acceleration of edge vision tasks.

We compared the performance of lightweight mobilefriendly deep image enhancement networks, including Unet
[35], Mobile Unet [13], and DnCNN [16]. In addition to
image enhancement methods, we also compared our model
with the LLIE methods like [11] and Retinexnet [49]. We
trained and tested all comparing methods under the same
data samples for a fair comparison. We summarised the
performance of deep methods with a peak signal-to-noise
ratio (PSNR) and structural similarity index (SSIM) [37]
and perceptual image patch similarity (LPIPS) [58]

Table. 1 illustrates the performance of the deep models on
LOL-v1 and LOL-v2 testing sets. The proposed method
can outperform the existing methods despite its dedicated
nature for the edge device. Particularly in a diverse dataset
like LOL-v2, the proposed method illustrates notable domination over its counterparts. In contrast, Retinexnet [49]
can achieve a higher fidelity score in LOL-v1 despite illustrating significant perceptual degradation. It is worth noting
that Retinexnet has been developed to perform well, particularly on the LOL-v1 dataset [49].

Fig. 3 further confirms the practicability of the proposed
method in real-world LLIE. The proposed method can handle complex real-world low-light scenes compared to its
counterparts. The typical mobile-friendly deep architecture
like mobile Unet illustrates significant deficiencies in addressing real-world low-light scenes. Such architectures’
limitations confirm the practicability of an efficient and effective lightweight LLIE method for real-world applications.

We selected our best pre-trained weight for quantization
and deployment. Infer efficiently on edge devices. Apart
from our quantization strategy, we converted our quantized
weight with TensorRT optimization library [30] to directly
deploy in the edge board. We leverage an NVIDIA Jetson
Orin board to deploy and test our proposed LLIE method.
We leveraged Docker to enable the utilization of CUDA
cores for GPU acceleration of the target edge board. We
deployed the optimized weight on our Docker environment
and operated the hardware in its efficient mode (30 watts)
[32] to ensure the practicability of our proposed method on
edge platforms.

Table. 2 illustrates the performance of the proposed optimization framework on different hardware platforms. The
proposed framework allowed us to infer speed 8x compared
to the full precision model. It achieved over 199 FPS on
Jetson Orin with an image dimension of $2 5 6 \times$ 256 $\times$ 3.
Notably, image dimension on inferencing edge devices can
affect the perceptual quality. Overall, it makes a trade-off
between FPS and fidelity score for LLIE on edge devices.

> **Figure 4. Visual comparison between full-precision LLIE model and its quantized version. In most cases, the quantized version can produce perceptually plausible images similar to its counterparts with a significant acceleration in inference speed.**

Apart from the quantatitive evaluation, we also perform a
subjective comparison between the full-precision and quantized models. In our assessment, the PTQ can produce
similar visual results in most cases compared to the fullprecision variants, as shown in Fig. 4.

EVA typically refers to computer vision tasks performed
directly on edge devices, such as smartphones, IoT devices, or embedded systems, without relying heavily on
cloud computing or external servers. Such vision computing approaches are becoming popular due to their ability to
perform real-time visual data processing directly on edge
devices, mitigating privacy concerns and reducing latency.
Despite countless possibilities, developing efficient and accurate vision solutions for low-power devices is among the
most complicated tasks. In this study, we aim to improve
the performance of such vision tasks by incorporating our
proposed LLIE.

Among existing EVA, OD is among the most widely used
applications of edge vision. It allows real-time identification and localization of objects within images or video
streams directly on the device. This technology has countless real-world uses, such as autonomous vehicles for identifying pedestrians and obstacles and intelligent retail systems for tracking inventory and customer behavior. Unfortunately, this widely used application also suffers in lowlight conditions due to missing visible information.

LOL-v1
LOLv-2
Average
Method
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
PSNR
SSIM
LPIPS
Unet [35]
15.37
0.6326
$0 . 4 0 0 7$
17.22
0.6516
0.3718
16.30
0.6421
0.3863
Retinexnet [49]
17.63
0.7185
0.4051
17.71
0.6049
0.4403
17.67
0.6617
0.4227
LIME [11]
16.28
0.6278
0.4315
17.52
0.5674
0.4115
16.90
0.5976
0.4215
15.80
0.6508
0.4047
18.54
0.6994
0.3668
17.17
0.6751
0.3857
MobileUnet [13]
DnCNN [16]
16.18
0.6839
0.4209
18.91
0.7063
0.3909
17.54
0.6951
0.4059
Ours
16.44
0.7006
0.3499
19.04
0.7558
0.3190
17.74
0.7282
0.3344

**Table 1. Qunatatitive comparison between deep methods for LLIE. Overall, the proposed network outperforms its counterpart in all
evaluation metrics.

Unet
DnCNN
Mobile Unet
Input
R**

| Method (Method) | LOL-v1 PSNR | SSIM | LPIPS | LOL-v2 PSNR | SSIM | LPIPS | Average PSNR | SSIM | LPIPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UNet [35] | 15.37 | 0.6326 | $0 . 4 0 0 7$ | 17.22 | 0.6516 | 0.3718 | 16.30 | 0.6421 | 0.3863 |
| RetinexNet [49] | 17.63 | 0.7185 | 0.4051 | 17.71 | 0.6049 | 0.4403 | 17.67 | 0.6617 | 0.4227 |
| LIME [11] | 16.28 | 0.6278 | 0.4315 | 17.52 | 0.5674 | 0.4115 | 16.90 | 0.5976 | 0.4215 |
| Mobile UNet [13] | 15.80 | 0.6508 | 0.4047 | 18.54 | 0.6994 | 0.3668 | 17.17 | 0.6751 | 0.3857 |
| DnCNN [16] | 16.18 | 0.6839 | 0.4209 | 18.91 | 0.7063 | 0.3909 | 17.54 | 0.6951 | 0.4059 |
| **Mô hình của chúng tôi (Ours)** | **16.44** | **0.7006** | **0.3499** | **19.04** | **0.7558** | **0.3190** | **17.74** | **0.7282** | **0.3344** |

**Table 2. Performance of the proposed LLIE method numerous

hardware platforms. The proposed method achieved 199 FPS on

edge devices by utilizing PTQ.**

| Device | Dimension | Precision | PSNR | SSIM | LPIPS | Params (M) | Time (ms) | Speed (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU (NVIDIA RTX A6000) | $5 1 2 \times 5 1 2 \times 3$ | Full (FP32) | 17.738 | 0.7282 | 0.3344 | 4.46 | 9.33 | 164.50 |
| ARM64 (Jetson Orin 30W) | $5 1 2 \times 5 1 2 \times 3$ | Full (FP32) | 17.738 | 0.7282 | 0.3344 | 4.46 | 140.87 | 7.09 |
| ARM64 (Jetson Orin 30W) | $5 1 2 \times 5 1 2 \times 3$ | INT8 | 16.340 | 0.6325 | 0.3506 | 4.46 | 18.37 | 54.42 |
| GPU (NVIDIA RTX A6000) | $2 5 6 \times 2 5 6 \times 3$ | Full (FP32) | 17.310 | 0.6719 | 0.2169 | 4.46 | 4.40 | 227.14 |
| ARM64 (Jetson Orin 30W) | $2 5 6 \times 2 5 6 \times 3$ | Full (FP32) | 17.310 | 0.6719 | 0.2169 | 4.46 | 39.39 | 25.39 |
| **ARM64 (Jetson Orin 30W)** | **$2 5 6 \times 2 5 6 \times 3$** | **INT8** | **15.920** | **0.5457** | **0.2403** | **4.46** | **5.01** | **199.67** |

### 4.4. Image Registration with SuperGlue

Image registration is a fundamental process in computer
vision that involves aligning multiple images of the same
scene or object from different viewpoints, angles, or times.
This alignment facilitates various applications across diverse domains, including autonomous vehicles, augmented
reality, industrial inspection, healthcare, and environmental monitoring. However, image registration can face challenges in low-light conditions, where images may suffer
from reduced visibility, noise, or distortion. In this section, we explore the concept of image registration, its wideranging applications, and the specific challenges encountered when performing registration in low-light environments, focusing on its feasibility in autonomous driving scenarios.

We incorporated our proposed method for improving image registration in an edge environment. Therefore, we used
the SOTA Superglue network [36] on real-world low-light
stereo image datasets to perform registration and matching
in the edge platform. Due to the lack of visibility, Superglue
illustrates a significant deficiency in finding perfect matches
on the image pairs. However, our proposed method can improve its performance by significantly enhancing the darker
images. Table. 4 illustrates the performance of Superglue
on matching-in-the-dark (MID) dataset [43]. Our proposed
method significantly improves the performance of the Superglue network.
Without LLIE, the Superglue network

struggles to identify the key features and perfect match on
the stereo images. Our proposed method helps it by improving visibility to achieve more accurate findings while mitigating false matches. On average, our model improves the
key point detection on both left and right images. The proposed method helps the Superglue improve its performance
by over 90 matches per pair.

Fig. 6 visually confirms the impact of the proposed
LLIE on image matching. Our models substantially improve the performance of the SUperglue network. These
help the image-matching algorithm minimize false matching and find perfect matches among stereo images.

It is important to note that most algorithms used for
matching images are designed to work with grayscale images to simplify the computational complexity of the matching process. Our INT8 model has been optimized for image
matching with quantized weights and can perform with the
same efficiency level as its full-precision counterpart. The
high-performance levels achieved by our model in image
matching with quantized weights indicate that it is wellsuited for EVA. Using a quantized model for image matching in EVA applications can significantly reduce computational and memory requirements while maintaining the
matching process’s accuracy and reliability. Hence, our
INT8 model emerges as an optimal (EVA) solution, offering
efficient and precise image-matching capabilities.

This study reveals a new dimension of LLIE on EVA. Our
optimization framework can significantly improve the performance of EVA by adding minor computational overhead.
The low computational cost of our proposed network with
post-training quantization helps our model achieve an efficient inference speed on edge devices with a decent fertility score. We tested our model on the NTIRE’24 lowlight challenge. Despite being significantly lightweight, our
model can be on par with the SOTA image restoration models like Uformer, MIRNet, etc. However, the SOTA LLIE
method can achieve a better fidelity score than the proposed
method. However, complex blocks like transformers make
that model impractical for edge devices due to their high

Class Name
Bottle
Bus
Car
$\mathbf { A l l }$
Bicycle
Boat
Cat
Chair
Cup
Dog
Bike
People
Table
ExDark
0.3930
0.1890
0.2570
0.6520
0.4210
0.4140
0.3150
0.2720
0.3960
0.3730
0.3960
0.0152
0.3410
INT8
0.4230
0.2050
0.2300
0.6520
0.4300
0.4180
0.3180
0.2790
0.3960
0.4210
0.4120
0.0150
0.3500
Full Precision
0.4240
0.2030
0.2350
0.6590
0.4330
0.4280
0.3270
0.2800
0.4050
0.4320
0.4150
0.0149
0.3550

**Table 3. Our proposed method accelerates the SOTA object detector in low-light conditions. Despite being significantly faster, our
quantized LLIE can also improve the performance of OD in challenging lighting conditions.**

| Class | ExDark (Ảnh gốc) | INT8 (Ours) | Full Precision (FP32) |
| :--- | :---: | :---: | :---: |
| Bicycle | 0.3930 | 0.4230 | 0.4240 |
| Boat | 0.1890 | 0.2050 | 0.2030 |
| Chai lọ (Bottle) | 0.2570 | 0.2300 | 0.2350 |
| Bus | 0.6520 | 0.6520 | 0.6590 |
| Car | 0.4210 | 0.4300 | 0.4330 |
| Cat | 0.4140 | 0.4180 | 0.4280 |
| Chair | 0.3150 | 0.3180 | 0.3270 |
| Cốc (Cup) | 0.2720 | 0.2790 | 0.2800 |
| Dog | 0.3960 | 0.3960 | 0.4050 |
| Xe máy (Motorbike/Bike) | 0.3730 | 0.4210 | 0.4320 |
| People | 0.3960 | 0.4120 | 0.4150 |
| Bàn (Table) | 0.0152 | 0.0150 | 0.0149 |
| **Average $\mathbf { A l l }$** | **0.3410** | **0.3500** | **0.3550** |

****

| Method | Metric | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Pair 6 | Pair 7 | Pair 8 | Pair 9 | Average |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MID [43] (Original)** | Keypoints (L : R) | 536 : 321 | 459 : 371 | 742 : 664 | 519 : 433 | 335 : 311 | 609 : 494 | 531 : 477 | 574 : 461 | 646 : 653 | 550 : 465 |
| | Pair khớp (Matches) | 123 | 204 | 297 | 164 | 191 | 212 | 242 | 195 | 333 | 217.89 |
| | Tỉ lệ Khớp / Tổng | 0.1435 | 0.2458 | 0.2112 | 0.1723 | 0.2957 | 0.1922 | 0.2401 | 0.1884 | 0.2564 | 0.2162 |
| **INT8 (Đề xuất)** | Keypoints (L : R) | 655 : 488 | 599 : 666 | 910 : 966 | 764 : 823 | 434 : 397 | 705 : 741 | 744 : 643 | 587 : 563 | 985 : 1024 | 709 : 701 |
| | Pair khớp (Matches) | 166 | 297 | 407 | 317 | $2 3 1$ | 302 | 305 | 213 | 506 | 304.89 |
| | Tỉ lệ Khớp / Tổng | 0.1452 | 0.2348 | 0.2170 | 0.1997 | 0.2780 | 0.2089 | 0.2199 | 0.1852 | 0.2519 | 0.2156 |
| **Full (FP32)** | Keypoints (L : R) | 655 : 488 | 599 : 666 | 910 : 966 | 764 : 823 | 434 : 397 | 705 : 741 | 744 : 643 | 587 : 568 | 985 : 1024 | 709 : 702 |
| | Pair khớp (Matches) | 166 | 297 | 407 | 317 | $2 3 1$ | 302 | 305 | 213 | 506 | 304.89 |
| | Tỉ lệ Khớp / Tổng | 0.1452 | 0.2348 | 0.2170 | 0.1997 | 0.2780 | 0.2089 | 0.2199 | 0.1844 | 0.2519 | 0.2155 |

complexity and unsupported learning mechanisms.

Our model can show visual artifacts caused by aggressive sensor noise in extremely low-light situations. However, it’s important to note that the number of real-world
training samples we could access was limited. Despite combining three different single-shot low-light image datasets,
our total number of training samples was insufficient for
handling real-world extreme low-light scenarios. Additionally, the images used in our study were taken with professional cameras, which have imaging characteristics different from those of compact camera sensors used in edge
environments like autonomous driving. Thus, it would be
beneficial to create a new LLIE dataset captured with edge
camera sensors to improve the performance of future edge
LLIE methods.

Besides the data limitation, our quantized model tends
to clip color and contrast due to INT8 conversion. Consequently, the model can illustrate on-par performance
(e.g., for image matching) with a full-precision model in
grayscale images. However, due to the range clipping in
RGB space, the quantized model can illustrate performance

downgrade (e.g., object detection) in such instances. Notably, the clipping range issues are a very classical problem
of PTQ. Therefore, we planned to study the effectiveness of
perceptual quantization-aware training for enhancing LLIE
in a future study.

This study proposes a novel framework for learning
optimized LLIE for EVA. Our method incorporates a
lightweight deep model, an optimization strategy (e.g.,
PTQ), and deployment on edge hardware. We compared
our proposed method with SOTA LLIE and outperformed
them in quantatitive and qualitative comparison. Later, we
deployed our method edge device to achieve an FPS of 199.
We also illustrated the practicability of such optimized on
generic vision tasks. The experimental results demonstrate
that the proposed method can significantly improve the performance of SOTA vision algorithms like YOLOv3 and Superglue networks. We planned to collect a real-world LLIE
dataset for edge environment and study the quantizationaware training in the foreseeable future.

## References

[1] SM A Sharif, Rizwan Ali Naqvi, and Mithun Biswas. Beyond joint demosaicking and denoising: An image processing pipeline for a pixel-bin image sensor. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 233–242, 2021.

[2] Haotong Qin, Yifu Ding, Mingyuan Zhang, Qinghua Yan, Aishan Liu, Qingqing Dang, Ziwei Liu, and Xianglong Liu. Bibit-srt: Bi-directional bit-level symmetric refinement for low-bit super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5841–5850, 2023.

[3] Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. Learning to see in the dark. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3291–3300, 2018.

[4] Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, and Yulun Zhang. Retinexformer: One-stage retinex-based transformer for low-light image enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12504–12513, 2023.

[5] SM A Sharif, Rizwan Ali Naqvi, and Mithun Biswas. Deep learning for extreme low-light image enhancement: An end-to-end modified unet approach. IEEE Access, 8:147775–147787, 2020.

[6] Feifan Shao, Yanyun Qu, Sibei Yang, Junchi Yan, and Yuan Xie. Transformer-based low-light image enhancement. IEEE Transactions on Circuits and Systems for Video Technology, 33(8):3745–3758, 2023.

[7] Haotong Qin, Ruihao Gong, Xianglong Liu, Xiao Bai, Jingkuan Song, and Nicu Sebe. Forward and backward information retention for accurate binary neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2250–2259, 2020.

[8] Kin Gwn Lore, Adedotun Akintayo, and Soumik Sarkar. Llnet: A deep autoencoder approach to natural low-light image enhancement. Pattern Recognition, 61:650–662, 2017.

[9] Ruixing Wang, Qing Zhang, Chi-Wing Fu, Xiaoyong Shen, Wei-Shi Zheng, and Jiaya Jia. Underexposed photo enhancement using deep illumination estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6849–6857, 2019.

[10] SM A Sharif, Rizwan Ali Naqvi, and Mithun Biswas. Llsenet: A light-weight fully convolutional network for low-light image enhancement. In 2020 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia), pages 1–4. IEEE, 2020.

[11] Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light image enhancement via illumination map estimation. IEEE Transactions on image processing, 26(2):982–993, 2016.

[12] Cheeun Hong, Heewon Kim, Sungjoo Yoo, et al. Daq: Channel-wise distribution-aware quantization for deep image super-resolution networks. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2675–2684, 2022.

[13] S M A Sharif and Mithun Biswas. Mobile-unet: An efficient convolutional neural network for image enhancement on mobile devices. In 2022 IEEE International Conference on Image Processing (ICIP), pages 1761–1765. IEEE, 2022.

[14] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4510–4520, 2018.

[15] Stephen M Pizer, E Philip Amburn, John D Austin, Robert Cromartie, Ari Geselowitz, Trey Greer, Bart ter Haar Romeny, John B Zimmerman, and Karel Zuiderveld. Adaptive histogram equalization and its variations. Computer vision, graphics, and image processing, 39(3):355–368, 1987.

[16] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7):3142–3155, 2017.

[17] Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, and Zhangyang Wang. Enlightengan: Deep light enhancement without paired supervision. IEEE Transactions on Image Processing, 30:2340–2349, 2021.

[18] Chongyi Li, Chunle Guo, Linghao Han, Junxu Jiang, Ming-Ming Cheng, Jiange Gu, and Chen Change Loy. Low-light image and video enhancement using deep learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12):9396–9416, 2021.

[19] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[20] H. Zhao, O. Gallo, I. Frosio, and J. Kautz. Loss functions for image restoration with neural networks. IEEE Transactions on Computational Imaging, 3(1):47–57, 2017.

[21] Edwin H Land. The retinex theory of color vision. Scientific American, 237(6):108–129, 1977.

[22] Huixia Li, Chenqian Yan, Shaohui Lin, Xiawu Zheng, Baochang Zhang, Fan Yang, and Rongrong Ji. Pams: Prequantization activation margin saver for trained deep neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10560–10569, 2020.

[23] Ke Xu, Xin Yang, Baocai Yin, and Rynson WH Lau. Learning to restore low-light images via decomposition-and-enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2281–2290, 2020.

[24] Xiaoning Liu, Zongwei Wu, Ao Li, Florin-Alexandru Vasluianu, Yulun Zhang, Shuhang Gu, Le Zhang, Ce Zhu, Radu Timofte, et al. Ntire 2024 challenge on low light image enhancement: Methods and results. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2024.

[25] Raghuraman Krishnamoorthi. Quantizing deep convolutional networks for efficient inference: A whitepaper. arXiv preprint arXiv:1806.08342, 2018.

[26] Yuen Peng Loh and Chee Seng Chan. Getting to know low-light images with the exclusively dark dataset. Computer Vision and Image Understanding, 178:30–42, 2019.

[27] Soong-Der Chen and Abd Rahman Ramli. Minimum mean brightness error bi-histogram equalization in contrast enhancement. IEEE transactions on Consumer Electronics, 49(4):1310–1319, 2003.

[28] Long Ma, Tengyu Ma, Risheng Liu, Xin Fan, and Zhongxuan Luo. Toward fast, flexible, and robust low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5637–5646, 2022.

[29] Zhihao Qu, Zhenda Xu, Tao Guo, Tao Lin, Qihua Zhou, and Jingcai Guo. Dcb: Dynamic convolution with binary weights for efficient image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5851–5860, 2023.

[30] NVIDIA TensorRT. High performance deep learning inference. https://developer.nvidia.com/tensorrt.

[31] PyTorch. PyTorch Quantization Library. https://pytorch.org/docs/stable/quantization.html.

[32] NVIDIA. Jetson AGX Orin Series. https://developer.nvidia.com/embedded/jetson-modules.

[33] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

[34] Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018.

[35] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.

[36] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4938–4947, 2020.

[37] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

[38] SM A Sharif, Rizwan Ali Naqvi, and Mithun Biswas. Learning to enhance extreme low-light images for camera sensors. IEEE Access, 9:163777–163789, 2021.

[39] SM A Sharif, Rizwan Ali Naqvi, and Mithun Biswas. Ultra-high-definition low-light image enhancement: A benchmark and a baseline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 849–858, 2022.

[40] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. Zero-reference deep curve estimation for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1780–1789, 2020.

[41] Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, and Jiaying Liu. Sparse gradient regularized deep retinex network for robust low-light image enhancement. IEEE Transactions on Image Processing, 30:2072–2086, 2021.

[42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[43] Chengyuan Wang, Jinyuan Liu, Zhiying Jiang, Long Ma, Xin Fan, and Risheng Liu. Matching in the dark: A dataset for feature matching in extreme low-light conditions. In 2023 IEEE International Conference on Image Processing (ICIP), pages 1400–1404. IEEE, 2023.

[44] S M A Sharif, Seongwan Kim, and Jaeho Lee. Attention-guided hierarchical network for extreme low-light image enhancement. In 2023 IEEE International Conference on Image Processing (ICIP), pages 3394–3398. IEEE, 2023.

[45] Yair Schiff, Diego Andrade, Junru Shao, and Phillip Howard. Ptq4sr: Post-training quantization for real-world image super-resolution. arXiv preprint arXiv:2311.16016, 2023.

[46] Fangcheng Zhong, Yuanhao Cai, Jiaming Liu, Kuan Cheng, Zhiwei Xiong, and Haoqian Wang. Fully quantized image super-resolution framework. arXiv preprint arXiv:2211.14480, 2022.

[47] Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. In Proceedings of the British Machine Vision Conference (BMVC), 2018.

[48] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5728–5739, 2022.

[49] Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. In BMVC, 2018.

[50] Risheng Liu, Long Ma, Jiaao Zhang, Xin Fan, and Zhongxuan Luo. Retinex-inspired unrolling with architecture search for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2501–2510, 2021.

[51] Zewei He, Yuhui Quan, and Hui Ji. Low-light image enhancement using cross-domain attention. IEEE Transactions on Image Processing, 32:2880–2892, 2023.

[52] Chongyi Li, Chunle Guo, and Chen Change Loy. Learning to see in the dark with deep recursive band network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2229–2238, 2020.

[53] Jiaying Liu, Dejia Xu, Wenhan Yang, Ming-Hsuan Yang, and Zongming Guo. Benchmarking low-light image enhancement: An expanded dataset and an objective model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(8):2709–2724, 2021.

[54] Hao Gao, Feifan Shao, and Sibei Yang. Diff-retinex: Generative diffusion retinex for low-light image enhancement. arXiv preprint arXiv:2308.06734, 2023.

[55] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 1833–1844, 2021.

[56] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general u-shaped transformer for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17683–17693, 2022.

[57] Yonghua Zhang, Jiawan Zhang, and Xiaojie Guo. Kindling the darkness: A practical low-light image enhancer. In Proceedings of the 27th ACM International Conference on Multimedia, pages 1632–1640, 2019.

[58] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.

[59] Shuhang Gu, Felix Heide, Lei Zhang, and Xin Yuan. Learning deep illumination estimation for low-light image enhancement. IEEE Transactions on Image Processing, 29:7922–7935, 2020.

[60] Xiaojie Guo and Jiawan Zhang. Kind: Kindling the darkness. In Proceedings of the 28th ACM International Conference on Multimedia, pages 4467–4469, 2020.

[61] Haotong Qin, Yifu Ding, Xianglong Liu, Jiakai Wang, Michele Magno, and Qi Tian. Dynamic dual-trainable bounds for ultra-low bit neural network quantization. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2125–2133, 2022.

[62] Qihua Zhou, Song Guo, Zhihao Qu, Jingcai Guo, Zhenda Xu, Jiewei Zhang, Tao Guo, Boyuan Luo, and Jingren Zhou. Octo: {INT8} training with loss-aware compensation and backward quantization for tiny on-device learning. In 2021 USENIX Annual Technical Conference (USENIX ATC 21), pages 177–191, 2021.
