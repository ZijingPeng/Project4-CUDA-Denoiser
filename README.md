CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

- Zijing Peng
- Tested on: Windows 22, i7-8750H@ 2.22GHz 16GB, NVIDIA GeForce GTX 1060 (laptop)

### Summary

In this project, I implemented a pathtracing denoiser that uses geometry buffers (G-buffers) to guide a smoothing filter, base on the paper "[Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)" by Dammertz, Sewtz, Hanika, and Lensch.

The following figure illustrates the basic steps of our algorithm. The input is the path-traced (noisy) image, along with a normal buffer and a position buffer. The algorithm use Gaussian filter with increasing step width for multiple iterations to get a denoised result.

![](/img/gbuffer.png)



### Results and Analysis

#### Denoising iterations

The following figures show the renders with different denoising iterations. Compared with the PT reference,  the result of 5 iterations shows best visual effect among those candidates, it is relatively "smooth", and the shadow looks good. 

| Input                   | PT Reference              |
| ----------------------- | ------------------------- |
| ![](/img/10samples.png) | ![](/img/5000samples.png) |
| **2 iterations**        |    **5 iterations**           |
| ![](/img/denoise2.png)  | ![](/img/denoise5.png)    |
|   **10 iterations**          |    **100 iterations**          |
| ![](/img/denoise10.png) | ![](/img/denoise100.png)  |

According to the profiling below, he denoising time increases approximately linearly with the number of times.

![](/img/perf3.png)

#### Material type

The material type does not affect the denoising time, since this algorithm is "post-processing".

![](/img/perf4.png)

#### Light size

Based on path tracing algorithm, a smaller light source means more samples needed to get a smooth result. Thus, if we take 10 iterations for path tracing (like we do before), the result is much nosier. More denoising iterations you'll add to bring a smooth result. However, although it is smooth, it is kind of dark. Another choice is to do the path tracing for more iterations before doing denoiser (see the last figure).

| Large Light + PT 10 iterations     | Small Light + PT 10 iterations       | Small Light + PT 5000 iterations     |
| ---------------------------------- | ------------------------------------ | ------------------------------------ |
| ![](/img/10samples.png)            | ![](/img/cornell10.png)              | ![](/img/cornell.png)                |
| **Denoising 5 iterations + 10 PT** | **Denoising 100 iterations + 10 PT** | **Denoising 5 iterations + 5000 PT** |
| ![](/img/smalllight5.png)          | ![](/img/smalllight100.png)          | ![](/img/smalllight5000.png)         |
