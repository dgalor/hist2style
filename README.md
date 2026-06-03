<h1 align="center">Hist2Style: Histogram-Guided Stylization with Bilateral Grids</h1>

<p align="center"><strong>CVPR 2026</strong></p>

<p align="center">
  <a href="https://dgalor.github.io/hist2style/">Project Page</a> &nbsp;·&nbsp;
  <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Galor_Hist2Style_Histogram-Guided_Stylization_with_Bilateral_Grids_CVPR_2026_paper.pdf">Paper</a> &nbsp;·&nbsp;
  <a href="https://openaccess.thecvf.com/content/CVPR2026/supplemental/Galor_Hist2Style_Histogram-Guided_Stylization_CVPR_2026_supplemental.zip">Supplementary</a> &nbsp;·&nbsp;
  <a href="https://cvpr.thecvf.com/virtual/2026/poster/39586">Poster</a>
</p>

<p align="center"><em>Source code is undergoing release approval and will be published here soon.</em></p>

---

## Abstract

Photorealistic style transfer aims to match the color and tone of an input image to that of a style target while preserving content and fine details. Existing large image-editing models can perform stylization, but their computational cost, hallucinations, and limited precision control make them impractical for high-resolution, interactive workflows.

**Hist2Style** distills a large editor into a lightweight model constrained to locally affine transformations in bilateral space. The model conditions on a histogram-based style embedding for an interpretable interface to color and tone control, delivering real-time, high-resolution photorealistic stylization without structural drift.

---

<p align="center">
  <a href="https://www.dekelgalor.com/">Dekel Galor</a><sup>1,2</sup>,
  <a href="https://www.linkedin.com/in/adam-pikielny-30a606152/">Adam Pikielny</a><sup>1</sup>,
  <a href="https://ztzhang.github.io/personal_webpage/">Zhoutong Zhang</a><sup>1</sup>,
  <a href="https://people.eecs.berkeley.edu/~kewang/">Ke Wang</a><sup>1</sup>,
  <a href="https://www.laurawaller.com/">Laura Waller</a><sup>2</sup>,
  <a href="https://people.csail.mit.edu/jiawen/">Jiawen Chen</a><sup>1</sup>,
  <a href="https://ilyac.info/">Ilya Chugunov</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Adobe Nextcam &nbsp;&nbsp;&nbsp;
  <sup>2</sup> University of California, Berkeley
</p>

---

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Galor_2026_CVPR,
    author    = {Galor, Dekel and Pikielny, Adam and Zhang, Zhoutong and Wang, Ke and Waller, Laura and Chen, Jiawen and Chugunov, Ilya},
    title     = {Hist2Style: Histogram-Guided Stylization with Bilateral Grids},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {29717-29726}
}
```
