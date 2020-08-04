<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="img/GICD_LOGO.png" alt="Logo" width="210" height="100">
  </a>

  <h3 align="center">Gradient-Induced Co-Saliency Detection</h3>

  <p align="center">
    Zhao Zhang*, Wenda Jin*, Jun Xu, Ming-Ming Cheng
    <br />
    <a href="http://zhaozhang.net/coca.html"><strong>⭐ Project Home »</strong></a>
    <br />
    <!-- <a href="https://arxiv.org/abs/2004.13364" target="_black">[PDF]</a>
    <a href="#" target="_black">[Code]</a>
    <a href="https://www.bilibili.com/video/BV1y5411a7Rq/" target="_black">[Short Video]</a>
    <a href="https://www.bilibili.com/video/BV1bi4y137c6" target="_black">[Long Video]</a>
    <a href="#" target="_black">[Slides]</a>
    <a href="#" target="_black">[中译版]</a>
    <a href="./papers/20_GICD/bibtex.txt" target="_black">[bib]</a>
    <br />
    <br /> -->
  </p>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2004.13364">
    <img src="https://img.shields.io/badge/PDF-%F0%9F%93%83-green" target="_blank" />
  </a>
  <a href="https://www.bilibili.com/video/BV1y5411a7Rq/">
    <img alt="Bilibili" src="https://img.shields.io/badge/Short%20Video-%F0%9F%8E%A5-orange" target="_blank" />
  </a>
  <a alt="Bilibili" href="https://www.bilibili.com/video/BV1bi4y137c6">
    <img src="https://img.shields.io/badge/Long%20Video-%F0%9F%8E%AC-blue" />
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Slides-%F0%9F%97%92-yellow">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-red">
  </a>
</p>


***
The official repo of the ECCV 2020 paper 
[Gradient-Induced Co-Saliency Detection](https://arxiv.org/abs/2004.13364).

More details can be found at our [project home.](http://zhaozhang.net/coca.html)



## Prerequisites
#### Environments
* PyTorch >= 1.0
* tqdm
#### Pretrained model
Download `gicd_ginet.pth` ([Baidu (05cl)](https://pan.baidu.com/s/1UF3wXY3MKdBLP_r7jppz6Q)/[Google Drive](https://drive.google.com/file/d/1gFA16C9m7GXli0TP501cofw0Bzt7-1CS/view?usp=sharing)).

<!-- USAGE EXAMPLES -->
## Usage
1. Configure the input root and the output root in `test.sh`

``` 
--param_path ./gicd_ginet.pth (pretrained model path)
--input_root your_data_root (categorize by subfolders)
--save_root your_output_root
```

2. Run by
```
sh test.sh
```
## Prediction results
The co-saliency maps of GICD can be found at our [project home.](http://zhaozhang.net/coca.html)

## Citation
If you find this work is useful for your research, please cite our paper:
```
@inproceedings{zhang2020gicd,
 title={Gradient-Induced Co-Saliency Detection},
 author={Zhang, Zhao and Jin, Wenda and Xu, Jun and Cheng, Ming-Ming},
 booktitle={European Conference on Computer Vision (ECCV)},
 year={2020}
}
```

## Contact
If you have any questions, feel free to contact me via `zzhang🥳mail😲nankai😲edu😲cn`