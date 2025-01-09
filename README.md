# Flattening-Net Reproduction

## 环境配置

本项目基于 Docker 容器运行。容器的构建和运行分别依赖于 Dockerfile 和 VSCode 的 Dev Containers 扩展，相关配置文件在 `.devcontainer` 目录下。请参考[相关文档](https://code.visualstudio.com/docs/devcontainers/tutorial)构建容器和启动项目。

## 运行

所有命令的运行都在 flattening 虚拟环境（Dockerfile 中自动创建）下执行。

### 训练

关于 Flattening-Net 网络的训练可以分为两部分：

- G2SD
  - `python src/train_g2sd.py`
- S2PF
  - `python src/train_s2pf.py`

基于 PGI 图像进行下游任务的训练（同时包含测试）运行如下命令：

```bash
# 分类任务
python src/main_cls.py
# 重建任务
python src/main_rec.py
```

> 所有预训练好的模型参数可以从[网盘](https://pan.baidu.com/s/1TB630TOlZufn5SBVAEYZjw?pwd=vaoj)（提取码：vaoj）下载，预训练的模型参数必须放在 `/ckpt` 目录下

### 测试

根据点云生成 PGI 图像可以运行 `python src/dataset.py` 命令。

## 数据集

本项目由于存在多个模块，所以存在多个不同的数据集：

- 训练 Flattening-Net 网络的数据集可以从[项目原址](https://github.com/keeganhk/Flattening-Net)下载
- 基于 PGI 图像进行下游任务的网络所需的（预处理）数据集从[网盘](https://pan.baidu.com/s/1TB630TOlZufn5SBVAEYZjw?pwd=vaoj)（提取码：vaoj）下载

下载好的数据集都放在 `data` 目录下，例如

```bash
/data
|--- /ModelNet40
|--- /PatchCollection
|--- /ShapeNetCore
```

## 引用

```bibtex
@article{zhang2023flattening,
    title={Flattening-Net: Deep Regular 2D Representation for 3D Point Cloud Analysis},
    author={Zhang, Qijian and Hou, Junhui and Qian, Yue and Zeng, Yiming and Zhang, Juyong and He, Ying},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2023}
}
```
