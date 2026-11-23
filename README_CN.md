<p align="center">
  <img src="assets/gfpgan_logo.png" height=130>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

## 🍴 关于本 fork

这是 [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)（基于上游 v1.3.8）的维护分支，升级至 **v1.4.0**：

- **全环境可跑**：兼容 torchvision ≥ 0.17、torch ≥ 2.6/2.7（含 RTX 50 系）、NumPy 2.x、Python 3.8–3.12
- **论文原版模型（V1）复活**：StyleGAN2 算子内置精确的纯 PyTorch 回退，无需编译 CUDA 扩展、支持 CPU 直接推理
- **修复约 60 个可复现的上游问题**（逐条分诊了 GFPGAN + Real-ESRGAN 共 961 个 open issue），详见 [CHANGELOG.md](CHANGELOG.md) 与英文 README 的 [Troubleshooting](README.md#ambulance-troubleshooting--compatibility) 章节
- **工程化**：18 个测试全绿（CI 跑 3.11/3.12）、Lint 流水线、CPU Docker 镜像、PEP 621 现代打包
- **新参数**：`--device`、`--fp32`、`--model_path`、`--no_parse`、`--seed`；API 增加 `GFPGANer(..., device=..., use_parse=...)`

## 🛠 常见问题与兼容性说明

本分支修复了上游大量环境兼容性问题，包括：torchvision ≥ 0.17 的 `functional_tensor` 报错、`tb-nightly` 安装失败、NumPy 2.x 不兼容、GPU 权重文件在纯 CPU 机器上无法加载、部分显卡背景变黑、离线/被墙环境下载失败、RTX 50 系显卡 "no kernel image" 等。完整说明见英文 README 的 [Troubleshooting & Compatibility](README.md#ambulance-troubleshooting--compatibility) 一节。

快速上手：

```bash
pip install -r requirements.txt
python setup.py develop
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.4 -s 2
```

## GitHub Release

本 fork 不会向 PyPI 发布已被上游占用的 `gfpgan` 包名。维护者将版本号同步到 `pyproject.toml` 和 `VERSION` 后，推送形如 `vX.Y.Z` 的 tag，即可触发 [`.github/workflows/release.yml`](.github/workflows/release.yml)。工作流会构建并校验源码包和 wheel，将构建产物保存为 Actions artifact，并附加到 GitHub Release。

常用参数：

- `--fp32`：修复部分显卡上背景增强输出全黑的问题
- `--model_path`：指定本地模型文件，无需联网自动下载
- `--device`：指定运行设备（`cuda`、`cuda:1`、`cpu`）
- `--seed`：固定 Python、NumPy 和 PyTorch 随机数，便于复现实验和批处理结果
- `-w`：权重参数对已发布模型无效（保留仅为兼容，会打印提示）
- 训练需 `basicsr==1.4.2`，报 `No module named 'basicsr.losses.gan_loss'` 说明 basicsr 版本过旧
- Windows 下含中文/特殊字符的文件名现已支持；模型文件损坏（如 `PytorchStreamReader failed reading zip archive`）会给出明确的重下提示
- 人脸检测/解析权重目录已改为包内绝对路径，从任意目录启动都能正常工作

还未完工，欢迎贡献！
