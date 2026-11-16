<p align="center">
  <img src="assets/gfpgan_logo.png" height=130>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

## 🛠 常见问题与兼容性说明

本分支修复了上游大量环境兼容性问题，包括：torchvision ≥ 0.17 的 `functional_tensor` 报错、`tb-nightly` 安装失败、NumPy 2.x 不兼容、GPU 权重文件在纯 CPU 机器上无法加载、部分显卡背景变黑、离线/被墙环境下载失败、RTX 50 系显卡 "no kernel image" 等。完整说明见英文 README 的 [Troubleshooting & Compatibility](README.md#ambulance-troubleshooting--compatibility) 一节。

快速上手：

```bash
pip install -r requirements.txt
python setup.py develop
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.4 -s 2
```

常用参数：

- `--fp32`：修复部分显卡上背景增强输出全黑的问题
- `--model_path`：指定本地模型文件，无需联网自动下载
- `--device`：指定运行设备（`cuda`、`cuda:1`、`cpu`）
- `-w`：权重参数对已发布模型无效（保留仅为兼容，会打印提示）
- 训练需 `basicsr==1.4.2`，报 `No module named 'basicsr.losses.gan_loss'` 说明 basicsr 版本过旧
- Windows 下含中文/特殊字符的文件名现已支持；模型文件损坏（如 `PytorchStreamReader failed reading zip archive`）会给出明确的重下提示
- 人脸检测/解析权重目录已改为包内绝对路径，从任意目录启动都能正常工作

还未完工，欢迎贡献！
