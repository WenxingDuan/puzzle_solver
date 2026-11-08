# 罗塞塔石碑拼图定位助手

针对大英博物馆 800 片罗塞塔石碑拼图的本地助手。电脑负责图像处理，安卓手机通过同一局域网中的网页拍照：上传单块照片后，自动去背景、分离黑白文字，并在固定石碑底图上给出候选位置。

本项目基于 [Benjamin-Wegener/jigsaw_puzzle_solver](https://github.com/Benjamin-Wegener/jigsaw_puzzle_solver) 修改，保留原版浏览器相机页面与许可证。

## 功能

- 内置完整石碑底图，无需上传参考图。
- 自动提取单块拼图，支持手动框选修正外轮廓。
- 在拼图内部校正光照、压制灰色颗粒，生成可调强度的黑底白字预览。
- 使用文字特征和带掩膜的模板匹配，搜索旋转、缩放及轻微比例变化。
- 按石碑真实轮廓过滤候选，排除透明背景；判断依据是拼图本体，而非外接矩形的空角。
- 同时查看整碑位置、方向对齐后的拼图和对应文字局部；支持切换候选、缩放拖动和叠加查看。
- 点击“拍下一块”连续使用；调整预处理参数后清除旧定位结果。

## 本地运行

需要 Python 3.9+ 和现代浏览器。当前版本在 Windows、Chrome 和安卓尺寸的浏览器环境中验证过。

```sh
git clone https://github.com/WenxingDuan/puzzle_solver.git
cd puzzle_solver
python -m venv .venv
```

Windows：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe serve.py
```

安装依赖后，也可以双击 `start.cmd`。启动器优先使用项目内的 `.venv`，否则使用当前系统的 `python`。

macOS / Linux：

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python serve.py
```

电脑访问 [http://localhost:8765/](http://localhost:8765/)。手机连接同一个局域网，访问 `http://<电脑的局域网 IP>:8765/`。Windows 可用 `ipconfig` 查询地址。电脑需要保持开机；如果系统询问网络访问权限，请允许用于当前局域网。

关闭运行窗口或按 `Ctrl+C` 停止服务。

## 使用

1. 点击“拍一块”或“选择照片”，正面靠近拍摄单块拼图。
2. 程序自动去背景并定位，通常需要约 20–30 秒，实际时间取决于电脑性能。
3. 核对整碑标记和两幅局部对照图，按照旋转提示放置拼图。必要时切换其他候选。
4. 如果外轮廓截错，展开调整区，框住整块拼图并保留一点背景。如果文字残缺或底纹过多，调节黑白分割强度，然后点击“用当前效果定位”。
5. 点击“拍下一块”继续。

候选位置需要目视核对。当前搜索重点覆盖四个直角方向附近的轻微倾斜，以及约 10% 的横向比例变化；强透视、严重反光、模糊或文字过少的照片可能无法可靠定位。

## 验证

运行可独立复现的检查：

```sh
python selftest.py
```

检查包括透明像素隐藏颜色不影响底图、不规则边缘拼图的形状判断，以及背景和底图标记不参与匹配。

开发时另外用 5 张实拍图验证了预处理和候选位置，其中同一块拼图的 4 张不同拍摄结果一致。实拍图保存在本机 `.runtime/fixtures/`，不随仓库公开；默认测试会跳过缺失的照片。拥有这些本地照片时可运行：

```sh
python selftest.py --match-photos
```

原版浏览器匹配检查保留在 `/selftest.html`，原版相机页面保留在 `/index.html`。手机通过普通局域网 HTTP 访问时，使用默认页面的原生拍照/照片选择入口。

## 文件与数据

| 文件 | 用途 |
| --- | --- |
| `serve.py` | 网页服务、固定底图和图像处理接口 |
| `vision.py` | 前景分割、黑白处理、轮廓过滤与定位 |
| `mobile.html` | 手机和电脑界面 |
| `assets/rosetta-reference.png` | 固定石碑参考图 |
| `selftest.py` | 图像处理与匹配检查 |
| `start.cmd` | Windows 启动器 |
| `index.html`、原有 JS 与 `opencv.js` | 保留的上游浏览器版本 |

上传照片仅通过局域网发送到运行程序的电脑，不调用外部图像服务；处理接口使用有限的内存缓存，不自动把上传照片写到磁盘。运行日志、开发用照片、浏览器缓存、虚拟环境和 Python 缓存均不纳入 Git。

参考图由使用者提供，原有红色定位标记在匹配时排除；显示时自动生成石碑透明外轮廓。代码沿用 MIT 许可证，见 [LICENSE](LICENSE)。
