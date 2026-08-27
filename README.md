# Artificial Intelligence Workshop

人工智能工作坊 Workshop 0–11：理论 Notebook、编程练习与 Mini-project。

> **仓库信息：** `MIKE-He-525/artificial-intelligence-workshop`  
> 当前 GitHub 远程名称仍为 `Artifitial-Intelligence-Workshop`（拼写待修正）

---

## 这是什么

本仓库包含人工智能课程的 13 个 Workshop（Workshop 0 至 11，包括 2A 和 2B），涵盖从基础概念到深度学习的完整学习路径。每个 Workshop 包含理论讲解 Notebook 和相应的 Python 编程练习。

## 完成度

| Workshop | 状态 | 说明 |
|----------|------|------|
| 0 / 1 / 2A | 理论 | 纯理论课程，无编程作业 |
| 2B–9 | ✅ 完成 | 数据预处理、分类、集成学习、聚类、推荐、逻辑编程、搜索、游戏 AI |
| 10 | ✅ 完成 | 人工神经网络（全部脚本已迁移至 `scikit-learn`） |
| 11 | ⚠️ 完成 | CNN 深度学习（Yale 人脸数据需自行准备） |

## 目录结构

```
artificial-intelligence-workshop/
├── AIWorkshop0 - Course Introduction/           # 课程介绍
├── AIWorkshop1 - AI with Python Primer Concepts/ # AI 与 Python 基础
├── AIWorkshop2A - An Overview on Machine Learning/ # 机器学习概览
├── AIWorkshop2B - Data Preparation/             # 数据预处理
├── AIWorkshop3 - Supervised Learning with Classification/ # 监督学习：分类
├── AIWorkshop4 - Predictive Analytics with Ensemble Learning/ # 集成学习
├── AIWorkshop5 - Detecting Patterns with Unsupervised Learning/ # 无监督学习
├── AIWorkshop6 - KNN for Building Recommendation Systems/ # KNN 推荐系统
├── AIWorkshop7 - Logic Programming/             # 逻辑编程
├── AIWorkshop8 - Heuristic Search Techniques/   # 启发式搜索
├── AIWorkshop9 - AI Games/                      # AI 游戏
├── AIWorkshop10 - Artificial Neural Networks/   # 人工神经网络
└── AIWorkshop11 - Deep Learning with Convolutional Neural Networks/ # CNN 深度学习
```

**主要内容：**

| Workshop | 主题 | Notebook | 关键脚本 |
|----------|------|----------|---------|
| 0 | 课程介绍 | `AIWorkshop0.ipynb` | 无 |
| 1 | AI 与 Python 概念 | `AIWorkshop1.ipynb` | 无 |
| 2A | 机器学习概述 | `AIWorkshop2A.ipynb` | 无 |
| 2B | 数据预处理 | `AIWorkshop2B.ipynb` | Notebook 内练习 |
| 3 | 监督学习与分类 | `AIWorkshop3.ipynb` | `utilities.py` |
| 4 | 集成学习 | `AIWorkshop4.ipynb` | `random_forests.py`、`grid_search.py` |
| 5 | 无监督学习 | `AIWorkshop5.ipynb` | `kmeans.py`、`mean_shift.py` |
| 6 | KNN 推荐系统 | `AIWorkshop6.ipynb` | `knn_classifier_*.py`、`movie_recommender.py` |
| 7 | 逻辑编程 | `AIWorkshop7.ipynb` | `family_tree.py`、`prime_LP.py` |
| 8 | 启发式搜索 | `AIWorkshop8.ipynb` | `maze_solver.py`、`puzzle_solver.py` |
| 9 | AI 游戏 | `AIWorkshop9.ipynb` | `connect4_*.py`、`hexapawn_*.py` |
| 10 | 人工神经网络 | `AIWorkshop10.ipynb` | `perceptron_classifier.py`、`ocr_*.py` |
| 11 | CNN 深度学习 | `AIWorkshop11.ipynb` | `cnn.py`、`face_cnn.py` |

> **注：** Workshop 0、1、2A、10 存在双层嵌套目录，Notebook 位于内层同名文件夹中。

## 快速开始

**克隆仓库：**

```bash
# 注意：当前远程仓库名称仍为 Artifitial-Intelligence-Workshop（拼写错误）
git clone https://github.com/MIKE-He-525/Artifitial-Intelligence-Workshop.git
cd Artifitial-Intelligence-Workshop
```

**安装依赖：**

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow opencv-python Pillow kanren sympy
```

**运行示例：**

```bash
# Workshop 4：集成学习
cd "AIWorkshop4 - Predictive Analytics with Ensemble Learning"
python random_forests.py

# Workshop 7：逻辑编程
cd "../AIWorkshop7 - Logic Programming"
python family_tree.py

# Workshop 10：人工神经网络（注意嵌套目录）
cd "../AIWorkshop10 - Artificial Neural Networks/AIWorkshop10 - Artificial Neural Networks"
python perceptron_classifier.py

# Workshop 11：CNN 深度学习
cd "../../AIWorkshop11 - Deep Learning with Convolutional Neural Networks"
python cnn.py
```

## 环境依赖

**Python 版本：** 3.10+

**核心依赖：**

| 包 | 用途 | Workshop |
|----|------|---------|
| `numpy`、`pandas`、`matplotlib` | 数据处理与可视化 | 全部 |
| `scikit-learn` | 机器学习算法、神经网络 | 3–6, 10 |
| `tensorflow` | 深度学习 / CNN | 11 |
| `opencv-python` | 图像处理 | 10, 11 |
| `Pillow` | 图像加载 | 10, 11 |
| `kanren` | 逻辑编程引擎 | 7 |
| `sympy` | 符号计算 | 7 |

**Workshop 11 数据准备（可选）：**

若使用 Yale 人脸数据集，需将 `yalefaces.tar` 解压至：

```
AIWorkshop11 - Deep Learning with Convolutional Neural Networks/Yale/yalefaces/
```

若数据缺失，`data_loader.py` 会自动回退至 Olivetti 人脸数据集。

## 注意事项

- **交互式脚本：** Workshop 9 的游戏脚本（如 `connect4_M2H_Negamax.py`）需要终端交互输入
- **GUI 要求：** Workshop 10 的 `character_visualizer.py` 使用 `cv2.imshow`，需要图形界面环境
- **长时运行：** Workshop 7 的 `puzzle_solver.py` 和 Workshop 8 的部分搜索脚本可能耗时较长
- **嵌套目录：** Workshop 0、1、2A、10 的脚本位于二级同名子目录内

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。
