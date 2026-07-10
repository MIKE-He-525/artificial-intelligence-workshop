# Artificial Intelligence Workshop

人工智能工作坊课程项目（Workshop 0–11），包含理论 Notebook、编程练习、Mini-project 及配套 Python 脚本。

**学生：** MIKE-HE-525  
**项目路径：** `Artifitial-Intelligence-Workshop`  
**许可证：** [MIT License](LICENSE)

---

## 目录结构

| Workshop | 主题 | 主 Notebook | 作业脚本 |
|----------|------|-------------|----------|
| 0 | 课程介绍 | `AIWorkshop0.ipynb` | 无 |
| 1 | AI 与 Python 概念 | `AIWorkshop1.ipynb` | 无 |
| 2A | 机器学习概述 | `AIWorkshop2A.ipynb` | 无 |
| 2B | 数据预处理 | `AIWorkshop2B.ipynb` | Notebook 内 |
| 3 | 监督学习分类 | `AIWorkshop3.ipynb` | — |
| 4 | 集成学习 | `AIWorkshop4.ipynb` | `random_forests.py` 等 |
| 5 | 无监督学习 | `AIWorkshop5.ipynb` | `kmeans.py` 等 |
| 6 | KNN 推荐系统 | `AIWorkshop6.ipynb` | `knn_system.py` 等 |
| 7 | 逻辑编程 | `AIWorkshop7.ipynb` | `family_tree.py` 等 |
| 8 | 启发式搜索 | `AIWorkshop8.ipynb` | `maze_solver.py` 等 |
| 9 | AI 游戏 | `AIWorkshop9.ipynb` | `tic_tac_toe.py` 等 |
| 10 | 人工神经网络 | `AIWorkshop10.ipynb` | `perceptron_classifier.py` 等 |
| 11 | CNN 深度学习 | `AIWorkshop11.ipynb` | `cnn.py`、`face_*.py` 等 |

> 部分 Workshop（0、1、2A、10）存在双层嵌套目录，Notebook 位于内层同名文件夹中。

---

## 环境依赖

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow opencv-python Pillow kanren sympy
```

| 包 | 用途 |
|----|------|
| `scikit-learn` | Workshop 10 全部神经网络脚本（已从 neurolab 迁移） |
| `tensorflow` | Workshop 11 CNN / 人脸分类 |
| `kanren` | Workshop 7 逻辑编程 |
| `opencv-python` | Workshop 10 字符可视化、Workshop 11 数据加载 |

**Python 版本建议：** 3.10+

---

## 完成度概览

| Workshop | 状态 | 说明 |
|----------|------|------|
| 0 / 1 / 2A | — | 纯理论，无编程作业 |
| 2B | ✅ | 数据预处理练习已完成 |
| 3–9 | ✅ | 主体作业与 Mini-project 已完成 |
| 10 | ✅ | 全部脚本已迁移至 sklearn 并验证通过 |
| 11 | ⚠️ | 代码完整；Yale 人脸数据需自行解压到 `Yale/yalefaces/` |

### Workshop 10 脚本（sklearn 版）

| 脚本 | 功能 |
|------|------|
| `perceptron_classifier.py` | 感知机分类器 |
| `slnn.py` | 单层神经网络 |
| `mlnn.py` / `mlnn_2.py` / `mlnn_3.py` | 多层神经网络（回归） |
| `tsrnn.py` / `tsrnn_N.py` | 时序振幅预测 |
| `ocr_characters.py` | 英文 OCR |
| `ocr_Chn_characters.py` | 中文 OCR（20 字符，测试准确率 100%） |
| `character_visualizer.py` | OCR 字符可视化（需 GUI） |

### Workshop 11 Yale 人脸 Mini-project

1. 下载 `yalefaces.tar` 并解压到：

   ```
   AIWorkshop11 - Deep Learning with Convolutional Neural Networks/Yale/yalefaces/
   ```

2. 若数据未就绪，`data_loader.py` 会自动使用 Olivetti 人脸集作为替代。

3. 运行：

   ```bash
   python face_slnn.py
   python face_mlnn.py
   python face_cnn.py
   ```

---

## 运行示例

```bash
# Workshop 4 集成学习
cd "AIWorkshop4 - Predictive Analytics with Ensemble Learning"
python random_forests.py --classifier-type erf
python run_grid_search.py

# Workshop 7 逻辑编程
cd "AIWorkshop7 - Logic Programming"
python family_tree.py
python prime_LP.py

# Workshop 10 神经网络
cd "AIWorkshop10 - Artificial Neural Networks/AIWorkshop10 - Artificial Neural Networks"
python perceptron_classifier.py
python ocr_Chn_characters.py

# Workshop 11 CNN
cd "AIWorkshop11 - Deep Learning with Convolutional Neural Networks"
python cnn.py
```

---

## 主要修改记录

### 代码修复

- **Workshop 4**：`Construct *.py` 重命名为 `run_grid_search.py`、`feature_importance.py`
- **Workshop 7**：`family_tree.py` 配偶查询改为 kanren 逻辑推导；`prime_LP.py` 限制质数搜索范围
- **Workshop 10**：全部 `neurolab` 脚本迁移至 `sklearn`（兼容 NumPy 2.x，训练更快）
- **Workshop 11**：修复 `face_*.py` 数据路径与权重保存格式

### 已删除的多余文件

- 缓存：`.ipynb_checkpoints`、`__pycache__`、`.jupyter` 等
- 旧版 Notebook：`*_old.ipynb`
- 选做题（Bonus）：`lastcoin_pro.py`、`connect4_HEN_TP.py`
- 自动生成重复图表、验证工具脚本、`logpy-master/` 本地库副本

### 选做题说明

课程中仅 **Workshop 9** 标注了 `(Bonus)` 选做题，相关 Bonus 脚本已删除。其余 Workshop 练习（含 `ocr_Chn_characters.py`、各 Mini-project）均为**必做**。

---

## 提交物清单

每节 Workshop 通常需提交：

1. **Notebook**（`.ipynb`）— 含 Workshop Answer 文字与运行输出
2. **Python 脚本**（`.py`）— 按作业要求命名
3. **结果图**（`.png`）— 如 `workshop10.1.2.png`、`4.2_*.png` 等

---

## 注意事项

- Workshop 9 人机对战脚本（`tic_tac_toe.py`、`lastcoin.py` 等）需交互输入，请在终端手动运行
- `character_visualizer.py` 使用 `cv2.imshow`，需图形界面
- Workshop 7 `puzzle_solver.py` 逻辑谜题搜索耗时较长，属正常现象

---

*最后更新：2026-07-10*
