# 基于 GPT-2 的短信垃圾分类器

本项目通过微调预训练语言模型 GPT-2，在 UCI 的 SMS Spam Collection 数据集上构建一个 **二分类短信垃圾检测模型**，实现对短信内容是否为垃圾短信（`spam`）的自动识别。

主要工作包括：

- 自动下载与解压公开数据集  
- 数据清洗与类别平衡（处理 `ham` / `spam` 样本不均衡问题）  
- 构建自定义 `Dataset` 与 `DataLoader`  
- 基于 GPT-2 的分类模型设计与部分参数微调  
- 训练过程中的损失、准确率监控与可视化  
- 提供简单易用的预测接口与模型持久化保存  

---

## 1. 项目结构

仓库推荐结构如下（你可以根据实际情况调整）：

```text
.
├── build_classifier_from_gpt2.ipynb   # 主 Notebook，包含完整流程
├── README.md                          # 项目说明文档
├── gpt2_classifier.pt                 # 训练好的模型权重（运行 Notebook 后生成）
├── Loss.png                           # 训练 / 验证损失曲线可视化（运行 Notebook 后生成）
└── .cache/                            # 缓存目录（数据集及拆分文件）
    ├── SMSSpamCollection.tsv          # 原始数据（自动下载）
    ├── train.csv                      # 训练集
    ├── val.csv                        # 验证集
    └── test.csv                       # 测试集
````
---

## 2. 环境依赖

建议使用 Python 3.9+。

核心依赖：

* [PyTorch](https://pytorch.org/)
* [Transformers](https://huggingface.co/docs/transformers/index)
* `pandas`
* `scikit-learn`
* `matplotlib`
* `tqdm`
* `requests`

示例安装命令（使用 `pip`）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # 按照自己环境选择 CPU/CUDA 版本
pip install transformers pandas scikit-learn matplotlib tqdm requests
```

如使用 `conda`，可以先创建环境：

```bash
conda create -n gpt2-spam python=3.10
conda activate gpt2-spam
# 然后再安装上述依赖
```

---

## 3. 数据集说明

本项目使用的公开数据集为 **SMS Spam Collection**，包含约 5,574 条英文短信文本，每条短信带有标签：

* `ham`：正常短信
* `spam`：垃圾短信

Notebook 中会自动：

1. 从 UCI 公开地址下载压缩包；
2. 解压并重命名为 `SMSSpamCollection.tsv`；
3. 使用 `pandas` 读入并完成后续预处理。

在此基础上，项目做了以下数据处理：

* **类别平衡**：

  * 统计 `spam` 样本数量；
  * 从 `ham` 中按照相同数量随机采样；
  * 拼接形成一个 **正负样本平衡** 的数据集。
* **标签编码**：

  * `spam` → `1`
  * `ham` → `0`
* **数据划分**（分层抽样，保证标签比例一致）：

  * 训练集：70%
  * 验证集：20%（来自剩余 30% 中的 1/3）
  * 测试集：10%（来自剩余 30% 中的 2/3）

划分结果会保存为 `.cache/train.csv` 等文件，方便重复使用。

---

## 4. 模型设计

### 4.1 文本编码与 Dataset

项目采用 HuggingFace 的 `AutoTokenizer` 对文本进行编码：

* 基础模型：`gpt2`
* 使用 GPT-2 的 `eos_token` 作为 `pad_token`
* 通过 `padding="max_length"` 与 `truncation=True` 控制序列长度

自定义了一个简单的 `Dataset` 类 `SimpleDataset`，主要负责：

* 接收 `pandas.DataFrame`（包含 `Text` 和 `Label` 列）
* 使用 tokenizer 将文本批量编码为：

  * `input_ids`
  * `attention_mask`
* 将标签转换为 `torch.tensor`

并基于该 `Dataset` 构建了 `DataLoader`：

* `batch_size = 8`
* 训练集：`shuffle=True, drop_last=True`
* 验证 / 测试集：`drop_last=True`

Notebook 中还提供了一个 **可选方案**：使用 `datasets.Dataset` 与 `DatasetDict` 来构建数据集与 `DataLoader`，目前代码中处于注释状态，如有需要可以自行开启。

### 4.2 GPT-2 分类模型

核心模型类为 `GPTModelForClassfication`，主要结构为：

* `self.gpt = AutoModel.from_pretrained(model_name)`

  * 加载预训练 GPT-2 作为编码器
* 获取 GPT-2 的词向量维度 `emb_dim`
* 追加一个线性分类头：

  * `self.classifier = nn.Linear(emb_dim, 2, bias=False)`

参数微调策略：

* **首先冻结** GPT-2 的全部参数：
  `p.requires_grad = False`
* **再解冻**：

  * 最后的 LayerNorm 层：`gpt.ln_f`
  * 最后一层 Transformer block：`gpt.h[-1]`

这样做的好处：

* 训练更稳定，不容易过拟合小数据集；
* 训练开销较小；
* 仍然能利用预训练模型的语义表示能力。

前向传播逻辑：

* 将 tokenizer 输出的字典（`input_ids`、`attention_mask` 等）传入 GPT-2；
* 取最后一个 token 的 hidden state（类似 “句子表征”）：

  * `cls_rep = outputs.last_hidden_state[:, -1, :]`
* 通过线性分类层映射到 2 类 logits：

  * `logits = self.classifier(cls_rep)`

---

## 5. 训练与评估

### 5.1 训练配置

* 优化器：`AdamW`

  * 仅更新 `requires_grad=True` 的参数
  * `lr = 5e-5`
  * `weight_decay = 0.1`
* 损失函数：`CrossEntropyLoss`
* 学习率调度：`get_cosine_schedule_with_warmup`

  * `num_training_steps = 5 * len(train_loader)`（5 个 epoch）
  * `num_warmup_steps = num_training_steps // 10`
* 训练轮数：`num_epochs = 5`
* 评估频率：

  * 每隔 `eval_freq` 若干 step，调用 `evaluate_model`：

    * 计算训练集和验证集上的平均损失（`train_loss` / `val_loss`）
    * 将结果记录到 `train_losses` / `val_losses`

设备选择（Notebook 中示例）：

```python
device = "mps"   # 适用于 Apple Silicon
model.to(device)
```

如在其他环境运行，建议手动修改为：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

### 5.2 指标计算

定义了若干辅助函数：

* `calc_loss_batch`：计算一个 batch 的交叉熵损失
* `calc_loss_loader`：在若干 batch 上平均计算损失
* `calc_accracy_loader`：在 DataLoader 上计算准确率
* `evaluate_model`：同时返回 train / val 集上的平均损失

训练循环中会定期打印：

* 当前 epoch、global step
* 训练损失 / 验证损失
* 每个 epoch 结束后：

  * 训练集准确率
  * 验证集准确率

### 5.3 可视化

定义函数 `plot_loss` 绘制损失曲线：

* x 轴一：epoch（0～5）
* x 轴二（上方）：累计样本数 `samples_seen`
* y 轴：训练 / 验证损失

图像会保存为：

```text
Loss.png
```

---

## 6. 使用方法

### 6.1 快速开始

1. 克隆仓库：

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. 创建并激活虚拟环境（可选）。

3. 安装依赖。

4. 启动 Jupyter：

   ```bash
   jupyter notebook
   # 或
   jupyter lab
   ```

5. 打开 `build_classifier_from_gpt2.ipynb`，从上到下依次运行各个 Cell：

   * 数据下载与预处理
   * 数据集与 DataLoader 构建
   * 模型定义与初始化
   * 训练与评估
   * 保存与推理示例

### 6.2 单条短信分类示例

Notebook 中提供了一个简单的预测函数 `classify`：

```python
def classify(model, text, tokenizer, device):
    inputs = tokenizer(
        [text],
        max_length=120,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(inputs)
    predicted = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted else "not spam"
```

使用示例：

```python
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify(model, text_1, tokenizer, device))
# 可能输出: "spam"

text_2 = (
    "Hey, just wanted to check if we're still on "
    "for dinner tonight? Let me know!"
)

print(classify(model, text_2, tokenizer, device))
# 可能输出: "not spam"
```

> 训练完成后，可以将 `model` 与 `tokenizer` 封装为脚本或 API，用于实际应用场景。

### 6.3 模型保存与加载

Notebook 中在训练结束后保存模型权重：

```python
torch.save(model.state_dict(), "gpt2_classifier.pt")
```

在新的脚本或 Notebook 中加载：

```python
model = GPTModelForClassfication("gpt2")
model.load_state_dict(torch.load("gpt2_classifier.pt", map_location=device))
model.to(device)
model.eval()
```

---

## 7. 实验结果（示例）

在当前 Notebook 配置下（5 个 epoch，基于平衡后的数据集），一次训练运行得到的结果如下（仅供参考，不同随机种子或参数可能略有差异）：

* 训练集准确率：约 **0.979**
* 验证集准确率：约 **0.951**
* 测试集准确率：约 **0.973**

从结果可以看到，基于 GPT-2 的微调模型在该短信垃圾分类任务上具有较好的表现。

---

## 8. 可扩展方向

你可以在此项目的基础上进行如下扩展：

* 支持更多预训练模型（如 `gpt2-medium`、`distilgpt2` 等）；
* 尝试解冻更多 Transformer 层，比较不同微调策略对效果的影响；
* 引入更多正则化手段（dropout、label smoothing 等）；
* 将 Notebook 中的训练流程抽取为 Python 脚本或命令行工具；
* 将分类接口封装成 REST API 或 Web Demo。
