# Batch Model Runner

这个小项目参考了 `ORBIT-main` 里的两部分思路：

- 用统一的 HuggingFace 封装来加载模型
- 用批量数据读取和批量生成来完成跑批

但这里**不做任何模型改动**，不做 activation steering，不加 hook，不改 forward，只保留“原始模型加载 + 批量推理 + 结果落盘”。

## 支持能力

- 支持 HuggingFace 因果语言模型直接加载
- 支持通过 `code/data/loader.py` 直接按数据集名读取 benchmark
- 支持 `generation` 和 `chat` 两种提示格式
- 支持按批处理
- 支持逐条写出 `jsonl`
- 支持断点续跑
- 自带一个最小示例输入：`examples/input.jsonl`

## 输入数据格式

脚本现在直接使用 `code/data/loader.py` 里的统一数据集入口：

- `--dataset-name gsm8k`
- `--dataset-split test`
- `--data-root /path/to/data_root`

脚本会把 loader 返回的 `(prompt, correct_answer, wrong_answer)` 自动转换成记录对象，再沿用原来的批处理推理流程。

## 运行方式

### 最简单

```bash
python run_batch.py \
  --model /path/to/your/model \
  --model /path/to/your/model \
  --dataset-name gsm8k \
  --dataset-split test \
  --data-root /path/to/data_root \
  --output-file /path/to/output.jsonl \
  --prompt-field prompt \
  --batch-size 8 \
  --max-new-tokens 128
```

### 使用聊天模板

```bash
python run_batch.py \
  --model /path/to/your/model \
  --dataset-name gsm8k \
  --dataset-split test \
  --data-root /path/to/data_root \
  --output-file /path/to/output.jsonl \
  --prompt-format chat \
  --system-prompt "你是一个严谨的助手。" \
  --batch-size 4
```

### 断点续跑

```bash
python run_batch.py \
  --model /path/to/your/model \
  --dataset-name gsm8k \
  --dataset-split test \
  --data-root /path/to/data_root \
  --output-file /path/to/output.jsonl \
  --prompt-field prompt \
  --resume
```

## 输出格式

输出文件当前固定写成 `jsonl`，每条记录会保留原字段，并新增一个默认字段：

- `model_output`

如果你想改名，可以使用：

```bash
--output-field prediction
```

## 依赖

你需要自行安装：

```bash
pip install torch transformers tqdm
```
