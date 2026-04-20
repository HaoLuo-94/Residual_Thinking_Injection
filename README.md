# Residual Decoding: 方法说明与工程化改进

## 摘要

本文对 `code/model/residual.py` 与 `code/batch_runner/model_wrapper.py` 中实现的 residual decoding 方法进行形式化说明。该方法旨在不改变主模型参数、不过训练目标的前提下，在自回归解码过程中由当前步输出分布动态构造 residual control signal，并将该信号注入后续 decoder layers 的 hidden states，以调控模型从内部“思考阶段”过渡到最终“回答阶段”的生成行为。与仅依赖 prompt 诱导或固定长度思考模板的方法相比，该方案同时引入了基于 logits 的语义残差构造机制与基于熵的自适应停止机制，从而在可控性、可解释性和工程可部署性上具有更强表现。

## 1. 问题定义

对于标准自回归语言模型，在给定输入 prompt 后，模型在第 `t` 个生成步输出词表上的 logits：

$$
\mathbf{z}_t \in \mathbb{R}^{V},
$$

其中 `V` 表示词表大小。传统解码策略通常仅基于 `\mathbf{z}_t` 执行贪心选择或采样，而不显式修改模型内部状态的演化路径。本文方法关注如下问题：

> 是否可以在推理期根据模型当前输出分布构造一个低成本、可解释的控制信号，并将其作为 residual 注入后续层表示，从而影响模型后续生成轨迹，尤其是“思考”与“回答”之间的切换过程？

为此，当前实现提出一种 decoding-time residual injection 框架。其核心不在于重新训练模型，而在于通过闭环式的推理控制，利用当前步输出分布反向影响下一步 hidden state 的更新。

## 2. 方法

### 2.1 基于 top-k 分布的 residual 构造

在第 `t` 步，给定 logits `\mathbf{z}_t`，首先选取 top-k 候选集合：

$$
\mathcal{K}_t = \operatorname{TopK}(\mathbf{z}_t, k).
$$

随后在该集合上重新归一化，得到局部概率分布：

$$
p_{t,i} = \frac{\exp(z_{t,i})}{\sum_{j \in \mathcal{K}_t}\exp(z_{t,j})}, \quad i \in \mathcal{K}_t.
$$

记 token `i` 对应的输入 embedding 为 `\mathbf{e}_i \in \mathbb{R}^{H}`，其中 `H` 为 hidden size，则当前步的 soft semantic representation 定义为：

$$
\mathbf{s}_t = \sum_{i \in \mathcal{K}_t} p_{t,i}\mathbf{e}_i.
$$

进一步通过线性投影得到 residual 向量：

$$
\mathbf{\delta}_t = W\mathbf{s}_t, \qquad W \in \mathbb{R}^{H \times H}.
$$

该设计对应 `SimpleDeltaBuilder` 的实现。与直接使用 argmax token embedding 或手工定义 steering direction 不同，这里的 `\mathbf{\delta}_t` 来自当前高概率候选集合的加权语义重心，因而能够更稳定地反映当前解码分布所蕴含的局部语义趋势。

### 2.2 Decoder 层中的 residual 注入

在 `residual.py` 中，原始 Qwen3 decoder layer 被替换为 `PatchedQwen3DecoderLayer`。注入位置位于 self-attention 之后、MLP 之前。设第 `l` 层在时刻 `t` 经 attention 残差更新后的 hidden states 为：

$$
\mathbf{h}^{(l)}_t,
$$

则注入后的表示为：

$$
\tilde{\mathbf{h}}^{(l)}_t = \mathbf{h}^{(l)}_t + \alpha \gamma_l \mathbf{\delta}_t.
$$

其中，`\alpha` 表示全局注入强度，`\gamma_l` 表示层相关的缩放因子。当前实现采用线性层深加权：

$$
\gamma_l = \frac{l+1}{L},
$$

其中 `L` 为总层数，`l = 0,1,\dots,L-1`。此外，注入层范围还可以通过 `layer_start` 与 `layer_end` 指定。

该设计的动机在于：深层表示通常承载更高层次的语义信息，因此对更深层施加更强的 residual 干预，能够在降低底层表示扰动的同时，更直接地影响最终解码决策。

### 2.3 基于熵的 thinking phase 判定

为了避免固定长度思考过程，当前方法引入基于输出分布熵的自适应停止机制。首先，基于全词表 logits 计算 softmax 概率：

$$
p_{t,i} = \frac{\exp(z_{t,i})}{\sum_j \exp(z_{t,j})}.
$$

对应的熵定义为：

$$
H_t = -\sum_i p_{t,i}\log p_{t,i}.
$$

当 `H_t` 较低时，说明模型输出分布更尖锐、生成决策更确定，可将其视为内部“思考”趋于收敛的信号。为增强鲁棒性，当前实现并不使用单步阈值，而是定义连续低熵计数器：

$$
c_t =
\begin{cases}
c_{t-1}+1, & H_t < \tau, \\
0, & H_t \ge \tau,
\end{cases}
$$

其中 `\tau` 为熵阈值。

在 `model_wrapper.py` 的工程化版本中，thinking phase 的结束条件定义为：

$$
\big(t \ge T_{\min} \land c_t \ge P\big) \;\lor\; t \ge T_{\max},
$$

其中：

- `T_{\min}` 表示最小思考步数；
- `T_{\max}` 表示最大思考步数；
- `P` 表示连续低熵 patience。

该停止规则相较原型实现更加稳健。其一，`T_{\min}` 提供了下界约束，避免模型在早期偶发低熵时过早结束；其二，`T_{\max}` 提供了上界约束，避免在特殊样本上产生过长的 think 段。

### 2.4 显式的阶段切换与输出控制

当 thinking phase 满足结束条件后，当前实现会将 `</think>` 对应 token 序列显式送入上下文，并在内部状态中切换到 answer phase。因而，模型内部的生成上下文可表示为：

$$
\text{prompt} \rightarrow \text{think tokens} \rightarrow </think> \rightarrow \text{answer tokens}.
$$

然而在最终输出阶段，系统只返回 `answer_tokens` 对应文本，而不显示 `think_tokens`。这意味着该方法同时满足两类需求：

1. 在模型内部保留具有结构边界的“思考-回答”过渡；
2. 在外部接口上保持输出简洁，避免暴露中间思考内容。

### 2.5 闭环控制视角下的递推过程

将上述过程写成递推形式，可得到：

$$
\mathbf{z}_t \rightarrow \big(H_t, \mathbf{\delta}_t\big),
$$

$$
\mathbf{\delta}_t \rightarrow \tilde{\mathbf{h}}^{(l)}_{t+1},
$$

$$
\tilde{\mathbf{h}}^{(l)}_{t+1} \rightarrow \mathbf{z}_{t+1}.
$$

因此，当前方法并非一次性注入固定 steering vector，而是构成一个轻量级闭环：当前步输出分布决定下一步 residual 注入，而下一步 residual 注入又进一步影响新的输出分布。

## 3. 相较原型实现的工程化改进

若将 `code/model/residual.py` 视为方法原型，则 `code/batch_runner/model_wrapper.py` 可以视为面向实际推理流程的工程化扩展。其改进主要体现在以下几个方面。

### 3.1 从单文件验证脚本到统一推理封装

原型实现主要关注方法可行性验证，包括：

- 是否能够在 Qwen3 decoder layer 中接入 residual injection；
- 是否能够从 logits 动态构造 `\mathbf{\delta}_t`；
- 是否能够通过 entropy gating 触发 think/answer 切换。

相比之下，`model_wrapper.py` 将这些能力整合进 `BatchModelRunner`，实现了统一的 tokenizer 加载、模型初始化、prompt 构造与批量推理接口。由此，方法从“实验性脚本”转变为“可复用推理组件”。

### 3.2 参数配置的结构化

工程化实现引入了 `GenerationConfig` 与 `ResidualGenerationConfig`，将方法相关参数系统化管理，包括：

- 注入强度 `alpha`；
- 注入层范围 `layer_start`, `layer_end`；
- 熵阈值 `entropy_threshold`；
- 连续低熵耐心值 `low_entropy_patience`；
- 最小与最大思考步数 `min_think_steps`, `max_think_steps`；
- residual 构造时的 `topk`；
- think 阶段边界标记 `think_end_token`。

这种结构化配置有利于实验复现、超参数扫描和消融研究。

### 3.3 停止机制的稳定化

原型版本主要依赖“连续低熵达到阈值”来结束 think 阶段，而工程化版本在此基础上增加了最小步数与最大步数约束，即：

- 低于 `T_{\min}` 时不允许结束；
- 达到 `T_{\min}` 后，若连续低熵满足 patience 则结束；
- 若达到 `T_{\max}`，则强制结束。

从控制策略角度看，这一改动将原先的单条件停止规则扩展为带上下界的自适应停止机制，显著提升了生成长度控制的稳定性。

### 3.4 输入格式的泛化

工程实现同时支持 `completion/generation` 与 `chat` 两类 prompt 格式，并在后者中自动调用 tokenizer 的 `apply_chat_template`。这使得该方法可以直接接入现代聊天模型推理流程，而不再依赖手工拼接 prompt。

### 3.5 与基线解码路径的兼容

`model_wrapper.py` 保留了普通 HuggingFace `generate` 路径，并仅在 `residual_config.enabled=True` 时启用自定义逐 token 解码循环。因此，同一套封装能够同时支持：

$$
\text{Base Generation} \quad \text{vs.} \quad \text{Residual Decoding}.
$$

这一设计降低了实验对比时的额外变量，使得基线与改进方法之间的差异更集中地体现在 residual injection 与 entropy gating 本身。

### 3.6 面向实际输出接口的行为设计

工程版本不仅维护 think/answer 两阶段内部状态，还支持：

- 仅返回 answer 段文本；
- 兼容多个 EOS token；
- 直接被 batch runner 逐条调用。

因此，其目标已不只是展示方法内部运行轨迹，而是提供可接入评测与推理流水线的可部署解码策略。

## 4. 方法特性与潜在价值

从方法论角度看，当前方案具有以下几个值得强调的特征。

第一，控制信号来源于模型自身当前步输出分布，而非外部人工指定方向，因此具有更强的上下文自适应性。

第二，注入对象是 decoder hidden states 而非 logits 后处理，因此干预发生在表示层面，能够对后续状态演化产生连续影响。

第三，思考阶段长度并非固定，而由熵、patience 与上下界共同决定，因而比固定模板长度更灵活。

第四，该方法完全工作于推理阶段，不要求重新训练主模型，在部署成本上较低。

综合而言，可以将本方法概括为：

> 本文方法利用当前解码分布中 top-k 候选的软语义重心构造 residual control signal，并通过层深加权方式注入后续 decoder hidden states；同时结合基于熵的自适应阶段切换机制，实现对“思考-回答”生成过程的推理期控制。

## 5. 结论

`residual.py` 给出了该方法的最小可行原型，验证了 residual injection、动态 `delta` 构造与 entropy-based phase switching 的基本可实现性；`model_wrapper.py` 则进一步完成了封装化、配置化和推理流程整合，使其成为可在批量实验环境中直接使用的 residual decoding 方案。

就当前实现而言，其核心改进可归纳为三点：

1. 基于 logits top-k 分布构造 soft semantic residual；
2. 采用层深加权方式将 residual 注入 decoder hidden states；
3. 采用熵阈值、patience 以及最小/最大思考步数组成的自适应阶段切换策略。

这些设计共同构成了一种无需额外训练、具备较强可控性与工程可落地性的推理期控制方法。
