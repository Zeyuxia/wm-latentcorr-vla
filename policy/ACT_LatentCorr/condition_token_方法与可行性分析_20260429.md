# Condition Token 方法与可行性分析

说明：下面所有公式都保持为可直接复制的单行 LaTeX 风格文本，便于粘贴到飞书或文档中。

## 1. 问题定义

我们在时刻 t 的真实部署输入只有当前信息 x_t = (o_t, q_t)，其中 o_t 是当前图像观测，q_t 是当前机器人状态。

原始方法里的 conditioned path 依赖 future condition token，而这个 token 来自未来观测，因此训练时可以构造，推理时无法获得。核心目标就是：把这个 future-dependent token，改造成一个 current-observation-conditioned、推理时可直接得到的 token。

## 2. 旧方法：teacher future token 路径

旧方法里，ACT 主干先从当前观测抽取视觉特征，然后 teacher 从未来图像提 future latent，再把它投到 ACT 可以消费的 condition token 空间。

```text
z_t^{act} = f_{act}(o_t)
```

```text
z_{t+\Delta}^{wm} = E_{wm}(o_{t+\Delta})
```

```text
\tilde z_{t+\Delta}^{wm} = A(z_{t+\Delta}^{wm})
```

```text
c_t^\star = P(\mathrm{pool}(\tilde z_{t+\Delta}^{wm}))
```

```text
\hat a_{t:t+H-1}^{cond} = \pi_{ACT}(x_t; c_t^\star)
```

```text
\mathcal{L}_{cond}^{teacher} = \ell(\hat a_{t:t+H-1}^{cond}, a_{t:t+H-1})
```

## 3. 旧方法的部署缺陷

问题在于 c_t^\star 依赖 o_{t+\Delta}。训练时优化的是带 teacher token 的路径，部署时却只能走不带该 token 的 base 路径。

也就是说，训练时和推理时不是同一条计算图，这就是 condition token 的割裂点。

```text
\text{train: } \pi_{ACT}(x_t; c_t^\star)
```

```text
\text{test: } \pi_{ACT}(x_t)
```

## 4. 新方法：预测一个可部署的 condition token

新的核心思想是：不再直接把 teacher future token 喂给 ACT，而是学习一个从当前输入预测 token 的 predictor。teacher token 只保留为监督目标。

实现上，不直接拿 raw image 做预测，而是先走 ACT 当前视觉特征和 projector，再拼接当前状态 q_t。

```text
z_t^{proj} = \Pi(z_t^{act}) = \Pi(f_{act}(o_t))
```

```text
h_t = [\mathrm{pool}(z_t^{proj}); q_t]
```

```text
\hat c_t = g_\phi(h_t)
```

```text
\hat a_{t:t+H-1}^{pred} = \pi_{ACT}(x_t; \hat c_t)
```

## 5. 新方法里的 teacher token 仍然是什么

teacher token 的定义本身没有变，它仍然表达未来状态信息；改变的是它的使用方式。它不再直接作为部署路径的输入，而是作为 predictor 的监督信号。

```text
z_{t+\Delta}^{wm} = E_{wm}(o_{t+\Delta})
```

```text
\tilde z_{t+\Delta}^{wm} = A(z_{t+\Delta}^{wm})
```

```text
c_t^\star = P(\mathrm{pool}(\tilde z_{t+\Delta}^{wm}))
```

## 6. 新增的 loss 设计

围绕 condition token，现在有三类核心监督：base action loss、predicted-token conditioned action loss，以及 token 对齐 loss。若保留 latent dynamics，还会有 dynamics loss。

其中 condition-related loss 需要渐进式升权，因为训练初期 \hat c_t 还不准确，不能太早把它当成强监督主路径。

```text
\hat a_{t:t+H-1}^{base} = \pi_{ACT}(x_t)
```

```text
\mathcal{L}_{base} = \ell(\hat a_{t:t+H-1}^{base}, a_{t:t+H-1})
```

```text
\hat a_{t:t+H-1}^{pred} = \pi_{ACT}(x_t; \hat c_t)
```

```text
\mathcal{L}_{cond-pred} = \ell(\hat a_{t:t+H-1}^{pred}, a_{t:t+H-1})
```

```text
\mathcal{L}_{token} = \|\hat c_t - c_t^\star\|_2^2
```

```text
\hat z_{t+\Delta} = F_\psi(z_t^{proj}, a_{t:t+\Delta-1})
```

```text
\mathcal{L}_{dyn} = \|\hat z_{t+\Delta} - \tilde z_{t+\Delta}^{wm}\|_2^2
```

```text
\mathcal{L} = \lambda_a \mathcal{L}_{base} + \lambda_c(t)\mathcal{L}_{cond-pred} + \beta(t)\lambda_{tok}\mathcal{L}_{token} + \beta(t)\mathcal{L}_{dyn} + \cdots
```

## 7. 为什么它不是“两次推理”

这条路线的关键工程优势是：它不是先推一次 future，再推一次 current。

推理时，condition token 是从当前特征同图预测出来的一个 head，随后立刻被当前 ACT forward 消费，所以本质上仍是一张图里的一次 forward。

```text
x_t \rightarrow z_t^{act} \rightarrow z_t^{proj} \rightarrow \hat c_t
```

```text
(x_t, \hat c_t) \rightarrow \hat a_{t:t+H-1}^{pred}
```

## 8. 可行性分析：我们到底在验证什么

真正要验证的是映射 x_t \mapsto c_t^\star 是否足够可预测。

如果这张映射严重 one-to-many，那么任何单点预测器 \hat c_t = g_\phi(x_t) 都很难学；反之，如果简单 probe 已经显著优于均值基线，就说明当前输入里确实含有 future token 的可恢复结构。

```text
x_t \mapsto c_t^\star
```

```text
\hat c_t = g_\phi(x_t)
```

## 9. 可行性分析：我们具体怎么做

我们没有直接在 raw pixel 上做统计，而是用与真实训练路径更一致的当前特征：pool(projector latent) 与 q_t 的拼接。

当前验证里，feature 维度是 18，target token 维度是 512。然后用两个最简单的 probe：线性回归和 kNN 回归，并与“永远输出平均 token”的均值基线比较。

```text
\phi(x_t) = [\mathrm{pool}(z_t^{proj}); q_t]
```

```text
\hat c_t = W\phi(x_t) + b
```

```text
\hat c_t = \mathrm{kNN}(\phi(x_t))
```

```text
\hat c_t = \bar c
```

## 10. 当前数值结果与含义

当前得到的结果如下。

```text
\mathrm{feature\_dim} = 18
```

```text
\mathrm{token\_dim} = 512
```

```text
\mathrm{linear\_probe\_mse} = 3.391 \times 10^{-6}
```

```text
\mathrm{mean\_baseline\_mse} = 7.468 \times 10^{-6}
```

```text
\mathrm{linear\_probe\_gain} = 2.20
```

```text
\mathrm{knn\_future\_mse} = 3.347 \times 10^{-6}
```

```text
\mathrm{global\_adjacent\_pair\_mse} = 1.504 \times 10^{-5}
```

```text
\mathrm{exact\_duplicate\_groups} = 1
```

```text
\mathrm{exact\_duplicate\_target\_mse} = 4.099 \times 10^{-7}
```

## 11. 如何解释这些数值

第一，线性 probe 和 kNN 的误差都显著低于均值基线，这说明当前输入特征中确实包含 future token 的可恢复信息。

第二，linear probe gain = 2.20，表示线性 probe 比永远输出平均 token 至少好约 2.2 倍，这是一个非常明确的正信号。

第三，global adjacent pair 的自然变化量级明显大于 probe 误差，说明 predictor 不是在随机拟合，而是在抓真实结构。

第四，几乎没有发现“相同输入对应完全不同 token”的重复组；那唯一一组重复输入，其 target 差异也非常小，说明没有观察到严重的一对多崩坏。

因此，这些数值支持的结论不是‘理论上绝对单值’，而是‘在当前数据分布和 token 定义下，这张映射是足够可学习、足够可部署的’。

## 12. 训练阶段与推理阶段如何表述

训练阶段：teacher 仍然从未来信息构造 c_t^\star；predictor 从当前信息预测 \hat c_t；ACT 同时优化 base path 和 predicted-conditioned path。

推理阶段：只需要当前输入 x_t = (o_t, q_t)，先预测 \hat c_t，再用它走 conditioned ACT 推理。全程不需要 future image、teacher、future action，也不需要第二次 ACT forward。

```text
\hat c_t = g_\phi([\mathrm{pool}(\Pi(f_{act}(o_t))); q_t])
```

```text
\hat a_{t:t+H-1}^{pred} = \pi_{ACT}(o_t, q_t; \hat c_t)
```

## 13. 最终结论

这套方法的本质，是把原来‘训练时才能拿到的 future condition token’，改造成‘当前时刻可预测、推理时可直接部署的 condition token’。

从目前的 probe 结果看，这条路线在统计上是可行的；从工程实现上看，它避免了两次推理；从算法设计上看，它把 teacher token 从部署输入改成了监督目标，因此训练图和推理图终于统一了。
