---
title: diffision推导
date: 2026-02-15 12:45:00
tags:
  - diffision
math: true
---

# 从vae、ddpm、ddim、diffision到flow matching公式推导
最近学习diffision model，整体的思路大概了解了，但是没有详细推导过其原理，觉得理解还是不够深入，因此进行一下完整的推导过程和相关理论知识学习。

# AE
对于ae的理解在于，是一个将原图像通过encoder压缩为低维信息，再通过decoder从信息恢复的过程。其并不是严格意义上的生成模型，因为它的目标是恢复出原来的输入，而不是生成新的图像，只是对信息进行了压缩和重建，从损失函数中也可以看出，只是计算了预测值和原始值的mse。

# VAE（Variational Auto-Encoding）

[讲了为什么要用变分推断](https://zhuanlan.zhihu.com/p/355019238)

[讲了vae gan flow推导](https://zhuanlan.zhihu.com/p/721196823)

1. 什么是生成问题？

  在生成问题中，我们目标是希望基于当前的样本，学习到一种数据分布，从这个数据分布中，能够采样得到不同的样本。那么有两种方式去得到这个数据分布：
  - 最大似然  
  学习到由参数$\theta$表示的数据分布，在给定数据集上，采样得到$x$的概率最大。

      $\theta=argmax(p(x|\theta))=argmax\prod_{i=1}^{n}p(x_{i}|\theta)=argmax\sum_{i=1}^{n}logp(x_{i}|\theta)$
  - 条件概率

      ${p(\theta|x)}=\frac{p(x,\theta)}{ p(x)}=\frac{p(x|\theta)p(\theta)}{p(x)}
      =\frac{\prod_{i=1}^{n}p(x_{i}|\theta)p(\theta)}{\int{\prod_{i=1}^{n}p(x_{i}|\theta)p(\theta)d\theta}}$

  最大似然估计存在几个问题：
  - 样本中未出现的情况会将概率估计为0，没有泛化能力
  - 无法利用先验知识
  
  因此，我们一般采样条件概率的方式求解数据分布：
  - 一种估计条件概率的方法是变分推断
  - 一种是蒙特卡洛
  
  变分推断的思想是用一个简单的分布$q(\theta)$拟合$p(\theta|x)$,并通过kl散度加以约束，求解$q(\theta)$相当于:
      $minKL(q(\theta)|p(\theta|x)) = min\int{q(\theta)\frac{q(\theta)}{p(\theta|x)}d\theta}$
  
  2. VAE是如何解决这个生成问题的

   假设我们有一堆$3*256*256$大小的样本图像$x_i$,这些样本是iid的，正常情况下我们希望能直接得到$196608$维度的分布，并直接从这里面采样得到图片，但是这么高维的数据中大部分采样得到的都是噪声。

   考虑到原始图像的信息是冗余的，我们能不能将图像进行压缩，得到低维的隐变量，隐变量与采样图像是对应的（不一定一一对应），通过从隐变量中解码得到图像。

   假设这些隐变量都是从一个已知的简单分布$z$采样得到的,那么我们要优化的目标就是最大化所有从$z$中采样$x$的边缘概率之积：

   $max\prod_{i=1}^{i=n}p(x_i)
   = max\sum_{i=1}^{i=n}logp(x_i)
   = max\sum_{i=1}^{i=n}log\int{p(x|z)p(z)dz}$
   
  - 规范以下定义, $x\to z\to x'$
    - $x$: 样本数据
    - $z$: 隐变量
    - $x'$: 生成的样本
    - $p(z|x)$: 编码器的真实分布
    - $q_\phi(z|x)$: 编码器的近似分布 
    - $p(z)$: 隐变量的分布
    - $p(x|z)$: 解码器的真实分布
    
  - 注意：$p(x)、p(x')$我们是不知道的

   在VAE中，我们假设隐变量的分布是高斯分布，即$p(z)=\mathcal{N}(0, I)$,解码器可以表示为$p_\theta(x|z)$（因为解码器的目标我们是知道的，但是不知道目标的分布，所以可以直接训练），这样我们如果能训练得到这个解码器，就可以从高斯分布中采样，然后解码得到图像了。
   
   为了保证采样得到的隐变量能够得到有用的图像，而不是噪声，采样时的隐变量其实是从图像$x$编码得到的，编码之后的隐变量是满足我们之前设置的高斯分布的，那我们如何保证所有的$x$经过编码器后得到的隐变量是满足分布$p(z)$呢？这样我们需要求解$p(z|x)$（但是这个无法直接求解，后面会说原因），我们只能假设一个近似分布$q_\phi(z|x)$，让编码器最终输出的是一个分布，包括均值$\mu$和方差$\sigma$,我们希望这个近似分布和真实分布$p(z|x)$是一致的,这也引出了我们的优化目标：

   $\begin{aligned}
   & KL(q_\phi(z|x)|p(z|x)) \\\\
   & =\int{q_{\phi}(z|x)log{\frac{q_{\phi}(z|x)}{p(z|x)}}}dz  \\\\
   & =E_{z\sim q_{\phi}(z|x)}[log{\frac{q_{\phi}(z|x)}{p(z|x)}}]  \\\\
   & =E_{z\sim q_{\phi}(z|x)}[log{q_{\phi}(z|x)} - log{p(z|x)}]   \\\\
   & =E_{z\sim q_{\phi}(z|x)}[log{q_{\phi}(z|x)} - log{p(z|x)}]   \\\\
   & =E_{z\sim q_{\phi}(z|x)}[log{q_{\phi}(z|x)} - log{\frac{p(x|z)p(z)}{p(x)}}]  \\\\
   & =E_{z\sim q_{\phi}(z|x)}[log{q_{\phi}(z|x)} - log{p(x|z)}-log{p(z)} + log{p(x)}]  \\\\
   & =KL(q_\phi(z|x)|p(z)) - E_{z\sim q_{\phi}(z|x)}[log{p(x|z)} - log{p(x)}]
   \end{aligned}
   $ 

   其中
   $\begin{aligned}
   & logp(x) = KL(q_\phi(z|x)|p(z|x)) - KL(q_\phi(z|x)|p(z)) + E_{z\sim q_{\phi}(z|x)}[log{p(x|z)}] \\\\
   & \mathcal{L} = -KL(q_\phi(z|x)|p(z)) + E_{z\sim q_{\phi}(z|x)}[log{p(x|z)}] \\\\
   & \mathcal{L} = E_{z\sim q_{\phi}(z|x)}[log{p(x,z)} - log{p(x|z)}]
   \end{aligned}
   $

  - $\int{q_{\phi}(z|x)log{\frac{q_{\phi}(z|x)}{p(z|x)}}}dz \\\\
   =E_{z\sim q_{\phi}(z|x)}[log{\frac{q_{\phi}(z|x)}{p(z|x)}}]$
      - 这一步积分变均值，需要满足
          - 非负性：$q_{\phi}(z|x) >= 0$
          - $\int{q_{\phi}(z|x)}dz=1$
   
   $\mathcal{L}$就是ELBO，有两种形式，我们目标是最大化$logp(x)$,其中$KL(q_\phi(z|x)|p(z|x))$是大于0的，所以可以将ELBO看作$logp(x)$的下界，我们可以通过对$\phi$求梯度的方式优化ELBO。

   求梯度前，我们先看看如何计算ELBO，直接计算肯定是不行的，我们只能用蒙特卡洛算法，通过采样的方式计算ELBO。下面这个式子肯定是能通过采样的方式计算均值的，但是需要先将$x$输入编码器，得到隐变量$z$,然后经过解码器，得到$log{p(x|z)}$,可以看到梯度是断的，我们无法直接优化编码器。这里引入重参数化，其思路在于z本来是从编码器的输出$\mu$和$\sigma$中采样得到的，这里我们引入一个标准正态分布$\varepsilon$,通过$z=\mu + \sigma\varepsilon$得到隐变量，这样整个优化过程就是连续的了：

   $E_{z\sim q_{\phi}(z|x)}[log{p(x|z)}] = E_{\varepsilon\sim p(\varepsilon)}[x|g_\phi(\varepsilon，x)]$

   然后我们通过蒙特卡洛法计算ELBO:

   $\begin{aligned}
   & \mathcal{L}(\theta,\phi;x^{i})  \\\\
   & = -KL(q_\phi(z|x^{i})|p_\theta(z)) + E_{z\sim q_{\phi}(z|x^{i})}[log{p_{\theta}(x^{i}|g_\phi(\varepsilon，x^{i}))}] \\\\
   & = -KL(q_\phi(z|x^{i})|p_\theta(z)) + \frac{1}{L}\sum_{l=1}^{L}logp_\theta{x^{(i)}|g_\phi(\varepsilon^{l}，x^{i})}
   \end{aligned}
   $
   注意，上述的推导最终是以$p(x)$为优化目标，所以需要最大化ELBO，在计算loss时，即最小化-ELBO。其中KL散度直接计算即可，logp(x|z)即计算bce误差或mse误差
  
  
# DDPM（Denoising Diffusion Probabilistic Models）
VAE的本质是将图像压缩到隐空间，然后从隐空间进行采样。ddpm中不考虑隐空间的问题，而是根据非平衡热力学的思想，将原图像加噪为完全随机的噪声，然后逐步去掉噪声，还原图像。
基于ddpm的思想，自然有两个问题：
- 加噪时应该如何加，才能将最终的图像加为完全的噪声
- 基于完全随机的噪声图像，如何去噪还原图像
## 1. 前向加噪过程
假设原图是$x_0$,加噪过程满足$q(x_t|x_{t-1})=\mathcal{N}(x_t；\sqrt{1-\beta_t}x_{t-1}，\beta_tI)$,则:

$x_t = \sqrt{a_t}x_{t-1} + \sqrt{\beta_t}\varepsilon_t,a_t=1-\beta_t$

这里有个问题，为什么加噪过程要这么设置？先继续往下推：
$\begin{aligned}
x_t &= \sqrt{a_t}x_{t-1} + \sqrt{\beta_t}\varepsilon_t \\\\
    &= \sqrt{a_t}(\sqrt{a_{t-1}}x_{t-2} + \sqrt{\beta_{t-1}}\varepsilon_{t-1}) + \sqrt{\beta_t}\varepsilon_t  \\\\
    &= \sqrt{a_{t}a_{t-1}..a_{1}}x_{0} + \sqrt{a_{t}a_{t-1}..a_{2}\beta_{1}}\varepsilon_{1} + ... \sqrt{a_{t}\beta_{t-1}}\varepsilon_{t-1} + \sqrt{\beta_t}\varepsilon_t 
\end{aligned}
$

因为$\sqrt{a_{t}a_{t-1}..a_{2}\beta_{1}}\varepsilon_{1} + ... \sqrt{a_{t}\beta_{t-1}}\varepsilon_{t-1} + \sqrt{\beta_t}\varepsilon_t $都是标准正态分布，所以整体的方差为$1-a_{t}...a_{1}$,推导如下：

$\begin{aligned}
& a_{t}a_{t-1}..a_{1} + a_{t}a_{t-1}..a_{2}\beta_{1} + ...+a_{t}\beta_{t-1}+\beta_{t}  \\\\
& = a_{t}a_{t-1}..a_{2}(a_1 + \beta_1) + ...  \\\\
& = a_{t}a_{t-1}..a_{3}(a_2 + \beta_2) + ...  \\\\
& = a_t + \beta_1 \\\\
& = 1
\end{aligned}
$

$q(x_{t}|x_0)=\sqrt{a_{t}a_{t-1}...a_1}x_0 + \sqrt{1-a_{t}a_{t-1}...a_1}\varepsilon=\mathcal{N}(x_t;\sqrt{\bar{a_{t}}}x_0;(1-\bar{a_{t}})I) \\\\
其中\bar{a_T}=\prod_{t=1}^{t=T}a_t
$

从上面的公式能看出来最终的$x_t$均值为$\sqrt{\bar{a_{t}}}x_0$,而$\beta$我们一般设置为很小的正数$[1e-4,2e-2]$,当$t\to\infty$时，$x_t$的均值基本趋近于0了，所以我们可以将其视作纯噪声。

## 反向过程
在反向过程中，我们希望在每个时间步t拟合的策略应该是：$p_\theta(x_{t-1}|x_{t})=\mathcal{N}(x_{t-1};\mu_\theta(x_t, t),\varepsilon_\theta(x_t,t))$,其中假设方差项$\varepsilon_\theta(x_t,t)=\delta_{t}^{2}=\beta_{t}$,下面推导去噪过程。

$\begin{aligned}
q(x_{t-1}|x_t，x_0) &=\frac{q(x_t|x_{t-1})q(x_{t-1})}{q(x_t)}  \\\\
               &=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}
\end{aligned}
$

其中

$\begin{aligned}
& q(x_t|x_{t-1},x_0)\sim\mathcal{N}(x_t;\sqrt{a_t}x_{t-1},\beta_tI);\mu=\sqrt{a_t},\delta^{2}=\beta_{t} \\\\
& q(x_{t-1}|x_0)\sim\mathcal{N}(x_{t-1};\sqrt{a_{t-1}...a_{1}}x_{0},(1-a_{t-1}...a_1)I); \mu=\sqrt{{\bar{a}_{t-1}}},\delta^{2}=1-\bar{a}_{t-1} \\\\
& q(x_{t}|x_0)\sim\mathcal{N}(x_{t};\sqrt{a_{t}...a_{1}}x_{0},(1-a_{t}...a_1)I); \mu=\sqrt{{\bar{a}_{t}}},\delta^{2}=1-\bar{a}_{t}
\end{aligned}
$

然后我们可以计算这几个高斯分布：

$\begin{aligned}
q(x_{t-1}|x_t，x_0) &= exp({-\frac{1}{2}(\frac{(x_t-\sqrt{a_t}x_{t-1})^{2}}{\beta_t} + \frac{(x_{t-1}-\sqrt{\bar{a}_{t-1}}x_0)^2}{1-\bar{a}_{t-1}} - \frac{(x_t-\sqrt{\bar{a}_{t}}x_0)^2}{1-\bar{a}_{t}})}) \\\\
& = exp({-\frac{1}{2}(\frac{(x_t^2-2\sqrt{a_t}x_{t}x_{t-1} + a_tx_{t-1}^2)}{\beta_t} + \frac{(x_{t-1}^2-2\sqrt{\bar{a}_{t-1}}x_{t-1}x_0 + \bar{a}_{t-1}x_0^2)}{1-\bar{a}_{t-1}} - \frac{(x_t^2-2\sqrt{\bar{a}_{t}}x_{t}x_{0} + \bar{a}_{t}x_{0}^2)}{1-\bar{a}_{t}})}) \\\\
& = exp(-\frac{1}{2}((\frac{a_t}{\beta_t} + \frac{1}{1-\bar{a}_{t-1}})x_{t-1}^2 - (\frac{2\sqrt{a_t}}{\beta_t}x_t + \frac{2{\bar{a}_{t-1}}}{1-\bar{a}_{t-1}}x_0)x_{t-1} + c(x_t, x_0)))
\end{aligned}
$

我们从另一个角度看$q(x_{t-1})$,$q(x_{t-1})$也是一个高斯分布，可以表示为$q(x_{t-1})=\frac{1}{\sqrt{2\pi\delta^2}}exp(-\frac{1}{2}(\frac{(x_{t-1}-\mu_{t-1})^2}{\delta^2}))$,跟上面的公式一一对应下，得到：
$\frac{1}{\delta^2}=\frac{a_t}{\beta_t} + \frac{1}{1-\bar{a}_{t-1}},-2\mu=\frac{- (\frac{2\sqrt{a_t}}{\beta_t}x_t + \frac{2{\bar{a}_{t-1}}}{1-\bar{a}_{t-1}}x_0)}{\frac{a_t}{\beta_t} + \frac{1}{1-\bar{a}_{t-1}}} \to  \delta^2 = \frac{1-\bar{a}_{t-1}}{1-\bar{a}_t}\beta_t, \mu = \frac{\sqrt{a}_t(1-\bar{a}_{t-1})}{1-\bar{a}_t}x_t + \frac{\sqrt{\bar{a}_{t-1}}\beta_t}{1-\bar{a}_t}x_0$

由前向推导时$x_t和x_0$的关系可以得到：$x_0 = \frac{1}{\sqrt{\bar{a}_t}}(x_t - \sqrt{1-\bar{a}_t}\varepsilon_t)$,将其带入上面的公式，即可得到：

$\begin{aligned}
x_{t-1} &= \frac{\sqrt{a}_t(1-\bar{a}_{t-1})}{1-\bar{a}_t}x_t + \frac{\sqrt{\bar{a}_{t-1}}\beta_t}{1-\bar{a}_t}x_0 + \sqrt{\frac{1-\bar{a}_{t-1}}{1-\bar{a}_t}\beta_t}z  \\\\
& = \frac{\sqrt{a}_t(1-\bar{a}_{t-1})}{1-\bar{a}_t}x_t + \frac{\sqrt{\bar{a}_{t-1}}\beta_t}{1-\bar{a}_t}{\frac{1}{\sqrt{\bar{a}_t}}(x_t - \sqrt{1-\bar{a}_t}\varepsilon_t)} + \sqrt{\frac{1-\bar{a}_{t-1}}{1-\bar{a}_t}\beta_t}z  \\\\
&= \frac{1}{\sqrt{a}_t}(x_t - \frac{\beta_t}{\sqrt{1-\bar{a}_t}}\varepsilon_{x_t,t}) + \delta_t z
\end{aligned}
$

这样就建立了只有$x_t 和 x_{t-1}$的联系，其中的$\varepsilon$就是我们要学习的参数。目前为止已经推导出了前向和反向过程的所有需要的目标：
- 前向时没有需要学习的参数，直接设置$\beta$，然后可以一步得到$x_t$
- 反向时，向网络输入$(x_t,t)$，预测$\varepsilon$

但是为什么通过预测噪声能够最终还原出原图像，并且有泛化性呢？