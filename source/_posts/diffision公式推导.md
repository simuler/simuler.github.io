---
title: diffision推导
date: 2026-02-15 12:45:00
tags:
  - diffision
---

# 从vae、ddpm、ddim、diffision到flow matching公式推导
最近学习diffision model，整体的思路大概了解了，但是没有详细推导过其原理，觉得理解还是不够深入，因此进行一下完整的推导过程和相关理论知识学习。
# VAE
对于vae的理解在于，是一个将原图像通过encoder加噪到高斯分布，再通过decoder从噪声中一步恢复的过程。其并不是严格意义上的生成模型，因为它的目标是恢复出原图像，而不是生成新的图像。
