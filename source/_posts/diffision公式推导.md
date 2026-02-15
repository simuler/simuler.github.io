---
title: diffision推导
date: 2026-02-15 12:45:00
tags:
  - diffision
---

# 从vae、ddpm、ddim、diffision到flow matching公式推导
最近学习diffision model，整体的思路大概了解了，但是没有详细推导过其原理，觉得理解还是不够深入，因此进行一下完整的推导过程和相关理论知识学习。

# AE
对于ae的理解在于，是一个将原图像通过encoder压缩为低维信息，再通过decoder从信息恢复的过程。其并不是严格意义上的生成模型，因为它的目标是恢复出原来的输入，而不是生成新的图像，只是对信息进行了压缩和重建，从损失函数中也可以看出，只是计算了预测值和原始值的mse。
