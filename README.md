# TinyTorch Starter — Java

TinyTorch 课程起始代码（Java）——从零实现一个教学版 PyTorch。

## 结构

```
src/main/java/dev/tensorhero/tinytorch/   # 核心实现（Tensor, Function, Layer 等）
src/main/java/dev/tensorhero/tinynum/     # TinyNum 依赖
tests/                                     # 每个 stage 一个测试 (TestE01 … TestE10)
tensorhero.yml                             # 课程元数据
pom.xml                                    # Maven 配置
```

## 开始

在核心文件中找到 `TODO` 注释，按 stage 顺序逐步实现。

查看 [tinytorch-tester](https://github.com/tensorhero/tinytorch-tester) 了解如何测试你的代码。
