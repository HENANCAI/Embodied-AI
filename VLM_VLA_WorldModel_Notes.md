# 具身智能三基石：VLM、VLA 与世界模型

> 个人学习笔记 · 2026-08-10
>
> 从理解到行动：感知当下 → 预测未来 → 做出动作

---

## 目录

1. [VLM：视觉语言模型](#1-vlm视觉语言模型)
2. [VLA：视觉语言动作模型](#2-vla视觉语言动作模型)
3. [世界模型 World Model](#3-世界模型-world-model)
4. [三者关系：感知—预测—行动闭环](#4-三者关系感知预测行动闭环)
5. [通往 VLA 研究的学习路线](#5-通往-vla-研究的学习路线)
6. [参考文献](#6-参考文献)

---

## 1. VLM：视觉语言模型

**VLM（Vision-Language Model，视觉语言模型）** 是同时处理图像与文本的多模态模型，核心能力是**理解**——看懂图像内容并用语言表达出来。

### 核心能力

| 能力 | 说明 | 示例 |
|------|------|------|
| 图像理解 | 识别图像中的物体、场景、关系 | "图中有几只猫？" |
| 视觉问答 VQA | 基于图像内容回答自然语言问题 | "这个人拿的是什么？" |
| 图像描述 | 生成自然语言描述 | "一个男孩在公园踢足球" |
| 多模态推理 | 结合视觉与语言做逻辑推断 | "桌上有苹果和刀，苹果被切开了吗？" |

### 典型架构

```
图像 ──→ 视觉编码器（ViT/CLIP）──┐
                                ├──→ 多模态融合 ──→ 语言模型（LLM）──→ 文本输出
文本 ──→ 文本编码器 ────────────┘
```

- 视觉编码器：如 ViT、SigLIP，把图像映射为特征向量
- 投影层：对齐视觉与语言特征空间
- 语言模型：如 Qwen、LLaMA 底座，生成回答

### 代表模型

- **闭源**：GPT-4V / GPT-4o、Gemini、Claude（多模态版本）
- **开源**：Qwen-VL、InternVL、LLaVA、DeepSeek-VL
- **技术源头**：CLIP（2021，图文对齐）、BLIP、Flamingo（2022）

### 关键点

- VLM 的输入是「图像 + 文本」，输出是**文本**
- VLM 解决的是"**看懂并说清楚**"，不产生任何物理动作
- 训练数据是海量「图像-文本对」，无需机器人数据

---

## 2. VLA：视觉语言动作模型

**VLA（Vision-Language-Action Model，视觉语言动作模型）** 在 VLM 基础上进一步输出**动作**，把感知、推理与执行整合进一个模型，是具身智能（Embodied AI）的核心范式。

### 核心能力

| 能力 | 说明 | 示例 |
|------|------|------|
| 指令跟随 | 根据语言指令规划动作 | "把红色杯子放到托盘上" |
| 场景理解 | 理解当前环境状态 | 识别物体位置、抓取点 |
| 动作生成 | 输出低层控制指令 | 关节角、末端位姿、夹爪开合 |
| 泛化 | 迁移到未见过的物体/场景 | 新杯子也能抓 |

### 典型架构

```
图像 + 文本指令 + 本体状态
        │
        ▼
   VLM 底座（视觉语言编码）
        │
        ▼
   Action Head（动作头 / 策略头）
        │
        ▼
   动作指令（关节角度 / 末端轨迹 / 夹爪开合）
```

### 动作输出的两代技术路线

| 路线 | 代表 | 原理 | 特点 |
|------|------|------|------|
| **token 化动作** | RT-2（Google） | 把动作离散成 token，像生成文字一样生成动作 | 直接复用 LLM 架构；精度受限 |
| **连续动作头** | OpenVLA、π0（Physical Intelligence） | VLM 后接专门的 action head，直接输出连续控制信号 | 精度更高，当前主流方向 |

### 代表模型

- **RT-1 / RT-2**（Google，2022-2023）：开山之作，从"互联网预训练 + 机器人微调"出发
- **OpenVLA**（Stanford，2023）：开源 7B VLA，社区广泛使用
- **π0（pi-zero）**（Physical Intelligence，2024）：flow matching 动作头，灵巧操作
- **GR00T N1**（NVIDIA，2025）：人形机器人通用基础模型
- **Figure Helix**、**智元 GO-1** 等：工业界落地探索

### 关键点

- VLA 的输入是「图像 + 文本 + 本体状态」，输出是**动作指令**
- VLA 训练需要**机器人操作数据**（遥操作采集），数据成本高——这是落地的核心瓶颈
- VLA 解决"**想好了，怎么做**"

---

## 3. 世界模型 World Model

**世界模型（World Model）** 是让 AI 在"大脑"里建立一个关于外部世界的**内部模拟器**——学会预测"如果我做了动作 A，世界会变成什么样"，从而在行动前先在脑子里推演后果。

### 核心思想

世界模型源自认知科学：人类不需要真的碰火才知道火会烫，因为大脑里有"心理模拟器"。AI 世界模型把它形式化为学习环境的**转移函数（transition function）**：

```
世界模型：(s_t, a_t) → s_{t+1}
```

给定当前状态和动作，预测下一个状态。有了它，AI 可以在虚拟的"内心世界"里试错、规划、想象。

### 三条演进路线

| 演进阶段 | 代表工作 | 形态 | 核心贡献 |
|---------|---------|------|---------|
| ① 认知科学源头 | Tolman（1948）"认知地图" | 概念理论 | 提出脑中存在环境内部表征 |
| ② RL·隐空间模型 | Ha & Schmidhuber（2018）《World Models》、Dreamer V1/V2/V3 | 小模型，潜空间预测 | 在压缩潜在空间"做梦"训练策略，样本效率大增 |
| ③ 生成式世界模型 | Sora、Genie 2、UniSim、NVIDIA Cosmos | 大模型，像素级视频生成 | 把"预测"升级为"生成未来视频" |

> 当前讨论的"世界模型"绝大多数指第 ③ 代——尤其 Sora 引爆的"视频生成即世界模拟"路线。OpenAI 官方对 Sora 的定义即"学习如何理解和模拟现实世界"。

### 三大用途

1. **规划 Planning**：在内心模拟器里试一万次，选最优动作再真实执行
2. **样本效率 Sample Efficiency**：Dreamer 系列证明在"梦"里训练策略可大幅减少真实交互
3. **想象与泛化 Imagination**：生成从未见过的场景组合，体现组合式推理

### 挑战与争议

- **"视频预测 ≠ 世界模型"**：Sora 生成的下一个画面好看，不等于掌握了物理定律（因果 vs 相关性之争）
- **评估难题**：如何判断世界模型预测是否准确？缺乏统一基准
- **幻觉问题**：会生成物理上不可能的未来（如杯子穿过桌子）

---

## 4. 三者关系：感知—预测—行动闭环

```
        VLM                    世界模型                    VLA
   理解当下 (这是什么)  →  预测未来 (会怎样)  →  做出动作 (怎么做)
        │                       │                        │
   多模态理解             内部模拟 / 想象            动作生成 / 执行
```

| 维度 | VLM | VLA | 世界模型 |
|------|-----|-----|---------|
| 全称 | Vision-Language Model | Vision-Language-Action Model | World Model |
| 核心任务 | 理解 | 理解 + 行动 | 预测 |
| 输入 | 图像 + 文本 | 图像 + 文本 + 本体状态 | 状态 + 动作 |
| 输出 | 文本 | 动作指令 | 未来状态 / 视频 |
| 应用 | 多模态对话、OCR、VQA | 机器人操控、人形机器人 | 规划、仿真、数据生成 |
| 代表 | GPT-4V、Qwen-VL、InternVL | RT-2、OpenVLA、π0、GR00T | Dreamer、Sora、Genie 2 |

**一句话记忆**：

- VLM 是"眼睛 + 大脑的语言区"
- VLA 是"眼睛 + 大脑 + 运动皮层"
- 世界模型是"大脑中的物理模拟器"

**VLA = VLM + 动作输出**；VLA 解决"手怎么动"，世界模型解决"动完之后会怎样"，两者互补——这也是 Figure、Physical Intelligence（π0）、NVIDIA GR00T 都在给 VLA 配世界模型的原因。

---

## 5. 通往 VLA 研究的学习路线

（面向机器人背景、目标进入具身智能研究领域的学习者）

### 五阶段路线图

| 阶段 | 时间 | 核心任务 | 说明 |
|------|------|---------|------|
| 0. 回血期 | 第 1-2 月 | 重建 Python/PyTorch 手感，跑入门 demo | 更新博客与简历 |
| 1. 知识补齐 | 第 2-4 月 | RL（CS285）+ Diffusion Policy + Dreamer V3 | 世界模型偏理论，无需大算力 |
| 2. 轻量复现 | 第 4-6 月 | LeRobot 小任务 → OpenVLA 云 GPU 微调 | 目标：跑通 + 能改 |
| 3. 做出作品 | 第 6-9 月 | 选"世界模型 + VLA"小切口题目，开源 + 投 workshop | 作品是敲门砖 |
| 4. 定路径冲刺 | 第 9-12 月 | 投研究岗 / 研究实习 / PhD 套磁 | 用作品反推路径 |

### 必读论文清单

**VLA 主线**：RT-1 → RT-2 → OpenVLA → π0 → GR00T N1

**世界模型主线**：Ha & Schmidhuber《World Models》(2018) → Dreamer V3 → Sora / Genie 2

**基础主线**：Transformer → CLIP → LLaVA → Diffusion Policy（Chi et al., 2023）

### 工具与资源

| 类别 | 资源 |
|------|------|
| 课程 | Berkeley CS285（强化学习）、CS231n（视觉）、HuggingFace VLA 教程 |
| 代码库 | HuggingFace LeRobot、OpenVLA、diffusion_policy |
| 算力 | 本地 CPU 开发 + AutoDL/Colab 按小时租云 GPU |
| 会议 | CoRL、ICRA、RSS、ICLR/NeurIPS Workshop |

---

## 6. 参考文献

- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021
- Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*, 2022
- Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, 2023
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024
- Ha & Schmidhuber. *World Models*, 2018
- Hafner et al. *Mastering Diverse Domains through World Models* (Dreamer V3), 2023
- Brooks et al. *Video Generation Models as World Simulators* (Sora), 2024
- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, 2023

---

*本文档为个人学习笔记，随学习进度持续更新。*
