# 钱张枫 Profile

## 基本信息

- 姓名：钱张枫
- 英文名或常用名：zfqian
- GitHub ID：`MAK1MAAaa`
- 常用语言：中文、英语（可阅读论文和技术文档）
- 公开身份：复旦大学人工智能专业2026级硕士生

## 个人简介

我本科就读于上海电力大学计算机科学与技术专业，即将进入复旦大学人工智能专业攻读硕士。目前主要关注计算机视觉、智能系统工程与模型推理优化，已独立完成电力巡检图像智能检测系统、Rasa 电商智能客服系统和 Bilibili 视频用户留存分析等公开项目。其中，图像检测项目打通了 YOLO 训练、FastAPI 服务、C++ TensorRT/CUDA 加速、Vue 3 前端与 Docker 部署链路。我习惯将问题拆解为可运行、可评测的模块，并通过实验记录和技术文档复盘结果。希望借 OpenMOSS 暑期集训继续补齐模型训练、系统优化、数据治理与可靠评测能力，形成更扎实、可复现的研究与工程实践。

## 学习与研究

### 最近关注的问题

- 如何在合法合规并获得必要授权的前提下，通过协议分析、系统行为观测和兼容性实现，开发更轻量、透明、可控的 aTrust 替代客户端，减少原客户端在个人设备上的侵入性改动、后台驻留和资源占用。
- 是否可以在仿真或自建环境中，结合职业比赛全局回放、模仿学习与强化学习训练《王者荣耀》智能体，并进一步学习宏观决策、阵容协同、资源运营和局部战术等更细粒度的打法。

### 过去探索过的方向

- 使用 NapCat 将个人 QQ 账号接入 AstrBot，搭建聊天机器人基础运行链路，并尝试参考 HarukiBot 等开源机器人项目继续扩展功能。当前主要处于功能探索和原型实践阶段。

### 项目与学习经历

#### 电力巡检图像智能检测系统

- 时间：2026.01
- 项目性质：个人独立完成
- 背景与问题：面向电力巡检图像中的设备与缺陷检测，构建兼顾模型训练、业务接口、交互展示和高性能部署的端到端系统。
- 个人工作：独立完成数据集拆分与 YOLO 训练环境、FastAPI 多模型推理服务、C++ TensorRT 推理引擎、pybind11 Python 绑定、Vue 3 检测工作站以及 Docker Compose GPU 部署链路。
- 方法与结果：实现 PyTorch 模型到 ONNX、TensorRT 的转换，支持 FP32/FP16/INT8 推理、`IInt8EntropyCalibrator2` 校准和动态 Batch 1–16；编写 CUDA Kernel 融合 YOLO 边界框解码与置信度过滤，公开文档记录 Device-to-Host 传输开销减少约 90%。系统同时支持批量检测、模型切换、PyTorch/TensorRT 对比、误检漏检反馈和本地历史记录。
- 局限与反思：公开仓库尚未提供固定硬件、固定数据集下的统一准确率、吞吐量和端到端延迟报告；后续需要系统量化 FP16/INT8 精度损失，并补充不同 Batch 和并发条件下的性能基准。
- 公开链接：https://github.com/oroiteS/image_detection

#### Rasa 电商智能客服系统

- 时间：2026.03 - 2026.06
- 项目性质：个人独立完成
- 背景与问题：探索规则式对话系统、通用大模型与 LoRA 模型在电商客服场景中的协同方式，并建立可重复的系统评测流程。
- 个人工作：独立完成 Vue 3 前端、FastAPI 后端、Rasa 助手、PostgreSQL/Redis 数据层、LoRA 推理链路和独立 benchmark，实现商品与订单接口、聊天路由、服务端记忆、知识库和附件处理等模块。
- 方法与结果：将评测划分为 `shared_core` 与 `agent_extension` 两类能力，按去重样本统计通过率、覆盖率与失败原因，并记录提示词文件及哈希，形成从数据集构建、状态重置、执行到中文报告生成的完整流程。
- 局限与反思：当前正式 benchmark 的并发固定为 1，结论主要适用于现有数据集和本地实验环境；后续需要补充高并发、跨数据分布和长期运行稳定性验证。
- 公开链接：https://github.com/MAK1MAAaa/Rasa-EC-bot

#### Bilibili 视频用户留存分析

- 时间：2024.12 - 2025.06（2026.03 整理公开仓库）
- 项目性质：个人独立完成
- 背景与问题：作为大数据管理与存储课程实践，尝试从公开视频和评论数据中分析互动指标与积极情绪评论之间的关系。
- 个人工作：独立实现视频列表与评论抓取、WBI 请求签名、JSON 数据落盘、MongoDB 导入、MySQL 结构化存储、情感分析及可视化分析流程。
- 方法与结果：项目使用 Requests、Selenium、PyMongo、SQLAlchemy、pandas、SnowNLP、Seaborn 等工具串联数据获取、清洗、特征计算和图表生成，并为主要抓取与分析模块保留测试脚本。
- 局限与反思：项目依赖外部接口、登录态和本地数据库环境，公开结果不足以支持因果结论；后续需要进一步加强凭据管理、异常恢复、数据合规与可复现部署。
- 公开链接：https://github.com/MAK1MAAaa/Retention

## CS336 学习计划

- 当前基础：具备模型训练与推理部署、智能应用开发、数据处理和 benchmark 实践，需要补齐语言模型训练、分布式系统、数据治理和后训练能力。
- A1 Basics：从头实现 Tokenizer、Transformer、优化器和最小语言模型，重点检查边界条件、数值稳定性和训练正确性。
- A2 Systems：使用 profiler 分析性能瓶颈，完成 Triton FlashAttention2 和内存高效的分布式训练，重点关注吞吐量、显存与通信开销。
- A3 Scaling：拟合 Scaling Law，分析模型规模、数据量、计算成本和 Loss 的关系，并建立模型选型与推理成本判断能力。
- A4 Data：完成 Common Crawl 数据处理、质量过滤和去重，重点检查数据质量、重复污染和处理前后的分布变化。
- A5 Alignment：完成 SFT 与 RLVR/GRPO 实践，重点分析奖励设计、reward hacking、训练稳定性和泛化表现，并在时间允许时学习 RLHF 与 DPO。
- A6 Harness：待题目发布后确定

## 技能与工具

- 编程与框架：熟悉 Python；使用过 C++17、CUDA、TensorRT、pybind11、PyTorch、Ultralytics YOLO 和 OpenCV；具备 FastAPI、Rasa、Vue 3、TypeScript 与 Tailwind CSS 项目实践。
- 工程工具：使用 Git/GitHub、Linux/WSL、uv、pytest、CMake、Docker 和 Docker Compose；使用过 ONNX、PostgreSQL、Redis、MongoDB、MySQL、SQLAlchemy，并能配置 GPU 推理、Ollama、vLLM 与 OpenAI-compatible 模型服务。
- 其他能力：端到端系统设计、模型训练与部署、FP16/INT8 量化和推理性能优化、benchmark 构建、数据抓取与清洗、数据分析与可视化、技术文档编写。

## 特长、爱好与日常

### 特长

- 跨语言全链路工程：能够将 Python 模型与服务、C++/CUDA 高性能模块、前端界面、数据库和容器化部署组织成完整系统。
- 推理优化与工程验证：具备模型格式转换、混合精度与 INT8 量化、动态 Batch、CUDA 后处理融合和 benchmark 设计实践。
- 数据分析与可视化：能够围绕公开数据完成抓取、清洗、特征统计、模型对比和结果可视化。

### 爱好

- 电影、动漫、音乐、咖啡、宠物

### 游戏

- CHUNITHM、オンゲキ、プロセカ、Enter the Gungeon、vivid/stasis、Noita、Dungreed、Celeste、鸣潮

### 饮食偏好

- 喜欢：虾、牛肉、鸡蛋
- 不喜欢：香菜

## 教育背景

- 本科：上海电力大学、计算机科学与技术、2022级
- 硕士：复旦大学、人工智能、2026级

## 个人信息卡

- 家乡：上海
- 生日：2月
- MBTI：INFP

## 公开链接

- GitHub：https://github.com/MAK1MAAaa

## 飞书文档主页

- 主页链接：https://fudan-nlp.feishu.cn/wiki/IgxdwBNAliaAn5kjS9uc41SPnqg
- 权限状态：组织内公开

飞书主页可以保存更完整的个人介绍、课程笔记、作业索引和组内材料，但不要保存 Secret、Token、Cookie、密码或私钥。飞书 URL 和文档标题会出现在公开 Git 历史中，标题本身也不能包含保密项目名。

## 公开声明

我确认本 GitHub Profile 中的信息可以长期公开展示，并可以用于 OpenMOSS-暑期集训-2026 的学习交流与作业审核。我的飞书文档主页正文设置为组织内公开。
