# AI Structure Plan: Transformer + PBT for CTF

本文档详细描述了针对Capture The Flag (CTF) 游戏的AI系统设计架构。本方案采用 **Transformer** 作为核心策略网络，结合 **Population-Based Training (PBT)** 进行进化训练，并引入 **地图标准化 (Map Normalization)** 和 **模仿学习 (Imitation Learning)** 以加速收敛并提升最终性能。

---

## 1. 核心设计理念

### 1.1 地图标准化策略 ("Always Left")
为了降低模型学习复杂度，我们采用**绝对位置标准化**策略。
- **原理**: 无论AI当前控制的是L队还是R队，输入给模型的地图数据始终通过变换（翻转），使得己方永远看起来在地图的**左侧** (x < 10)，敌方永远在**右侧**。
- **优势**: 
    - 模型只需学习一种进攻模式（从左往右）。
    - 训练数据利用率翻倍，无需额外的数据增强。
    - 简化特征工程，位置编码更具一致性。
- **实现**:
    - **预处理**: 如果当前是R队，将地图矩阵沿X轴中心翻转 ($x' = W - 1 - x$)。
    - **后处理**: 模型输出的动作如果是左右移动，需要根据队伍身份反转回真实世界的方向 (Left ↔ Right)。

### 1.2 混合训练策略 (Imitation -> RL)
为了解决强化学习冷启动（Cold Start）困难的问题，采用分阶段训练策略。
- **Phase 0 (引导期)**: 利用现有的启发式脚本（如 `walk_to_first_flag_and_return`）生成专家轨迹，使用 **行为克隆 (Behavioral Cloning)** 对Transformer进行预训练。
- **Phase 1 (提升期)**: 切换到 **PPO (Proximal Policy Optimization)** 算法，使用预训练权重初始化，进行强化学习微调。
- **Phase 2 (进化期)**:引入 **PBT** 和 **Self-Play**，在种群中进行对抗演化，突破专家策略的天花板。

---

## 2. 模型输入设计 (Input Representation)

状态空间被设计为 **Multi-Channel 2D Grid + Entity Tokens** 的混合形式，结合了CNN的空间感知能力和Transformer的全局注意力机制。

### 2.1 全局地图编码 (Spatial Grid)
输入形状: `(Batch, Channels, Height, Width) = (B, 20, 20, 20)`
**标准化**: 确保己方基地始终在 Grid 左侧。

| 通道 ID | 描述 (标准化后) |
| :--- | :--- |
| 0-2 | 己方玩家 1-3 位置 (One-hot per channel) |
| 3-5 | 己方携带旗帜玩家 位置 |
| 6-8 | 敌方玩家 1-3 位置 |
| 9-11 | 敌方携带旗帜玩家 位置 |
| 12 | 监狱区域 (Prison) |
| 13 | 己方基地 (Home Base) |
| 14 | 己方基地已有旗帜 |
| 15 | 敌方基地 (Target Base) |
| 16 | 障碍物/墙壁 (Static) |
| 17 | 空白区域 |
| 18 | 己方目标旗帜 (待夺取) |
| 19 | 敌方旗帜 (需防守) |

### 2.2 实体与元数据 (Entity & Metadata)
虽然Grid包含了位置信息，但不包含非空间属性。我们将以下信息编码为 **Global Token**:
- **Game Info**: 剩余时间、当前比分差。
- **Player State**: 每个玩家的 `inPrisonTimeLeft` (归一化)。

---

## 3. 网络架构设计 (Network Architecture)

采用 **Vision Transformer (ViT)** 变体，专门针对多智能体网格游戏优化。

### 3.1 Backbone: CNN-Transformer Encoder
1.  **Feature Extraction**: 使用浅层 CNN (3层 Conv2d) 提取局部空间特征，将 `(20, 20, 20)` 映射为 `(D_model, 5, 5)` 的特征图。
2.  **Flatten & Patch Embedding**: 将特征图展平为序列。
3.  **Token Injection**: 拼接 Global Token (元数据)。
4.  **Positional Encoding**: 加入可学习的 2D 位置编码。
5.  **Transformer Encoder**: 
    - Layers: 4-6 层
    - Attention Heads: 8
    - Hidden Dim: 256
    - Feedforward Dim: 1024
    - Activation: GeLU

### 3.2 Multi-Agent Coordination Mechanism
为了让3个玩家协同工作（如：一人吸引火力，一人偷家），在Encoder后加入 **Agent-Specific Attention**:
- 定义3个可学习的 **Query Token**，分别代表 Player 0, 1, 2。
- 利用 Cross-Attention 机制，让每个 Player Query 从全局特征中关注对自己有用的信息。

### 3.3 Policy Head (Action Head)
- **Output**: 3 个独立的 Head，分别对应 3 个玩家。
- **Type**: Multi-Discrete Action Space。
- **Format**: `(batch, 3_players, 5_actions_logits)`。
- **Action Map**: 
    - 0: Up
    - 1: Down
    - 2: Left (标准化后：向己方基地移动)
    - 3: Right (标准化后：向敌方基地移动)
    - 4: Stay

---

## 4. 训练系统设计 (Training System)

### 4.1 模仿学习流程 (Imitation Phase)
1.  **数据收集**: 
    - 运行 `walk_to_first_flag_and_return` 脚本，在随机生成的地图上进行左右互搏。
    - 记录 `(State, Action)` 对。注意在记录时应用 **Map Normalization**。
    - 收集约 10k-50k 步数据。
2.  **预训练 (BC)**:
    - 损失函数: Cross Entropy Loss。
    - 目标: 最小化模型预测动作与脚本专家动作的差异。
    - 验证: 在验证集上准确率达到 80%+ 即可停止，避免过度拟合死板规则。

### 4.2 强化学习流程 (RL Phase)
1.  **算法**: PPO (Proximal Policy Optimization)。
    - 适合离散动作空间，训练稳定。
2.  **Reward Design (奖励塑形)**:
    - **稀疏奖励 (Sparse)**: 夺旗 (+10), 运回旗帜 (+50), 抓捕敌人 (+5), 被抓 (-5)。
    - **稠密奖励 (Dense)**: 
        - 距离目标旗帜更近 (势场引导)。
        - 携带旗帜时距离家更近。
        - 探索未访问区域 (Exploration Bonus)。
3.  **Curriculum Learning (课程学习)**:
    - **Stage 1**: 对手静止，学习导航和机制。
    - **Stage 2**: 对手为简单脚本 (BFS)，学习躲避和抓捕。
    - **Stage 3**: 对手为 PBT 种群中的历史版本，进行高强度对抗。
        - 3.1: With Dense Reward
        - 3.2: Only with Sparse Reward

### 4.3 PBT (Population-Based Training) 架构
1.  **种群规模**: 16 个 Agent 并行训练。
2.  **超参数进化**: 动态调整 Learning Rate, Entropy Coefficient, Reward Weights。
3.  **淘汰与继承**: 每隔 `N` 步评估，性能最差的 20% Agent 淘汰，替换为最优 Agent 的权重并进行突变 (Mutation)。

---

## 5. 实现路线图 (Implementation Roadmap)

### Phase 1: 基础设施 (Week 1)
- [ ] 完善 `CTFMatrixConverter`，实现 "Always Left" 的 `convert_to_normalized_matrix`。
- [ ] 编写 `DataCollector`，利用现有脚本生成 HDF5 格式的专家数据。
- [ ] 搭建 Transformer Backbone 代码 (PyTorch)。

### Phase 2: 模仿学习 (Week 1.5)
- [ ] 训练 BC 模型，验证其能否走出基本的夺旗路线。
- [ ] 可视化 Attention Map，确认模型关注到了关键实体（如旗帜、敌人）。

### Phase 3: PPO基线 (Week 2)
- [ ] 接入强化学习环境。
- [ ] 加载 BC 权重，开启 PPO 训练。
- [ ] 调试 Reward Function，确保 Agent 不会产生"刷分"等怪异行为。

### Phase 4: PBT与大规模训练 (Week 3+)
- [ ] 部署多进程/多GPU训练。
- [ ] 开启 Self-Play 循环。
- [ ] 最终模型评估与导出。
