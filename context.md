# Quadruped Multi-skill Locomotion（对话上下文汇总）

> 面向：硕士毕设交付 + 可扩展为后续 LiDAR 研究
> 主题：用 **前向深度相机（Depth-only）** 学**多技能敏捷行走/越障（parkour）**，以 **特权高度/BEV 监督**对齐几何语义，并通过 # AW-BC 实现完成 ✅ + DAgger 回放计划

**AW-BC（优势加权蒸馏）已实现完成**，按照"双进程：collect 持续采 + train 离线吃数据"的架构完整落地。下面是实现总结和后续 DA---

## 下一步：DAgger 回放 / 难段过采样 (计划中)

基于已完成的 AW-BC 基础，继续实现 DAgger 难段过采样，结合你的"双进程：collect 持续采 + train 离线吃数据"架构。

### 计划实现的功能

1. **难度分数计算 (Collect 端扩展)**
   ```pytho---

## Git Commit 总结

### 本次提交内容: AW-BC (优势加权行为克隆) 完整实现

**主要变更文件:**
- `rsl_rl/storage/rollout_files/rollout_dataset.py` - 扩展 Transition 数据结构
- `rsl_rl/storage/rollout_storage.py` - ActionLabelRollout 支持优势字段
- `rsl_rl/runners/demonstration.py` - GAE 优势计算和轨迹保存
- `rsl_rl/runners/dagger_saver.py` - 教师价值函数计算
- `rsl_rl/algorithms/tppo.py` - 优势权重应用到蒸馏损失

**技术特性:**
- ✅ 完整的 GAE 优势计算流水线
- ✅ 分位数归一化 + 截断的稳定权重机制  
- ✅ 完全向后兼容旧数据文件
- ✅ 丰富的 TensorBoard 监控和统计
- ✅ 双进程架构完美集成 (collect + train)

**预期效果:**
- 减少达到目标成功率所需的训练步数
- 重点学习高价值/关键时刻片段
- 为后续 DAgger 难段过采样奠定基础

**下一阶段:** 实现 DAgger 回放和难段过采样机制
   # 难度分数: D_t = α·norm(A_t^+) + β·norm(near_risk)
   near_risk = max(0, τ - min_range)  # 近距风险
   difficulty_score = α * norm(positive_advantage) + β * norm(near_risk)
   ```
   - 基于已有的 `positive_advantages`
   - 添加近距风险: `(τ - min_range)+` 或距离阈值触发
   - 可选扩展: 边缘检测、碰撞前预警窗口

2. **老师接管模式 (Collect 端)**
   - 触发条件: 连续 M 步 `D_t > τ_hi` 
   - 接管操作: 老师控制 N=20-50 步录制正解示范
   - 数据标记: `is_teacher_demo=True`, `difficulty_level`
   - 比例控制: ≤10% 避免过度依赖老师策略

3. **难段过采样 (Train 端)**
   - **WeightedSampler**: Top 20-30% 难样本采样权重 ×2
   - **目标占比**: 每批次 ≥25% 来自难题/近期样本  
   - **新鲜度权重**: 按时间戳或 checkpoint_idx 优先新数据
   - **损失加权**: 可选择将 `difficulty_scores` 也应用到损失

4. **监控与调优**
   - 难题子集指标: 梅花桩/坑沟边缘/高速近距成功率
   - 采样占比监控: 避免难段过拟合 (≤50%)
   - 全局性能检查: 确保简单场景不退化

### 实现时间线
- **Day 1-2**: 难度分数计算，数据标记扩展
- **Day 3-4**: WeightedSampler 实现，难段过采样调试  
- **Day 5-6**: 老师接管模式，完整测试
- **Day 7**: 参数调优，结果验证，文档更新

### 预期效果叠加
- **AW-BC 单独**: 减少训练步数，曲线更稳
- **+ 难段过采样**: 几何敏感场景 +5-10% 成功率
- **+ 老师接管**: 极难片段的安全兜底机制回放计划。

## ✅ AW-BC 实现完成 (已提交)

### 实现的核心组件

1. **数据结构扩展**
   - `RolloutDataset.Transition` 新增: `teacher_advantages`, `positive_advantages`, `difficulty_scores`
   - `ActionLabelRollout` 完整支持优势数据的存储、传递和批次处理
   - 向后兼容旧数据文件（自动填充零值）

2. **Collect 端 (DaggerSaver + DemonstrationSaver)**
   - `get_transition()` 计算教师价值函数 (`teacher_values`)
   - `wrap_up_trajectory()` 实现 GAE 优势计算
   - 保存完整优势数据: `advantages`, `positive_advantages`, `difficulty_scores`

3. **Train 端 (TPPO)**
   - `compute_advantage_weights()` 实现分位数归一化 + 截断
   - `compute_losses()` 应用优势权重到蒸馏损失
   - 完整的日志记录和统计监控
   - 支持 'percentile' 和 'softmax' 两种权重计算方法

4. **数据流完整验证**
   ```
   DaggerSaver.get_transition() [7个返回值]
   → DemonstrationSaver.wrap_up_trajectory() [GAE计算]
   → RolloutDataset._refresh_traj_data() [加载优势数据]
   → ActionLabelRollout [存储和批次处理]
   → TPPO.compute_losses() [应用权重]
   ```

### 关键特性

- **批内分位归一化**: `w_t = min(A_t^+ / P95(A^+), 1.0)` 避免权重爆炸
- **梯度隔离**: 权重计算使用 `detach()` 不参与反向传播
- **向后兼容**: 自动处理不包含优势数据的旧轨迹文件
- **丰富监控**: TensorBoard 记录权重分布、分桶损失、高权重占比等

### 预期效果
- 减少达到固定成功率所需的训练步数
- 将学习重心转向高价值/关键时刻片段
- 提升几何敏感场景的学习效率势加权蒸馏 + 轻量 DAgger 回放** 提升学习效率与鲁棒性。LiDAR 融合作为可选副线/后续工作。

---

## 1) 项目背景与角色

* **总体**：*Quadruped Multi-skill Navigation with Foundation Models*（后期可与导航/大模型结合）。
* **平台**：Isaac Gym（先期），可迁移 Isaac Lab/真机。
* **你的侧重点**：**多技能运动策略（parkour skills）**，高有效性地形与课程设计。

---

## 2) 感知与任务设定（当前主线）

* **主线选择**：优先 **Depth-only**（前向深度相机），放弃在 30 天硬上 LiDAR 的高风险端到端方案。
* **核心思想**：学生先从深度图**重建“高度/BEV 语义”**（Pred-Height / Multi-Layer Elevation Map, MLEM），再出动作。
* **代表地形**（不含转弯）：

  * **Stepping-stones**（梅花桩，全向/前向）
  * **Gap/坑沟/盲顶**
  * **台阶/窄台/跨杆**
  * **粗糙/波纹地面**
  * **高速直线 + 急停**
* **扩展（可选）**：LiDAR 作为近距/全向安全与薄障碍补强，后续再做。

---

## 3) 教师-学生与观测差异（关键认知）

* **独立网络**：学生**不会**拷贝/恢复教师权重；两者**完全独立**。
* **观测不同**：

  * **教师**：`height_measurements`（MLP 编码，特权）
  * **学生**：`forward_depth`（CNN 编码，无特权）
* **训练本质**：教师仅前向产出监督信号（动作/价值等），学生在自己的配置下被**监督学习/蒸馏**。

---

## 4) 双进程工作流（你当前管线）

* **Collect 进程（常驻）**

  * 学生开车与环境交互（周期热加载最新 student ckpt）。
  * 老师**只打标签**（teacher\_action、value/advantage 等）。
  * 将样本与标签**持续落盘**到 `pretrain_dataset.data_dir`。
* **Train 进程（始终离线）**

  * `pretrain_iterations = -1` → **TwoStageRunner 永远离线**：不 `env.step()`、不跑老师，只**消费数据集**。
  * 训练形式：**监督学习/行为克隆**（蒸馏 L1/KL + 你的辅助重建损失）。

> 结论：系统整体是“**持续采集 + 持续离线训练**”的半在线模式；训练本身不与环境交互。

---

## 5) “三件套”技术路线（均已定为要做）

1. **AW-BC（优势加权蒸馏）**

   * 权重来源：采集端用老师 critic 计算 **GAE 优势**，取正优势 $A_t^+$。
   * 训练端：对蒸馏 L1 乘以 $w_t=\min\{A_t^+/\text{P95}(A^+),1\}$（批内分位归一+截断）。
   * 目标：把学习重心放在“关键时刻/高价值片段”。

2. **Depth→Height / 多层 Elevation-BEV 重建（强几何监督）**

   * 在学生编码器后加 **Pred-Height**（环形高度向量）或 **Pred-MLEM**（elev\_mean/elev\_var/slope/neg\_gap/min\_range…）。
   * 损失：Huber/MSE 或异方差 NLL，$\lambda_{\text{aux}}=0.2\to0.1$。
   * 作用：让学生先学会“可踩/不可踩”的几何语义，显著提升几何敏感地形表现与泛化。

3. **DAgger 回放 / 难段过采样（difficulty-aware）**

   * 采集端计算难度分数 $D_t$：如 $A_t^+$ + 近距风险（min\_range 阈值）等，并可在**极难片段**插入短段老师示范。
   * 训练端用 **WeightedSampler** 提高“难题/近期样本”采样概率（目标≥25%/batch），或乘到损失权重。
   * 目的：修复长尾难题（梅花桩/坑缘/高速近距）与分布偏移。

---

## 6) 实验与指标（主线 Depth-only）

* **KPI**：成功率、碰撞/跌倒、到达时间、**min-safety-distance**、动作平滑度、速度/航向 RMSE。
* **感知指标**：Pred-Height / MLEM 的 MSE/NLL、相关系数；可视化 GT vs Pred。
* **鲁棒性曲线**：深度空洞/模糊/丢帧/相机俯仰抖动扫参的 AUC。
* **未见地形**：更窄的梁、更小石台等，检验泛化。
* **消融矩阵**：Baseline → +预训练 → +辅助重建 → +AW-BC → +回放/过采样（Full）。
* **结构对比（可选）**：\*\*“CNN 局部特征 + 点级注意”\*\*优于“纯 ViT/下采样 CNN/大 Transformer”（对应 D.3 讨论）。

---

## 7) 最小实现要点（改动点清单）

* **Collect 端**

  * 计算并写入：`adv_teacher`（GAE）、`D_t`（难度，A⁺+近距等）、可选 `is_teacher_demo`。
  * （可选）当 `D_t` 连续超阈 → 插入 N 步 teacher takeover 片段。
* **Train 端**

  * **AW-BC**：在蒸馏 L1 处乘 $w_t$（批内分位归一+截断，`detach`）。
  * **辅助头**：挂 Pred-Height/BEV，$\lambda_{\text{aux}}=0.2\to0.1$。
  * **回放/过采样**：WeightedSampler 提高 Top 20–30% 难样本采样率（目标≥25%/batch）。
  * **日志**：权重直方图、分桶 L1（Q50/Q75/Q90）、难题子集指标、Pred vs GT 可视化。

---

## 8) 训练加速与稳定技巧

* **Stage 1.5 预训练**：仅 `(Depth, GT-Height/BEV)` 训练编码器+重建头数小时→显著提速。
* **Teacher 前向放采集端**；Train 端 AMP；小 batch 高频更新；早期冻结新头 1–3k 步再全解冻。
* **Estimator**：replace\_state\_prob 1.0 → 0.5/0.3 退火。
* **小 KL（0.02）**：曲线不稳时启用，后期退火至 0。

---

## 9) 里程碑与时间线（30 天）

* **W1**：Depth-only + Pred-Height/BEV 跑通 3 类核心地形；写 Method/设置。
* **W2**：全量评测 + 鲁棒性；接入 **AW-BC** 并固化加速收益。
* **W3**：接入 **DAgger 回放/难段过采样**；做难题子集与未见地形评测。
* **W4**：图表/统计/失败案例；（可选）短程 PPO 微调 + 真机小范围验证。

---

## 10) 论文写法要点

* **贡献**：

  1. *Privileged-to-Visual* 的高度/BEV 语义对齐（几何可解释）；
  2. AW-BC + 难段回放带来 **更快收敛/更少采样**；
  3. 多技能基准与未见地形泛化；
  4. 可选真机小验证。
* **图表**：学习效率曲线、子集成功率柱状、鲁棒性退化曲线、Pred vs GT 热图、难题样本占比与收益。

---

## 11) LiDAR 的位置（当前决定）

* **本次主线不纳入**（降低风险，保证论文完备与可交付）。
* **后续入口**：在现有 MLEM/BEV 表征上**无缝加一条 LiDAR 通道**或做中层融合（FiLM/交叉注意/门控），先以**感知指标**（Pred-MLEM 误差）验证增益，再推到行为。

---

## 12) 常见风险与止损

* **权重爆炸**：AW-BC 必做分位归一 + 截断；权重 `detach`。
* **难题过拟合**：难题采样占比 ≤ 50%；关注全局/简单场景是否退化。
* **重建学不稳**：Pred-Height 用 Huber；BEV 聚合用中位/20%分位；无效束掩码。
* **分布老化**：collect 端热加载 ckpt 足够频；“近期样本”在采样器中加新鲜度加权。

---

### 一句话总括

**Depth-only + 几何强监督** 是你 30 天内最稳、最可解释、最容易出漂亮图表的主线；在此基础上叠加 **AW-BC** 与 **DAgger 回放**，可以把训练效率和难题表现同时拉起来。LiDAR 先留作后续扩展点，不阻塞毕设交付。
---

---

# 先上 AW-BC，再上 DAgger 回放 难段过采样

太好了，就按你说的顺序干：**先上 AW-BC**，再上 **DAgger 回放/难段过采样**。下面是完全对着你这套“双进程：collect 持续采 + train 离线吃数据”的落地说明，尽量白话、一步一步来。



## 先做 AW-BC（优势加权蒸馏）

### 0) 要达到什么效果

把蒸馏损失（学生动作 ≈ 老师动作）**在“关键时刻/高价值片段”上放大**，在“平平无奇的走路片段”上减小权重。这样**同样的训练步数**里，模型更快学到“要点”。

### 1) 权重从哪来（A_t）

你现在是**collect 持续采样 → 写入数据集**，**train 完全离线**。  
因此**最佳做法**：**在 collect 端就把老师优势 A_t 算好并写进样本**（后面 DAgger 难段过采样也要用到它）。

- **A_t 计算**（GAE）：
    
    - collect 每个 episode 完成后，用老师 critic 的 $V(s_t)$ 和存下的奖励 $r_t$，按 γ,λ 算 GAE：  
	    $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)，$  
        $A_t = \text{GAE}_t(\delta, \gamma, \lambda)$。
        
    - 存成字段：`adv_teacher`（float），**只保留正优势**的也可以另存一个 `adv_pos = max(A_t, 0)`。
        

> 如果你现在的 DaggerSaver 已经存了 `teacher_value/advantage`，直接复用；没有就加 30 行代码在 episode 结束时回填即可。

### 2) 训练端怎么用（不改网络，只改损失）

- **批内规范化 + 截断**（推荐稳妥版本）
    
    - 取正优势：$A_t^+ = \max(A_t, 0)$
        
    - 做分位归一：$\tilde w_t = A_t^+ / \text{P95}(A^+ \text{ in batch})$
        
    - 截断到 [0,1][0,1]： $w_t = \min(\tilde w_t, 1)$
        
    - 蒸馏项改为：$L_{\text{distill}} = \mathbb{E}[\, w_t \cdot \|\mu_s - \mu_t\|_1\,]$
        
- **实现点位**（不写具体代码，告诉你“在哪乘上 w_t”）：
    
    - 你现在的蒸馏在 **TPPO/EstimatorTPPO 的 `compute_losses()`** 里算（或等价位置）。
        
    - 读取 batch 里的 `adv_teacher` → 变成 `w_t`（按上面三步）→ 直接乘到蒸馏 L1 上。
        
    - **注意**：`w_t` **不参与梯度**（`detach`），并且**每个 batch 重新按分位归一**（避免尺度漂移）。
        
- **两个可切换的权重风格**（供你AB试验）
    
    - 线性版（上面那个）：稳定，最推荐。
        
    - 软指数版：$w_t = \exp(A_t^+ / \tau)$，τ\tau 取 **P90** 或一个常数（如 1.0），再归一到 [0,1][0,1]。指数版更激进，容易抖。
        

### 3) 日志 & 自检（避免“越加权越抖”）

- 打印 **权重直方图**（0~1）和 **>0.7 的占比**，看是否集中在上半区。
    
- 把蒸馏 L1 按 **权重分位**（Q50/Q75/Q90）分桶，看高权重桶是否先降。
    
- 观察“**epoch→epoch** 成功率”，高权重策略不应让曲线更抖；若抖，先把截断从 P95 改 P90 或开个小 KL=0.02 稳一手。
    

---

## 然后做 DAgger 回放 / 难段过采样（结合你的双进程）

你的 train 是**离线**，所以我们把“**难度意识**”主要放在**collect 端打标 + train 端采样器**两处，**不破坏**现有双进程结构。

### 4) 定义“难度分数” D_t（collect 端写入）

**最简版**就用两项，够用也好实现：

- **正优势**：$A_t^+$（已经有了）。
    
- **近距风险**：`near_obs = 1[min_range < τ]` 或 `(τ - min_range)_+`（如果你存了最小测距）。
    

合成一个分数：  
$D_t = \alpha \cdot \text{norm}(A_t^+) + \beta \cdot \text{norm}((\tau - \text{min\_range})_+)，$  
`norm` 就是除以批内 P95 或最大值，α=β=1 起步。

（有余力可再加：`is_edge_of_stepstone`、`pre-collision window`、`fall_flag` 等。）

### 5) 收集时的“小行为”（collect 端）

- **学生优先**（维持你现状）：平时学生在开车，老师只打标签。
    
- **难段兜底**（可选）：若连续 M 步 `D_t > τ_hi` 或触发 `near_obs` 严重阈值，**让老师接管 N 步**（teacher_act_prob=1）录一段“正解”。
    
- 全部写入数据集：`D_t`、`is_teacher_demo` 标志、`episode_id`、时间戳等。
    

### 6) 训练端“怎么多学难题”（不接环境，只改 DataLoader）

- **WeightedSampler**：
    
    - 给每条样本一个采样权 `s_t = c1 * rank(D_t) + c2 * is_teacher_demo`（rank 正规化到 0~1，`c2=+`），
        
    - 让 **Top 20–30% 难样本**被抽到的概率**×2**；
        
    - **目标占比**：每个 batch **≥25%** 来自“难题/近期”。
        
- **“近期样本”优先权**：按文件时间或 `checkpoint_idx` 给一个“新鲜度 bonus”，避免总是吃老数据（DAgger 的精髓是**跟着学生分布走**）。
    
- **损失再加权**（和 AW-BC 可以叠加）：
    
    - 直接把 DtD_t 也乘到蒸馏项上（或与 wtw_t 取 `max`/`avg`）；
        
    - 或者**只用在采样器**，损失里仍然只用 wtw_t（更稳）。
        

### 7) 两个可调旋钮（经验值）

- **难段占比**：从 **25%** 起步，观察训练是否过拟合难场景；最多别超过 **50%**。
    
- **“老师接管”插入率**：先关；若难段总是学不会，再开“`D_t` 超阈就插入 N 步老师示范”的模式，比例控制在 **≤10%**。
    

---

## 建议的 7 天小日程（不打乱你整体节奏）

**Day 1–2：AW-BC**

- collect 端补写 `adv_teacher`（若已有则跳过）。
    
- train 端把蒸馏 L1 改成 `w_t * L1`；做 P95 截断与日志。
    
- 小规模跑 2–3 个地形，看曲线是否更快下，成功率是否先起。
    

**Day 3–4：难度打标**

- collect 端加 `D_t`（A++A^+ + min_range 触发），随数据一起写。
    
- train 端 DataLoader 改 WeightedSampler（Top 30% ×2；batch 中 ≥25% 难样本）。
    
- 做 A/B：Baseline vs +AW vs +AW+采样权。
    

**Day 5：老师接管（可选）**

- 在 collect 端按 `D_t > τ_hi` 插入 **N=20–50 步** teacher demo；
    
- 训练端 `is_teacher_demo` 权重+0.2；
    
- 看“难题子集”（梅花桩/坑沟边缘/高速近距）的**成功率和碰撞率**是否继续改善。
    

**Day 6–7：扫参 & 固化**

- 调 P95→P90、难段占比 25→35%、是否叠加 KL=0.02；
    
- 固化一套“达标更快”的配置，开始全量跑与制图。
    

---

## 对结果的预期（现实口径）

- **AW-BC 单独上**：通常能**明显减少**达到固定成功率所需的 **更新数/采样步数**，曲线更稳。
    
- **+ 难段过采样**：**几何敏感子集**（梅花桩、坑沟边缘、高速近距）成功率继续**+5～10%**，碰撞率下降。
    
- **不该发生的**：全局指标抖得更厉害或“简单场景退化明显”。出现这种情况，先降权重、降占比、或开小 KL。
    

---

## 最后提醒的三个小坑

1. **权重爆炸**：一定做分位归一 + 截断，`detach` 掉权重。
    
2. **数据偏斜**：难段采样别超过 50%，每个 epoch 跟踪“新/旧样本比例”。
    
3. **指标要拆桶看**：别只看全局成功率，要盯“难题子集”的成功率/碰撞率和 Pred-Height 的误差。
    

---

一句话：  
**先把 AW-BC 上起来**（只是给蒸馏乘个权重，收益快且稳），**再用 DAgger 的“难题打标 + 采样/损失加权”**把“老是学不会的 20%”反复吃。你的双进程结构 **不用改**，只是在 **collect 端多存两个字段**、在 **train 端改损失和采样器**，就能拿到“更快、更稳、难题更会”的提升。