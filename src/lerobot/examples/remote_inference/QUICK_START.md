# 🚀 LeRobot远程推理快速开始指南

## 📋 简介

这个示例展示如何将**SmolVLA策略部署到AutoDL GPU服务器**，然后在**本地笔记本控制真实机器人**。

## ✅ 前提条件

### 硬件
- **AutoDL GPU服务器** (推荐RTX 3090/4090)
- **本地笔记本** (连接机器人)
- **LeRobot机器人** (如SO100)

### 软件
- Python 3.10+
- PyTorch 2.0+
- LeRobot

## 📦 文件说明

```
examples/remote_inference/
├── README.md                              # 详细文档
├── DATA_FORMAT.md                         # 数据格式说明 ⭐
├── QUICK_START.md                         # 本文件
├── policy_server_simple.py                # 策略服务器
├── policy_client_simple.py                # 简单客户端
├── utils.py                               # 数据格式转换工具 ⭐
└── robot_control_complete_example.py      # 完整示例 ⭐
```

## 🎯 核心答案：数据格式

**是的！客户端和服务器端使用的都是LeRobot标准格式！**

### LeRobot标准格式

**Observation (观测)**:
```python
{
    "observation.state": torch.Tensor,       # (batch, state_dim), float32
    "observation.images": List[torch.Tensor], # [(batch, C, H, W), ...], float32, [0,1]
    "task": List[str]                        # 任务描述（VLA模型需要）
}
```

**Action (动作)**:
```python
torch.Tensor  # (batch, action_dim), float32
```

### 转换流程

```
机器人原始格式 → robot_obs_to_policy_input() → LeRobot格式 → 网络传输
                                                      ↓
策略推理 ← LeRobot格式 ← deserialize_observation() ← 服务器接收
    ↓
LeRobot格式 → policy_action_to_robot_action() → 机器人格式 → 执行
```

详见 **[DATA_FORMAT.md](./DATA_FORMAT.md)** 获取完整说明！

## ⚡ 10分钟快速开始

### 步骤1: AutoDL服务器部署 (5分钟)

```bash
# SSH登录AutoDL
ssh root@your-autodl.com

# 安装依赖
pip install pyzmq torch

# 启动策略服务器
cd /path/to/lerobot
python examples/remote_inference/policy_server_simple.py \
    --policy_path=lerobot/smolvla_base \
    --port=5555 \
    --device=cuda
```

**输出**:
```
Loading policy from lerobot/smolvla_base
Policy server started on port 5555
Device: cuda
Waiting for requests...
```

### 步骤2: 本地测试连接 (2分钟)

```bash
# 在本地笔记本
pip install pyzmq torch numpy

# 测试连接 (替换YOUR_AUTODL_IP)
python examples/remote_inference/policy_client_simple.py \
    --server_ip=YOUR_AUTODL_IP \
    --port=5555 \
    --test
```

**预期输出**:
```
==================================================
Testing connection to policy server...
==================================================
1. Testing ping...
✓ Ping successful
2. Testing reset...
✓ Reset successful
3. Testing inference with dummy data...
✓ Inference successful. Action shape: torch.Size([1, 6])
4. Running latency benchmark (10 requests)...
✓ Average latency: 138.5ms (±8.3ms)
==================================================
All tests passed! ✓
==================================================
```

### 步骤3: 真机控制 (3分钟)

```bash
# 方法1: 使用完整示例（推荐）
python examples/remote_inference/robot_control_complete_example.py \
    --server_ip=YOUR_AUTODL_IP \
    --server_port=5555 \
    --robot_type=so100_follower \
    --robot_port=/dev/ttyUSB0 \
    --task="Pick up the red cube" \
    --max_steps=500 \
    --fps=30

# 方法2: 使用简单客户端
python examples/remote_inference/policy_client_simple.py \
    --server_ip=YOUR_AUTODL_IP \
    --robot_type=so100_follower \
    --robot_port=/dev/ttyUSB0
```

**运行输出**:
```
Initializing robot: so100_follower
Connecting to robot...
Robot connected!
Connected to policy server at tcp://123.456.789.0:5555
Motor names: ['shoulder_pan', 'shoulder_lift', 'elbow_flex', ...]

Validating observation format...
✓ observation.state: shape=(1, 6), dtype=float32, device=cpu
✓ observation.images[0]: shape=(1, 3, 480, 640), dtype=float32, ...

Starting episode (max_steps=500, fps=30)
Step 0/500 | Total: 145.2ms (Inference: 85.3ms, Network: 52.1ms)
Step 10/500 | Total: 138.7ms (Inference: 82.1ms, Network: 49.3ms)
...
```

## 📊 数据格式示例

### 从机器人到策略

```python
# 1. 机器人原始观测
robot_obs = {
    "shoulder_pan.pos": 0.1,
    "shoulder_lift.pos": 0.2,
    "elbow_flex.pos": -0.3,
    # ...更多关节
    "observation.images.top": np.array(...),     # (480, 640, 3) uint8
    "observation.images.wrist": np.array(...),   # (480, 640, 3) uint8
}

# 2. 转换为LeRobot格式 ✨
from utils import robot_obs_to_policy_input

policy_obs = robot_obs_to_policy_input(
    robot_obs,
    motor_names=["shoulder_pan", "shoulder_lift", "elbow_flex", ...],
    device="cpu"
)

# 3. 结果 - LeRobot标准格式 ✅
policy_obs = {
    "observation.state": torch.Tensor([[0.1, 0.2, -0.3, ...]]),  # (1, 6)
    "observation.images": [
        torch.Tensor(...),  # (1, 3, 480, 640), float32, [0,1]
        torch.Tensor(...),  # (1, 3, 480, 640), float32, [0,1]
    ]
}

# 4. 序列化发送
from utils import serialize_observation
obs_bytes = serialize_observation(policy_obs)
# 通过网络发送...
```

### 从策略到机器人

```python
# 1. 服务器返回LeRobot格式动作
action = torch.Tensor([[0.15, 0.25, -0.28, ...]])  # (1, 6)

# 2. 转换为机器人格式 ✨
from utils import policy_action_to_robot_action

robot_action = policy_action_to_robot_action(
    action,
    motor_names=["shoulder_pan", "shoulder_lift", "elbow_flex", ...]
)

# 3. 结果 - 机器人执行格式 ✅
robot_action = {
    "shoulder_pan.pos": 0.15,
    "shoulder_lift.pos": 0.25,
    "elbow_flex.pos": -0.28,
    # ...
}

# 4. 发送给机器人
robot.send_action(robot_action)
```

## 🛠️ 工具函数

我们提供了 `utils.py` 处理所有数据格式转换：

```python
from utils import (
    # 格式转换
    robot_obs_to_policy_input,      # 机器人 → LeRobot
    policy_action_to_robot_action,  # LeRobot → 机器人
    
    # 序列化
    serialize_observation,          # dict → bytes
    deserialize_observation,        # bytes → dict
    serialize_tensor,               # tensor → bytes
    deserialize_tensor,             # bytes → tensor
    
    # 验证
    check_observation_format,       # 检查观测格式
    check_action_format,            # 检查动作格式
    
    # 图像处理
    compress_image_to_bytes,        # 压缩图像
    decompress_image_from_bytes,    # 解压图像
)
```

## 📈 性能指标

### 典型延迟 (AutoDL + 国内网络)

| 组件 | 延迟 |
|------|------|
| 模型推理 (GPU) | 50-80ms |
| 网络往返 (RTT) | 30-80ms |
| 序列化/反序列化 | 5-10ms |
| **总延迟** | **85-170ms** |

### 优化建议

1. **图像压缩** - 减少50-80%传输量:
   ```python
   policy_obs = robot_obs_to_policy_input(
       robot_obs, motor_names,
       compress_images=True,  # 启用JPEG压缩
       image_size=(224, 224)  # 降低分辨率
   )
   ```

2. **选择更近的服务器** - 减少网络延迟

3. **批量推理** - 提高吞吐量（不适用于实时控制）

## 🔍 故障排查

### 问题1: 连接超时
```
TimeoutError: Server did not respond in time
```
**解决**: 检查AutoDL防火墙，确保端口5555已开放

### 问题2: 数据格式错误
```
ValueError: observation.state should be 2D (batch, state_dim)
```
**解决**: 使用 `check_observation_format()` 验证格式
```python
from utils import check_observation_format
check_observation_format(policy_obs, verbose=True)
```

### 问题3: GPU内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 在服务器使用 `--device=cpu` 或选择更大显存的GPU

## 📚 更多资源

- **[DATA_FORMAT.md](./DATA_FORMAT.md)** - 完整数据格式说明
- **[README.md](./README.md)** - 详细架构和部署文档
- **[REMOTE_INFERENCE_ANALYSIS.md](../../REMOTE_INFERENCE_ANALYSIS.md)** - 深入分析

## 💡 关键要点

1. ✅ **统一格式**: 客户端和服务器都使用LeRobot标准格式
2. ✅ **自动转换**: `utils.py`处理所有转换逻辑
3. ✅ **验证工具**: 内置格式检查，确保数据正确
4. ✅ **开箱即用**: 提供完整示例，直接运行
5. ✅ **易于调试**: 详细日志和错误信息

## 🎉 开始使用

现在你已经了解了数据格式和工作流程，可以：

1. **测试连接** - 运行步骤2确保通信正常
2. **查看数据格式** - 阅读 [DATA_FORMAT.md](./DATA_FORMAT.md)
3. **运行真机** - 使用步骤3控制机器人
4. **自定义修改** - 根据需求调整代码

祝你使用愉快！🚀


