# LeRobot 远程推理数据格式详解

## 📋 概述

**是的，客户端和服务器端使用的是LeRobot标准格式！**

LeRobot定义了一套标准的键名约定来标识观测和动作数据。这确保了策略模型能够正确理解输入输出。

## 🔑 LeRobot标准键名

在 `lerobot/common/constants.py` 中定义：

```python
# 观测相关
OBS_STATE = "observation.state"              # 机器人状态 (关节位置等)
OBS_IMAGES = "observation.images"            # 多个相机图像
OBS_IMAGE = "observation.image"              # 单个相机图像  
OBS_ENV_STATE = "observation.environment_state"  # 环境状态

# 动作
ACTION = "action"

# 其他
REWARD = "next.reward"
```

## 📊 数据格式规范

### 1. Observation (观测) 格式

策略的 `select_action()` 方法期望接收以下格式的字典：

```python
observation = {
    # 必需：机器人状态 (关节位置、速度等)
    "observation.state": torch.Tensor,  # shape: (batch_size, state_dim)
    
    # 可选：图像观测 (如果策略使用视觉)
    "observation.images": List[torch.Tensor],  # 多相机
    # 或
    "observation.image": torch.Tensor,  # 单相机
    
    # 可选：环境状态
    "observation.environment_state": torch.Tensor,  # shape: (batch_size, env_dim)
    
    # 可选：任务描述 (用于VLA模型如SmolVLA)
    "task": List[str],  # shape: (batch_size,)
}
```

#### 详细规范

**1.1 `observation.state` (机器人状态)**
- **类型**: `torch.Tensor`
- **形状**: `(batch_size, state_dim)`
- **数据类型**: `torch.float32`
- **取值范围**: 通常是归一化后的 [-1, 1] 或原始值
- **内容**: 关节位置、速度、力矩等

例子：
```python
# SO100机器人有6个关节
observation_state = torch.tensor([[0.1, 0.2, -0.3, 0.0, 0.5, 0.8]])  # (1, 6)
```

**1.2 `observation.images` (多相机图像)**
- **类型**: `List[torch.Tensor]` 或 `torch.Tensor`
- **形状**: 每个相机 `(batch_size, channels, height, width)`
- **数据类型**: `torch.float32`
- **取值范围**: [0.0, 1.0] (归一化后)
- **通道顺序**: Channel-first (C, H, W)

例子：
```python
# 2个相机，224x224 RGB图像
camera1 = torch.rand(1, 3, 224, 224)  # 值在[0,1]
camera2 = torch.rand(1, 3, 224, 224)
observation_images = [camera1, camera2]

# 或者堆叠成一个tensor
# observation_images = torch.stack([camera1, camera2], dim=1)  # (1, 2, 3, 224, 224)
```

**1.3 `task` (任务描述) - 用于VLA模型**
- **类型**: `List[str]`
- **形状**: `(batch_size,)`
- **内容**: 自然语言任务描述

例子：
```python
task = ["Pick up the red cube"]  # batch_size=1
```

### 2. Action (动作) 格式

策略的 `select_action()` 返回：

```python
action = torch.Tensor  # shape: (batch_size, action_dim)
```

- **类型**: `torch.Tensor`
- **形状**: `(batch_size, action_dim)`
- **数据类型**: `torch.float32`
- **取值范围**: 取决于策略输出（可能已归一化）

例子：
```python
# SO100机器人6个关节的目标位置
action = torch.tensor([[0.2, 0.3, -0.1, 0.0, 0.6, 0.9]])  # (1, 6)
```

## 🔄 从机器人格式转换到LeRobot格式

### 机器人原始格式

机器人的 `get_observation()` 返回的格式：

```python
robot_observation = {
    # 关节位置 (每个关节一个键)
    "shoulder_pan.pos": 0.1,
    "shoulder_lift.pos": 0.2,
    "elbow_flex.pos": -0.3,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.5,
    "gripper.pos": 0.8,
    
    # 相机图像 (numpy数组，HWC格式，uint8)
    "observation.images.top": np.array(...),  # shape: (480, 640, 3), dtype: uint8
    "observation.images.wrist": np.array(...),  # shape: (480, 640, 3), dtype: uint8
}
```

### 转换过程

```python
import torch
import numpy as np

def robot_obs_to_policy_input(robot_obs: dict, motor_names: list, device: str = "cuda") -> dict:
    """
    将机器人原始观测转换为LeRobot策略输入格式
    
    Args:
        robot_obs: 机器人get_observation()的输出
        motor_names: 关节名称列表，按顺序排列
        device: torch设备
    
    Returns:
        策略输入字典
    """
    policy_obs = {}
    
    # 1. 提取关节状态
    state_values = []
    for motor in motor_names:
        key = f"{motor}.pos"
        if key in robot_obs:
            state_values.append(robot_obs[key])
    
    state_tensor = torch.tensor([state_values], dtype=torch.float32, device=device)
    policy_obs["observation.state"] = state_tensor  # (1, n_motors)
    
    # 2. 处理图像
    image_keys = [k for k in robot_obs.keys() if k.startswith("observation.images.")]
    
    if image_keys:
        images = []
        for img_key in sorted(image_keys):  # 排序确保顺序一致
            img = robot_obs[img_key]  # numpy array, (H, W, C), uint8
            
            # 转换为tensor
            img_tensor = torch.from_numpy(img)  # (H, W, C)
            
            # 重排为channel-first
            img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
            
            # 转换为float32并归一化到[0, 1]
            img_tensor = img_tensor.float() / 255.0
            
            # 添加batch维度
            img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
            
            images.append(img_tensor)
        
        policy_obs["observation.images"] = images
    
    return policy_obs

# 使用示例
motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", 
               "wrist_flex", "wrist_roll", "gripper"]

robot_obs = robot.get_observation()
policy_input = robot_obs_to_policy_input(robot_obs, motor_names)

# 现在可以直接传给策略
action = policy.select_action(policy_input)
```

## 📡 网络传输格式

### 问题：Tensor无法直接序列化

`torch.Tensor` 和 `numpy.ndarray` 无法直接通过网络发送，需要序列化。

### 解决方案 1: PyTorch序列化

```python
import torch
import io

def serialize_observation(obs: dict) -> bytes:
    """将LeRobot格式观测序列化为bytes"""
    # 将所有tensor移到CPU
    obs_cpu = {}
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            obs_cpu[key] = value.cpu()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            obs_cpu[key] = [v.cpu() for v in value]
        else:
            obs_cpu[key] = value
    
    # 使用torch.save序列化
    buffer = io.BytesIO()
    torch.save(obs_cpu, buffer)
    return buffer.getvalue()

def deserialize_observation(data: bytes, device: str = "cuda") -> dict:
    """从bytes反序列化观测"""
    buffer = io.BytesIO(data)
    obs = torch.load(buffer, weights_only=True)
    
    # 移动到目标设备
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            obs[key] = value.to(device)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            obs[key] = [v.to(device) for v in value]
    
    return obs
```

### 解决方案 2: 数值数组序列化 (更小)

```python
import numpy as np

def serialize_observation_compact(obs: dict) -> dict:
    """将观测转换为可JSON序列化的格式"""
    serialized = {}
    
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            # 转为numpy然后编码
            arr = value.cpu().numpy()
            serialized[key] = {
                "data": arr.tobytes(),
                "shape": arr.shape,
                "dtype": str(arr.dtype)
            }
        elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
            serialized[key] = [
                {
                    "data": v.cpu().numpy().tobytes(),
                    "shape": v.shape,
                    "dtype": str(v.dtype)
                }
                for v in value
            ]
        else:
            serialized[key] = value
    
    return serialized

def deserialize_observation_compact(data: dict, device: str = "cuda") -> dict:
    """从紧凑格式反序列化"""
    obs = {}
    
    for key, value in data.items():
        if isinstance(value, dict) and "data" in value:
            # 重建numpy数组
            arr = np.frombuffer(value["data"], dtype=value["dtype"])
            arr = arr.reshape(value["shape"])
            # 转为tensor
            obs[key] = torch.from_numpy(arr).to(device)
        elif isinstance(value, list) and isinstance(value[0], dict):
            obs[key] = [
                torch.from_numpy(
                    np.frombuffer(v["data"], dtype=v["dtype"]).reshape(v["shape"])
                ).to(device)
                for v in value
            ]
        else:
            obs[key] = value
    
    return obs
```

## 🌐 完整的客户端-服务器数据流

### 客户端 (本地机器人)

```python
# 1. 从机器人获取原始观测
robot_obs = robot.get_observation()
# 格式: {"motor1.pos": 0.1, "motor2.pos": 0.2, "observation.images.cam1": np.array(...)}

# 2. 转换为LeRobot格式
policy_obs = robot_obs_to_policy_input(robot_obs, motor_names)
# 格式: {"observation.state": torch.Tensor, "observation.images": [torch.Tensor, ...]}

# 3. 序列化准备发送
obs_bytes = serialize_observation(policy_obs)
# 格式: bytes

# 4. 通过网络发送 (ZMQ/gRPC/HTTP)
request = {
    "command": "select_action",
    "observation": obs_bytes,
    "task": "Pick up the cube"  # 可选
}
socket.send(pickle.dumps(request))

# 5. 接收动作
response = pickle.loads(socket.recv())
action_bytes = response["action"]

# 6. 反序列化动作
action = deserialize_tensor(action_bytes)
# 格式: torch.Tensor, shape: (1, action_dim)

# 7. 转换回机器人格式
action_dict = {}
for i, motor in enumerate(motor_names):
    action_dict[f"{motor}.pos"] = action[0, i].item()
# 格式: {"motor1.pos": 0.15, "motor2.pos": 0.25, ...}

# 8. 发送给机器人
robot.send_action(action_dict)
```

### 服务器 (AutoDL GPU)

```python
# 1. 接收请求
request = pickle.loads(socket.recv())

# 2. 反序列化观测
obs_bytes = request["observation"]
policy_obs = deserialize_observation(obs_bytes, device="cuda")
# 格式: {"observation.state": torch.Tensor(cuda), "observation.images": [torch.Tensor(cuda), ...]}

# 3. 添加任务描述 (如果是VLA模型)
if "task" in request:
    policy_obs["task"] = [request["task"]]

# 4. 推理
with torch.no_grad():
    action = policy.select_action(policy_obs)
# 格式: torch.Tensor(cuda), shape: (1, action_dim)

# 5. 序列化动作
action_bytes = serialize_tensor(action)

# 6. 发送响应
response = {
    "action": action_bytes,
    "status": "success",
    "inference_time_ms": 85.3
}
socket.send(pickle.dumps(response))
```

## 📏 数据大小估算

### 典型SO100机器人示例

**观测数据**:
- `observation.state`: 6个float32 = 24 bytes
- `observation.images`: 2个相机，224x224 RGB
  - 原始: 2 × 224 × 224 × 3 × 4 bytes (float32) = 1.2 MB
  - JPEG压缩后: ~50-100 KB

**动作数据**:
- `action`: 6个float32 = 24 bytes

**总计 (每次请求)**:
- 未压缩: ~1.2 MB
- 压缩图像: ~50-100 KB

## ⚡ 优化建议

### 1. 图像压缩

```python
import cv2

def compress_image(img_tensor: torch.Tensor, quality: int = 85) -> bytes:
    """压缩图像减少传输量"""
    # tensor (C, H, W) -> numpy (H, W, C)
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # JPEG压缩
    _, buffer = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()

def decompress_image(img_bytes: bytes) -> torch.Tensor:
    """解压图像"""
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    return img_tensor
```

### 2. 降低分辨率

```python
import torch.nn.functional as F

def downsample_image(img: torch.Tensor, target_size: tuple = (224, 224)) -> torch.Tensor:
    """降低图像分辨率"""
    return F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
```

### 3. 只在变化时发送图像

```python
class SmartImageSender:
    def __init__(self, send_every_n_frames: int = 5):
        self.send_every_n = send_every_n_frames
        self.frame_count = 0
        self.last_images = None
    
    def should_send_images(self) -> bool:
        self.frame_count += 1
        return self.frame_count % self.send_every_n == 0
    
    def prepare_observation(self, obs: dict) -> dict:
        if self.should_send_images():
            return obs  # 发送完整观测
        else:
            # 只发送状态，不发送图像
            return {"observation.state": obs["observation.state"]}
```

## 🔍 调试技巧

### 检查数据格式

```python
def check_observation_format(obs: dict) -> None:
    """验证观测格式是否正确"""
    print("Observation keys:", obs.keys())
    
    if "observation.state" in obs:
        state = obs["observation.state"]
        print(f"  observation.state: shape={state.shape}, dtype={state.dtype}, device={state.device}")
        assert state.ndim == 2, "state应该是2D: (batch_size, state_dim)"
        assert state.dtype == torch.float32, "state应该是float32"
    
    if "observation.images" in obs:
        images = obs["observation.images"]
        print(f"  observation.images: {len(images)} cameras")
        for i, img in enumerate(images):
            print(f"    camera {i}: shape={img.shape}, dtype={img.dtype}, device={img.device}")
            assert img.ndim == 4, "image应该是4D: (batch_size, C, H, W)"
            assert img.dtype == torch.float32, "image应该是float32"
            assert 0 <= img.min() <= img.max() <= 1, "image应该在[0,1]范围内"

def check_action_format(action: torch.Tensor) -> None:
    """验证动作格式是否正确"""
    print(f"Action: shape={action.shape}, dtype={action.dtype}, device={action.device}")
    assert action.ndim == 2, "action应该是2D: (batch_size, action_dim)"
    assert action.dtype == torch.float32, "action应该是float32"
```

## 📝 总结

**关键要点**:

1. ✅ **使用LeRobot标准键名**: `observation.state`, `observation.images`, `action`
2. ✅ **正确的tensor形状**: 包含batch维度，图像是channel-first
3. ✅ **正确的数据类型**: float32，图像归一化到[0,1]
4. ✅ **序列化传输**: 使用torch.save或numpy序列化
5. ✅ **设备管理**: 客户端CPU，服务器GPU
6. ✅ **压缩优化**: JPEG压缩图像减少带宽

遵循这些规范，你的远程推理系统将完全兼容LeRobot生态系统！


