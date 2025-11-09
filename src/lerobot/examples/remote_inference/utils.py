#!/usr/bin/env python3
"""
数据格式转换工具 - 在机器人格式和LeRobot策略格式之间转换

符合LeRobot标准:
- observation.state: (batch_size, state_dim) float32
- observation.images: List[(batch_size, C, H, W)] float32, range [0,1]
- action: (batch_size, action_dim) float32
"""

import io
from typing import Any

import cv2
import numpy as np
import torch


def robot_obs_to_policy_input(
    robot_obs: dict,
    motor_names: list[str],
    image_keys: list[str] | None = None,
    device: str = "cpu",
    compress_images: bool = False,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """
    将机器人原始观测转换为LeRobot策略输入格式
    
    Args:
        robot_obs: 机器人get_observation()的输出
            格式: {"motor.pos": float, "observation.images.cam": np.array, ...}
        motor_names: 关节名称列表（不含.pos后缀），按顺序排列
        image_keys: 相机键名列表，如["observation.images.top", "observation.images.wrist"]
                   如果为None，自动检测所有observation.images.*键
        device: torch设备 ("cpu" 或 "cuda")
        compress_images: 是否压缩图像（用于网络传输）
        image_size: 目标图像尺寸 (height, width)，None表示不缩放
    
    Returns:
        LeRobot格式字典:
        {
            "observation.state": torch.Tensor,  # (1, n_motors)
            "observation.images": List[torch.Tensor],  # [(1, 3, H, W), ...]
        }
    """
    policy_obs = {}
    
    # 1. 提取关节状态
    state_values = []
    for motor in motor_names:
        key = f"{motor}.pos"
        if key not in robot_obs:
            raise KeyError(f"Motor key '{key}' not found in observation. Available keys: {list(robot_obs.keys())}")
        state_values.append(float(robot_obs[key]))
    
    state_tensor = torch.tensor([state_values], dtype=torch.float32, device=device)
    policy_obs["observation.state"] = state_tensor  # (1, n_motors)
    
    # 2. 处理图像
    if image_keys is None:
        # 自动检测所有image键
        image_keys = sorted([k for k in robot_obs.keys() if k.startswith("observation.images.")])
    
    if image_keys:
        images = []
        for img_key in image_keys:
            if img_key not in robot_obs:
                raise KeyError(f"Image key '{img_key}' not found in observation")
            
            img = robot_obs[img_key]  # numpy array, (H, W, C), uint8
            
            # 确保是numpy数组
            if not isinstance(img, np.ndarray):
                raise TypeError(f"Image at '{img_key}' is not a numpy array: {type(img)}")
            
            # 检查格式
            if img.ndim != 3:
                raise ValueError(f"Image at '{img_key}' should be 3D (H,W,C), got shape {img.shape}")
            
            # 调整大小（如果需要）
            if image_size is not None:
                img = cv2.resize(img, (image_size[1], image_size[0]))  # cv2使用(width, height)
            
            if compress_images:
                # 压缩为JPEG bytes
                img_tensor = compress_image_to_bytes(img)
            else:
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


def policy_action_to_robot_action(
    action_tensor: torch.Tensor,
    motor_names: list[str]
) -> dict[str, float]:
    """
    将策略输出的动作tensor转换为机器人格式
    
    Args:
        action_tensor: 策略输出 (batch_size, action_dim)
        motor_names: 关节名称列表（不含.pos后缀）
    
    Returns:
        机器人格式字典: {"motor.pos": float, ...}
    """
    if action_tensor.ndim != 2:
        raise ValueError(f"Action should be 2D (batch_size, action_dim), got shape {action_tensor.shape}")
    
    if action_tensor.shape[0] != 1:
        raise ValueError(f"Expected batch_size=1, got {action_tensor.shape[0]}")
    
    if action_tensor.shape[1] != len(motor_names):
        raise ValueError(
            f"Action dimension {action_tensor.shape[1]} doesn't match "
            f"number of motors {len(motor_names)}"
        )
    
    action_dict = {}
    action_values = action_tensor[0].cpu().numpy()  # (action_dim,)
    
    for i, motor in enumerate(motor_names):
        action_dict[f"{motor}.pos"] = float(action_values[i])
    
    return action_dict


def serialize_observation(obs: dict) -> bytes:
    """
    将LeRobot格式观测序列化为bytes（用于网络传输）
    
    使用torch.save进行序列化，保持完整精度
    """
    # 将所有tensor移到CPU
    obs_cpu = {}
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            obs_cpu[key] = value.cpu()
        elif isinstance(value, list):
            obs_cpu[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in value]
        else:
            obs_cpu[key] = value
    
    # 序列化
    buffer = io.BytesIO()
    torch.save(obs_cpu, buffer)
    return buffer.getvalue()


def deserialize_observation(data: bytes, device: str = "cuda") -> dict:
    """
    从bytes反序列化观测到指定设备
    """
    buffer = io.BytesIO(data)
    obs = torch.load(buffer, weights_only=True)
    
    # 移动到目标设备
    obs_device = {}
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            obs_device[key] = value.to(device)
        elif isinstance(value, list):
            obs_device[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        else:
            obs_device[key] = value
    
    return obs_device


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """序列化单个tensor"""
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes, device: str = "cpu") -> torch.Tensor:
    """反序列化单个tensor"""
    buffer = io.BytesIO(data)
    tensor = torch.load(buffer, weights_only=True)
    return tensor.to(device)


def compress_image_to_bytes(img: np.ndarray, quality: int = 85) -> bytes:
    """
    压缩图像为JPEG bytes
    
    Args:
        img: numpy数组 (H, W, C), uint8
        quality: JPEG质量 0-100
    
    Returns:
        压缩后的bytes
    """
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes()


def decompress_image_from_bytes(img_bytes: bytes, device: str = "cpu") -> torch.Tensor:
    """
    从JPEG bytes解压图像为tensor
    
    Returns:
        torch.Tensor: (1, C, H, W) float32, range [0,1]
    """
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # BGR -> RGB
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # 转为tensor
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # (C, H, W)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    
    return img_tensor


def check_observation_format(obs: dict, verbose: bool = True) -> bool:
    """
    验证观测格式是否符合LeRobot标准
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    errors = []
    
    # 检查observation.state
    if "observation.state" not in obs:
        errors.append("Missing 'observation.state'")
    else:
        state = obs["observation.state"]
        if not isinstance(state, torch.Tensor):
            errors.append(f"observation.state should be torch.Tensor, got {type(state)}")
        elif state.ndim != 2:
            errors.append(f"observation.state should be 2D (batch, state_dim), got shape {state.shape}")
        elif state.dtype != torch.float32:
            errors.append(f"observation.state should be float32, got {state.dtype}")
        elif verbose:
            print(f"✓ observation.state: shape={state.shape}, dtype={state.dtype}, device={state.device}")
    
    # 检查observation.images（如果存在）
    if "observation.images" in obs:
        images = obs["observation.images"]
        if not isinstance(images, list):
            errors.append(f"observation.images should be list, got {type(images)}")
        else:
            for i, img in enumerate(images):
                if not isinstance(img, torch.Tensor):
                    errors.append(f"observation.images[{i}] should be torch.Tensor, got {type(img)}")
                elif img.ndim != 4:
                    errors.append(
                        f"observation.images[{i}] should be 4D (batch, C, H, W), got shape {img.shape}"
                    )
                elif img.dtype != torch.float32:
                    errors.append(f"observation.images[{i}] should be float32, got {img.dtype}")
                elif img.min() < 0 or img.max() > 1:
                    errors.append(
                        f"observation.images[{i}] should be in range [0,1], "
                        f"got range [{img.min():.3f}, {img.max():.3f}]"
                    )
                elif verbose:
                    print(
                        f"✓ observation.images[{i}]: shape={img.shape}, "
                        f"dtype={img.dtype}, device={img.device}, "
                        f"range=[{img.min():.3f}, {img.max():.3f}]"
                    )
    
    if errors:
        raise ValueError("Observation format errors:\n  " + "\n  ".join(errors))
    
    return True


def check_action_format(action: torch.Tensor, expected_dim: int | None = None, verbose: bool = True) -> bool:
    """
    验证动作格式是否符合LeRobot标准
    
    Args:
        action: 动作tensor
        expected_dim: 期望的动作维度，None表示不检查
        verbose: 是否打印详细信息
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    errors = []
    
    if not isinstance(action, torch.Tensor):
        errors.append(f"Action should be torch.Tensor, got {type(action)}")
    elif action.ndim != 2:
        errors.append(f"Action should be 2D (batch, action_dim), got shape {action.shape}")
    elif action.dtype != torch.float32:
        errors.append(f"Action should be float32, got {action.dtype}")
    elif expected_dim is not None and action.shape[1] != expected_dim:
        errors.append(f"Action dimension should be {expected_dim}, got {action.shape[1]}")
    elif verbose:
        print(
            f"✓ Action: shape={action.shape}, dtype={action.dtype}, device={action.device}, "
            f"range=[{action.min():.3f}, {action.max():.3f}]"
        )
    
    if errors:
        raise ValueError("Action format errors:\n  " + "\n  ".join(errors))
    
    return True


if __name__ == "__main__":
    # 测试数据格式转换
    print("Testing data format conversion...")
    
    # 模拟机器人观测
    robot_obs = {
        "shoulder_pan.pos": 0.1,
        "shoulder_lift.pos": 0.2,
        "elbow_flex.pos": -0.3,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.5,
        "gripper.pos": 0.8,
        "observation.images.top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "observation.images.wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }
    
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    # 转换为策略格式
    policy_obs = robot_obs_to_policy_input(robot_obs, motor_names, device="cpu")
    
    # 验证格式
    print("\n检查观测格式:")
    check_observation_format(policy_obs)
    
    # 模拟动作
    action = torch.randn(1, 6)
    
    # 验证动作格式
    print("\n检查动作格式:")
    check_action_format(action, expected_dim=6)
    
    # 转换回机器人格式
    robot_action = policy_action_to_robot_action(action, motor_names)
    print("\n机器人动作格式:")
    for k, v in robot_action.items():
        print(f"  {k}: {v:.3f}")
    
    # 测试序列化
    print("\n测试序列化/反序列化:")
    obs_bytes = serialize_observation(policy_obs)
    print(f"  序列化大小: {len(obs_bytes) / 1024:.1f} KB")
    
    obs_restored = deserialize_observation(obs_bytes, device="cpu")
    print("  反序列化成功")
    
    print("\n✓ 所有测试通过！")


