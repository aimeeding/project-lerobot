# LeRobot 远程策略推理示例

将SmolVLA策略部署到AutoDL GPU服务器，在本地笔记本控制真实机器人。

## 📋 架构图

```
┌─────────────────────────────────┐
│   AutoDL GPU服务器               │
│                                  │
│  ┌────────────────────────────┐ │
│  │  Policy Server             │ │
│  │  - SmolVLA模型加载         │ │
│  │  - GPU推理                 │ │
│  │  - ZMQ服务 (port 5555)     │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
           ↕ ZMQ (TCP)
           ↕ 观测 & 动作
┌─────────────────────────────────┐
│   本地笔记本                     │
│                                  │
│  ┌────────────────────────────┐ │
│  │  Policy Client             │ │
│  │  - 发送观测                │ │
│  │  - 接收动作                │ │
│  └────────────────────────────┘ │
│              ↕                   │
│  ┌────────────────────────────┐ │
│  │  真实机器人                │ │
│  │  - SO100/Koch/等           │ │
│  │  - USB串口连接             │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
```

## 🚀 快速开始

### 1. 服务端部署 (AutoDL)

```bash
# SSH登录到AutoDL实例
ssh root@your-autodl-instance.com

# 克隆代码
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 安装依赖
pip install -e ".[smolvla]"
pip install pyzmq

# 启动策略服务器
python examples/remote_inference/policy_server_simple.py \
    --policy_path=lerobot/smolvla_base \
    --port=5555 \
    --device=cuda
```

### 2. 客户端运行 (本地笔记本)

```bash
# 在本地安装依赖
pip install pyzmq torch numpy

# 测试连接 (替换为你的AutoDL IP)
python examples/remote_inference/policy_client_simple.py \
    --server_ip=123.456.789.0 \
    --port=5555 \
    --test

# 与真机集成
python examples/remote_inference/policy_client_simple.py \
    --server_ip=123.456.789.0 \
    --port=5555 \
    --robot_type=so100_follower \
    --robot_port=/dev/ttyUSB0
```

## 📝 AutoDL配置指南

### 选择实例

1. 登录 [AutoDL](https://www.autodl.com/)
2. 租用GPU实例:
   - **GPU**: RTX 3090 / RTX 4090 (推荐)
   - **内存**: 32GB+
   - **存储**: 50GB+

### 开放端口

在AutoDL控制台:
1. 点击 "容器实例" → "更多" → "自定义服务"
2. 添加端口映射:
   - 容器端口: `5555`
   - 协议: TCP
   - 描述: Policy Server

### 获取公网IP

```bash
# 在AutoDL实例上运行
curl ifconfig.me
```

记录这个IP，用于客户端连接。

## 🔧 配置选项

### 服务器参数

```bash
python policy_server_simple.py --help

Options:
  --policy_path TEXT    策略模型路径 (HuggingFace repo或本地路径)
  --device TEXT         推理设备 (cuda/cpu)
  --port INTEGER        监听端口 (默认: 5555)
```

### 客户端参数

```bash
python policy_client_simple.py --help

Options:
  --server_ip TEXT      策略服务器IP地址
  --port INTEGER        策略服务器端口 (默认: 5555)
  --timeout_ms INTEGER  请求超时 (毫秒, 默认: 5000)
  --test               运行连接测试
  --robot_type TEXT     机器人类型 (如: so100_follower)
  --robot_port TEXT     机器人串口 (如: /dev/ttyUSB0)
```

## 📊 性能测试

### 运行基准测试

```bash
python policy_client_simple.py \
    --server_ip=your-autodl-ip \
    --test
```

预期输出:
```
==================================================
Testing connection to policy server...
==================================================
1. Testing ping...
✓ Ping successful: {'status': 'pong', 'timestamp': 1234567890.123}

2. Testing reset...
✓ Reset successful

3. Testing inference with dummy data...
✓ Inference successful. Action shape: torch.Size([1, 6])

4. Running latency benchmark (10 requests)...
   Request 1/10: 145.2ms
   Request 2/10: 132.8ms
   ...
✓ Average latency: 138.5ms (±8.3ms)
✓ Min: 125.1ms, Max: 156.7ms

==================================================
All tests passed! ✓
==================================================
```

### 延迟分析

典型延迟分解 (AutoDL + 国内网络):

| 组件 | 延迟 |
|------|------|
| 模型推理 (SmolVLA on GPU) | 50-80ms |
| 数据序列化/反序列化 | 5-10ms |
| 网络往返 (RTT) | 30-80ms |
| **总计** | **85-170ms** |

## 🐛 故障排查

### 问题 1: 连接超时

**症状**: `TimeoutError: Server did not respond in time`

**解决方案**:
1. 检查AutoDL防火墙是否开放端口
2. 确认服务器正在运行: `ps aux | grep policy_server`
3. 测试网络连接: `ping your-autodl-ip`

### 问题 2: CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 在服务器启动前设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 或者使用CPU
python policy_server_simple.py --device=cpu
```

### 问题 3: 序列化错误

**症状**: `pickle.UnpicklingError` 或类似错误

**解决方案**:
- 确保服务器和客户端使用相同版本的PyTorch
- 检查Python版本是否一致 (推荐3.10+)

### 问题 4: 延迟过高

**优化建议**:
1. **降低图像分辨率**:
   ```python
   # 在客户端压缩图像
   import cv2
   resized = cv2.resize(image, (224, 224))
   ```

2. **使用JPEG压缩**:
   ```python
   _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
   ```

3. **更换网络**:
   - 使用专线或VPN
   - 选择物理距离更近的AutoDL区域

## 🔐 安全建议

⚠️ **当前实现不安全** - 仅用于开发和测试

生产环境建议:

1. **添加身份验证**:
   ```python
   # 在请求中添加token
   request = {
       "command": "select_action",
       "token": "your-secret-token",
       "observation": observation
   }
   ```

2. **使用TLS加密**:
   - 使用`zmq.CURVE`进行加密
   - 或者通过SSH隧道转发端口

3. **限流和监控**:
   - 限制每个客户端的请求速率
   - 记录所有请求日志

## 📚 进阶使用

### 自定义策略加载

```python
# 在服务器端加载本地训练的策略
python policy_server_simple.py \
    --policy_path=/path/to/your/custom/policy \
    --device=cuda
```

### 批量推理 (提高吞吐量)

修改服务器以支持批量请求:
```python
# 在 policy_server_simple.py 中
def _handle_select_action_batch(self, request: dict) -> dict:
    observations = request.get("observations", [])  # 列表
    # 批量推理
    actions = self.policy.select_action(batch_observations)
    return {"actions": actions}
```

### 多客户端支持

服务器已支持多客户端 (ZMQ自动处理队列)。
每个客户端请求会按顺序处理。

如需并行处理，可以使用`zmq.ROUTER-DEALER`模式。

## 🎯 下一步

1. ✅ 基本远程推理
2. ⬜ 添加TLS加密
3. ⬜ 实现gRPC版本 (更高性能)
4. ⬜ 支持模型热更新
5. ⬜ 添加Prometheus监控
6. ⬜ 实现自动重连机制

## 📞 支持

遇到问题? 
- 提交Issue: https://github.com/huggingface/lerobot/issues
- Discord: https://discord.com/invite/s3KuuzsPFb


