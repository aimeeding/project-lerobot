#!/usr/bin/env python3
"""
远程策略客户端 - 在本地笔记本上运行，连接到远程策略服务器

使用方法:
    # 测试连接
    python policy_client_simple.py \
        --server_ip=your.autodl.server.com \
        --port=5555 \
        --test

    # 与真机集成
    python policy_client_simple.py \
        --server_ip=your.autodl.server.com \
        --port=5555 \
        --robot_type=so100_follower \
        --robot_port=/dev/ttyUSB0

依赖:
    pip install pyzmq torch numpy
"""

import argparse
import io
import logging
import time
from typing import Any

import numpy as np
import torch
import zmq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RemotePolicyClient:
    """远程策略客户端"""
    
    def __init__(self, server_ip: str, port: int = 5555, timeout_ms: int = 5000):
        self.server_address = f"tcp://{server_ip}:{port}"
        self.timeout_ms = timeout_ms
        
        # 初始化ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(self.server_address)
        
        logger.info(f"Connected to policy server at {self.server_address}")
        
        # 统计信息
        self.total_requests = 0
        self.total_time = 0.0
    
    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """序列化tensor"""
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()
    
    def _deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """反序列化tensor"""
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)
    
    def _send_request(self, request: dict) -> dict:
        """发送请求并接收响应"""
        import pickle
        
        try:
            # 发送请求
            self.socket.send(pickle.dumps(request))
            
            # 接收响应
            message = self.socket.recv()
            response = pickle.loads(message)
            
            return response
            
        except zmq.Again:
            logger.error(f"Request timeout after {self.timeout_ms}ms")
            raise TimeoutError("Server did not respond in time")
        except Exception as e:
            logger.error(f"Communication error: {e}")
            raise
    
    def ping(self) -> dict:
        """测试服务器连接"""
        request = {"command": "ping"}
        response = self._send_request(request)
        return response
    
    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """请求动作推理"""
        start_time = time.time()
        
        # 准备观测数据 (序列化tensor)
        serialized_obs = {}
        for key, value in observation.items():
            if isinstance(value, torch.Tensor):
                serialized_obs[key] = self._serialize_tensor(value)
            elif isinstance(value, np.ndarray):
                serialized_obs[key] = self._serialize_tensor(torch.from_numpy(value))
            else:
                serialized_obs[key] = value
        
        # 构建请求
        request = {
            "command": "select_action",
            "observation": serialized_obs
        }
        
        # 发送请求
        response = self._send_request(request)
        
        if response.get("status") == "failed":
            raise RuntimeError(f"Server error: {response.get('error')}")
        
        # 反序列化动作
        action_bytes = response["action"]
        action = self._deserialize_tensor(action_bytes)
        
        # 统计
        total_time = (time.time() - start_time) * 1000
        inference_time = response.get("inference_time_ms", 0)
        network_time = total_time - inference_time
        
        self.total_requests += 1
        self.total_time += total_time
        
        logger.debug(
            f"Total: {total_time:.1f}ms "
            f"(Inference: {inference_time:.1f}ms, Network: {network_time:.1f}ms)"
        )
        
        return action
    
    def reset(self):
        """重置策略状态"""
        request = {"command": "reset"}
        response = self._send_request(request)
        
        if response.get("status") != "reset":
            raise RuntimeError(f"Reset failed: {response.get('error')}")
        
        logger.info("Policy reset")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        if self.total_requests == 0:
            return {"avg_latency_ms": 0, "total_requests": 0}
        
        return {
            "avg_latency_ms": self.total_time / self.total_requests,
            "total_requests": self.total_requests,
            "total_time_s": self.total_time / 1000
        }
    
    def close(self):
        """关闭连接"""
        self.socket.close()
        self.context.term()


def test_connection(client: RemotePolicyClient):
    """测试与服务器的连接"""
    logger.info("=" * 50)
    logger.info("Testing connection to policy server...")
    logger.info("=" * 50)
    
    # 1. Ping测试
    logger.info("1. Testing ping...")
    try:
        response = client.ping()
        logger.info(f"✓ Ping successful: {response}")
    except Exception as e:
        logger.error(f"✗ Ping failed: {e}")
        return False
    
    # 2. Reset测试
    logger.info("2. Testing reset...")
    try:
        client.reset()
        logger.info("✓ Reset successful")
    except Exception as e:
        logger.error(f"✗ Reset failed: {e}")
        return False
    
    # 3. 推理测试
    logger.info("3. Testing inference with dummy data...")
    try:
        # 创建虚拟观测
        dummy_obs = {
            "observation.state": torch.randn(1, 6),  # 假设6个关节
            "observation.images": [torch.randn(1, 3, 224, 224)]  # 假设224x224 RGB图像
        }
        
        action = client.select_action(dummy_obs)
        logger.info(f"✓ Inference successful. Action shape: {action.shape}")
        
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        return False
    
    # 4. 性能测试
    logger.info("4. Running latency benchmark (10 requests)...")
    latencies = []
    for i in range(10):
        start = time.time()
        action = client.select_action(dummy_obs)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        logger.info(f"   Request {i+1}/10: {latency:.1f}ms")
    
    logger.info(f"✓ Average latency: {np.mean(latencies):.1f}ms (±{np.std(latencies):.1f}ms)")
    logger.info(f"✓ Min: {np.min(latencies):.1f}ms, Max: {np.max(latencies):.1f}ms")
    
    logger.info("=" * 50)
    logger.info("All tests passed! ✓")
    logger.info("=" * 50)
    
    return True


def run_with_robot(client: RemotePolicyClient, robot_config: dict):
    """与真实机器人集成运行"""
    logger.info("Starting robot control with remote policy...")
    
    # 导入并初始化机器人
    from lerobot.common.robots import make_robot_from_config
    
    robot = make_robot_from_config(robot_config)
    robot.connect()
    
    try:
        # 重置策略
        client.reset()
        
        # 控制循环
        episode_length = 1000
        for step in range(episode_length):
            # 获取观测
            observation = robot.get_observation()
            
            # 转换为tensor格式 (假设observation已经是正确格式)
            obs_tensor = {}
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    obs_tensor[key] = torch.from_numpy(value).unsqueeze(0)
                else:
                    obs_tensor[key] = value
            
            # 远程推理
            action = client.select_action(obs_tensor)
            
            # 执行动作
            action_dict = {}
            for i, motor_name in enumerate(robot.action_features.keys()):
                action_dict[motor_name] = action[0, i].item()
            
            robot.send_action(action_dict)
            
            if step % 100 == 0:
                stats = client.get_stats()
                logger.info(
                    f"Step {step}/{episode_length}, "
                    f"Avg latency: {stats['avg_latency_ms']:.1f}ms"
                )
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.disconnect()
        
        # 显示统计
        stats = client.get_stats()
        logger.info("=" * 50)
        logger.info("Session statistics:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Average latency: {stats['avg_latency_ms']:.1f}ms")
        logger.info(f"  Total time: {stats['total_time_s']:.1f}s")
        logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Remote Policy Client")
    parser.add_argument(
        "--server_ip",
        type=str,
        required=True,
        help="IP address of the policy server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port of the policy server"
    )
    parser.add_argument(
        "--timeout_ms",
        type=int,
        default=5000,
        help="Request timeout in milliseconds"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run connection tests instead of robot control"
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        help="Robot type (e.g., so100_follower)"
    )
    parser.add_argument(
        "--robot_port",
        type=str,
        help="Robot serial port"
    )
    
    args = parser.parse_args()
    
    # 创建客户端
    client = RemotePolicyClient(
        server_ip=args.server_ip,
        port=args.port,
        timeout_ms=args.timeout_ms
    )
    
    try:
        if args.test:
            # 运行测试
            test_connection(client)
        else:
            # 与机器人集成
            if not args.robot_type or not args.robot_port:
                logger.error("Please specify --robot_type and --robot_port for robot control")
                return
            
            robot_config = {
                "type": args.robot_type,
                "port": args.robot_port,
            }
            run_with_robot(client, robot_config)
    
    finally:
        client.close()


if __name__ == "__main__":
    main()


