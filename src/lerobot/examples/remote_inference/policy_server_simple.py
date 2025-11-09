#!/usr/bin/env python3
"""
简单的策略推理服务器 - 基于ZMQ实现
可以部署到AutoDL等远程GPU服务器上

使用方法:
    python policy_server_simple.py \
        --policy_path=lerobot/smolvla_base \
        --port=5555 \
        --device=cuda

依赖:
    pip install pyzmq torch
"""

import argparse
import io
import logging
import time
from typing import Any

import torch
import zmq

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyInferenceServer:
    """远程策略推理服务器"""
    
    def __init__(self, policy_path: str, device: str = "cuda", port: int = 5555):
        logger.info(f"Loading policy from {policy_path}")
        self.policy = self._load_policy(policy_path, device)
        self.device = device
        
        # 初始化ZMQ socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)  # Request-Reply pattern
        self.socket.bind(f"tcp://*:{port}")
        
        logger.info(f"Policy server started on port {port}")
        logger.info(f"Device: {device}")
        logger.info("Waiting for requests...")
        
    def _load_policy(self, policy_path: str, device: str) -> PreTrainedPolicy:
        """加载策略模型"""
        try:
            # 尝试直接加载预训练策略
            from lerobot.common.policies.pretrained import PreTrainedPolicy
            policy = PreTrainedPolicy.from_pretrained(policy_path)
        except Exception as e:
            logger.warning(f"Failed to load as pretrained: {e}")
            # 如果失败，尝试使用factory
            from lerobot.configs.policies import PreTrainedConfig
            config = PreTrainedConfig.from_pretrained(policy_path)
            from lerobot.common.policies.factory import make_policy
            policy = make_policy(config)
        
        policy.to(device)
        policy.eval()
        return policy
    
    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """序列化tensor为bytes"""
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()
    
    def _deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """从bytes反序列化tensor"""
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)
    
    def _process_observation(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """处理观测数据，转换为tensor并移动到正确的设备"""
        processed = {}
        for key, value in observation.items():
            if isinstance(value, bytes):
                # 如果是序列化的tensor
                tensor = self._deserialize_tensor(value)
            elif isinstance(value, torch.Tensor):
                tensor = value
            else:
                # 尝试转换为tensor
                tensor = torch.tensor(value)
            
            processed[key] = tensor.to(self.device)
        
        return processed
    
    def handle_request(self, request: dict) -> dict:
        """处理客户端请求"""
        command = request.get("command", "select_action")
        
        if command == "select_action":
            return self._handle_select_action(request)
        elif command == "reset":
            return self._handle_reset()
        elif command == "ping":
            return {"status": "pong", "timestamp": time.time()}
        else:
            return {"error": f"Unknown command: {command}"}
    
    def _handle_select_action(self, request: dict) -> dict:
        """处理动作推理请求"""
        start_time = time.time()
        
        try:
            # 获取观测数据
            observation = request.get("observation", {})
            observation = self._process_observation(observation)
            
            # 推理
            with torch.no_grad():
                action = self.policy.select_action(observation)
            
            # 移回CPU并序列化
            action_cpu = action.cpu()
            action_bytes = self._serialize_tensor(action_cpu)
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            logger.info(f"Inference time: {inference_time:.2f}ms")
            
            return {
                "action": action_bytes,
                "inference_time_ms": inference_time,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _handle_reset(self) -> dict:
        """重置策略状态"""
        try:
            self.policy.reset()
            return {"status": "reset"}
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def serve(self):
        """主服务循环"""
        try:
            while True:
                # 接收请求 (使用pickle进行简单序列化)
                import pickle
                message = self.socket.recv()
                request = pickle.loads(message)
                
                logger.debug(f"Received request: {request.get('command', 'unknown')}")
                
                # 处理请求
                response = self.handle_request(request)
                
                # 发送响应
                self.socket.send(pickle.dumps(response))
                
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            self.socket.close()
            self.context.term()


def main():
    parser = argparse.ArgumentParser(description="Remote Policy Inference Server")
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to the policy model (local path or HuggingFace repo)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port to listen on"
    )
    
    args = parser.parse_args()
    
    server = PolicyInferenceServer(
        policy_path=args.policy_path,
        device=args.device,
        port=args.port
    )
    
    server.serve()


if __name__ == "__main__":
    main()


