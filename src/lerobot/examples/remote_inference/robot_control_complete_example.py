#!/usr/bin/env python3
"""
完整的远程推理 + 真机控制示例

展示如何将LeRobot机器人与远程AutoDL策略服务器集成

使用方法:
    python robot_control_complete_example.py \
        --server_ip=your.autodl.com \
        --server_port=5555 \
        --robot_type=so100_follower \
        --robot_port=/dev/ttyUSB0 \
        --task="Pick up the red cube"

依赖:
    pip install pyzmq torch numpy opencv-python
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import zmq

# 导入我们的工具函数
from utils import (
    check_action_format,
    check_observation_format,
    deserialize_tensor,
    policy_action_to_robot_action,
    robot_obs_to_policy_input,
    serialize_observation,
    serialize_tensor,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RemotePolicyController:
    """
    远程策略控制器 - 连接机器人和远程策略服务器
    """
    
    def __init__(
        self,
        robot,
        server_ip: str,
        server_port: int = 5555,
        motor_names: list[str] | None = None,
        timeout_ms: int = 5000,
        task: str = "",
    ):
        """
        Args:
            robot: LeRobot机器人实例
            server_ip: 策略服务器IP
            server_port: 策略服务器端口
            motor_names: 关节名称列表（按顺序）
            timeout_ms: 请求超时（毫秒）
            task: 任务描述（用于VLA模型）
        """
        self.robot = robot
        self.server_address = f"tcp://{server_ip}:{server_port}"
        self.timeout_ms = timeout_ms
        self.task = task
        
        # 自动获取电机名称
        if motor_names is None:
            # 从robot.action_features提取电机名称
            self.motor_names = [
                key.replace(".pos", "") 
                for key in robot.action_features.keys() 
                if key.endswith(".pos")
            ]
        else:
            self.motor_names = motor_names
        
        logger.info(f"Motor names: {self.motor_names}")
        
        # 初始化ZMQ连接
        self._init_zmq()
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "total_time": 0.0,
            "inference_times": [],
            "network_times": [],
            "total_times": [],
        }
    
    def _init_zmq(self):
        """初始化ZMQ连接"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(self.server_address)
        logger.info(f"Connected to policy server at {self.server_address}")
    
    def _send_request(self, request: dict) -> dict:
        """发送请求并接收响应"""
        import pickle
        
        try:
            self.socket.send(pickle.dumps(request))
            message = self.socket.recv()
            return pickle.loads(message)
        except zmq.Again:
            raise TimeoutError(f"Request timeout after {self.timeout_ms}ms")
    
    def reset_policy(self):
        """重置策略状态"""
        logger.info("Resetting policy...")
        request = {"command": "reset"}
        response = self._send_request(request)
        
        if response.get("status") != "reset":
            raise RuntimeError(f"Reset failed: {response.get('error')}")
        
        logger.info("Policy reset successful")
    
    def get_action(self, robot_obs: dict) -> tuple[dict, dict]:
        """
        获取远程推理动作
        
        Args:
            robot_obs: 机器人原始观测
        
        Returns:
            (action_dict, info_dict)
            - action_dict: 机器人格式的动作 {"motor.pos": value}
            - info_dict: 推理信息 {"inference_time_ms": ..., "total_time_ms": ...}
        """
        step_start = time.time()
        
        # 1. 转换为LeRobot策略格式
        policy_obs = robot_obs_to_policy_input(
            robot_obs,
            self.motor_names,
            device="cpu"  # 客户端在CPU上
        )
        
        # 验证格式（仅第一次）
        if self.stats["total_steps"] == 0:
            logger.info("Validating observation format...")
            check_observation_format(policy_obs)
        
        # 2. 添加任务描述（如果是VLA模型）
        if self.task:
            policy_obs["task"] = [self.task]
        
        # 3. 序列化并发送
        obs_bytes = serialize_observation(policy_obs)
        
        request = {
            "command": "select_action",
            "observation": obs_bytes,
        }
        
        network_start = time.time()
        response = self._send_request(request)
        network_time = (time.time() - network_start) * 1000
        
        if response.get("status") == "failed":
            raise RuntimeError(f"Inference failed: {response.get('error')}")
        
        # 4. 反序列化动作
        action_bytes = response["action"]
        action_tensor = deserialize_tensor(action_bytes, device="cpu")
        
        # 验证格式（仅第一次）
        if self.stats["total_steps"] == 0:
            logger.info("Validating action format...")
            check_action_format(action_tensor, expected_dim=len(self.motor_names))
        
        # 5. 转换为机器人格式
        action_dict = policy_action_to_robot_action(action_tensor, self.motor_names)
        
        # 6. 收集统计信息
        total_time = (time.time() - step_start) * 1000
        inference_time = response.get("inference_time_ms", 0)
        
        info = {
            "inference_time_ms": inference_time,
            "network_time_ms": network_time,
            "total_time_ms": total_time,
            "conversion_time_ms": total_time - network_time,
        }
        
        self.stats["total_steps"] += 1
        self.stats["total_time"] += total_time
        self.stats["inference_times"].append(inference_time)
        self.stats["network_times"].append(network_time)
        self.stats["total_times"].append(total_time)
        
        return action_dict, info
    
    def run_episode(
        self,
        max_steps: int = 1000,
        log_interval: int = 10,
        fps: int = 30,
    ) -> dict:
        """
        运行一个episode
        
        Args:
            max_steps: 最大步数
            log_interval: 日志打印间隔
            fps: 目标帧率
        
        Returns:
            episode统计信息
        """
        logger.info(f"Starting episode (max_steps={max_steps}, fps={fps})")
        
        # 重置策略
        self.reset_policy()
        
        # 控制循环
        step = 0
        episode_start = time.time()
        
        try:
            for step in range(max_steps):
                loop_start = time.time()
                
                # 1. 获取观测
                obs = self.robot.get_observation()
                
                # 2. 远程推理
                action, info = self.get_action(obs)
                
                # 3. 执行动作
                self.robot.send_action(action)
                
                # 4. 日志
                if step % log_interval == 0:
                    avg_total = np.mean(self.stats["total_times"][-log_interval:])
                    avg_inference = np.mean(self.stats["inference_times"][-log_interval:])
                    avg_network = np.mean(self.stats["network_times"][-log_interval:])
                    
                    logger.info(
                        f"Step {step}/{max_steps} | "
                        f"Total: {avg_total:.1f}ms "
                        f"(Inference: {avg_inference:.1f}ms, "
                        f"Network: {avg_network:.1f}ms)"
                    )
                
                # 5. 控制帧率
                elapsed = time.time() - loop_start
                sleep_time = max(0, 1.0/fps - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Episode interrupted by user")
        
        except Exception as e:
            logger.error(f"Episode failed: {e}", exc_info=True)
            raise
        
        finally:
            episode_time = time.time() - episode_start
            
            # 收集统计
            episode_stats = {
                "total_steps": step + 1,
                "episode_time_s": episode_time,
                "avg_fps": (step + 1) / episode_time,
                "avg_total_latency_ms": np.mean(self.stats["total_times"]),
                "avg_inference_time_ms": np.mean(self.stats["inference_times"]),
                "avg_network_time_ms": np.mean(self.stats["network_times"]),
                "std_total_latency_ms": np.std(self.stats["total_times"]),
            }
            
            logger.info("=" * 60)
            logger.info("Episode statistics:")
            logger.info(f"  Total steps: {episode_stats['total_steps']}")
            logger.info(f"  Episode time: {episode_stats['episode_time_s']:.1f}s")
            logger.info(f"  Average FPS: {episode_stats['avg_fps']:.1f}")
            logger.info(f"  Average latency: {episode_stats['avg_total_latency_ms']:.1f}ms "
                       f"(±{episode_stats['std_total_latency_ms']:.1f}ms)")
            logger.info(f"    - Inference: {episode_stats['avg_inference_time_ms']:.1f}ms")
            logger.info(f"    - Network: {episode_stats['avg_network_time_ms']:.1f}ms")
            logger.info("=" * 60)
        
        return episode_stats
    
    def close(self):
        """关闭连接"""
        self.socket.close()
        self.context.term()
        logger.info("Connection closed")


def main():
    parser = argparse.ArgumentParser(description="Robot Control with Remote Policy")
    
    # 服务器配置
    parser.add_argument("--server_ip", type=str, required=True, help="Policy server IP")
    parser.add_argument("--server_port", type=int, default=5555, help="Policy server port")
    parser.add_argument("--timeout_ms", type=int, default=5000, help="Request timeout (ms)")
    
    # 机器人配置
    parser.add_argument("--robot_type", type=str, required=True, help="Robot type (e.g., so100_follower)")
    parser.add_argument("--robot_port", type=str, required=True, help="Robot serial port")
    parser.add_argument("--robot_cameras", type=str, help="Camera config JSON string")
    
    # 任务配置
    parser.add_argument("--task", type=str, default="", help="Task description for VLA models")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--fps", type=int, default=30, help="Target control frequency")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    
    args = parser.parse_args()
    
    # 1. 初始化机器人
    logger.info(f"Initializing robot: {args.robot_type}")
    
    from lerobot.common.robots import make_robot_from_config
    
    robot_config = {
        "type": args.robot_type,
        "port": args.robot_port,
    }
    
    # 添加相机配置（如果提供）
    if args.robot_cameras:
        import json
        robot_config["cameras"] = json.loads(args.robot_cameras)
    
    robot = make_robot_from_config(robot_config)
    
    logger.info("Connecting to robot...")
    robot.connect()
    logger.info("Robot connected!")
    
    try:
        # 2. 创建远程策略控制器
        controller = RemotePolicyController(
            robot=robot,
            server_ip=args.server_ip,
            server_port=args.server_port,
            timeout_ms=args.timeout_ms,
            task=args.task,
        )
        
        # 3. 运行episode
        stats = controller.run_episode(
            max_steps=args.max_steps,
            log_interval=args.log_interval,
            fps=args.fps,
        )
        
        # 4. 保存统计（可选）
        import json
        stats_file = Path("episode_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_file}")
    
    finally:
        # 5. 清理
        logger.info("Disconnecting robot...")
        robot.disconnect()
        controller.close()
        logger.info("Done!")


if __name__ == "__main__":
    main()


