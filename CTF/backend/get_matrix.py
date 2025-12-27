#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将游戏状态转换为二维列表并输出到终端
根据 AI_Design.md 和 matrix_util.py 的设计实现
"""

import json
import os
from lib.matrix_util import CTFMatrixConverter


def load_game_data(backend_path):
    """加载游戏初始化和状态数据"""
    init_file = os.path.join(backend_path, "example_init.json")
    status_file = os.path.join(backend_path, "example_plan_next_actions.json")
    
    with open(init_file, 'r', encoding='utf-8') as f:
        init_data = json.load(f)
    
    with open(status_file, 'r', encoding='utf-8') as f:
        status_data = json.load(f)
    
    return init_data, status_data


def get_matrix_as_2d_list(init_data, status_data, width=20, height=20):
    """获取游戏状态的二维列表"""
    # 创建转换器
    converter = CTFMatrixConverter(width=width, height=height)
    
    # 初始化静态地图（墙、障碍、区域等）
    converter.initialize_static_map(init_data)
    
    # 转换当前状态为矩阵
    matrix = converter.convert_to_matrix(status_data)
    
    # 将 numpy 数组转换为 Python 原生的二维列表
    matrix_list = matrix.tolist()
    
    return matrix_list


def print_2d_list(matrix_list):
    """Print 2D list to terminal"""
    print("\n" + "="*80)
    print("Game State 2D List:")
    print("="*80)
    print("[")
    for row in matrix_list:
        # Format output, each element takes 2 digits, filled with 0
        formatted_row = "[" + ", ".join(f"{cell:2d}" for cell in row) + "]"
        print(f"  {formatted_row},")
    print("]")
    print("="*80)
    
    # Python format for direct use
    print("\n" + "="*80)
    print("Python Format (Ready to Copy):")
    print("="*80)
    print(matrix_list)
    print("="*80 + "\n")


def main():
    """Main function"""
    # Get backend path
    backend_path = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Reading game data from: {backend_path}\n")
    
    # Load game data
    init_data, status_data = load_game_data(backend_path)
    print("OK - Loaded initialization data")
    print("OK - Loaded status data\n")
    
    # Get 2D list
    matrix_list = get_matrix_as_2d_list(init_data, status_data)
    print(f"OK - Generated {len(matrix_list)}x{len(matrix_list[0])} matrix\n")
    
    # Output to terminal
    print_2d_list(matrix_list)
    
    # Output ID descriptions
    print("\n" + "="*80)
    print("ID Descriptions (from AI_Design.md):")
    print("="*80)
    print("00-02: Player 1-3")
    print("03-05: Player 1-3 (With Flag)")
    print("06-08: Opposite Player 0-2")
    print("09-11: Opposite Player 0-2 (With Flag)")
    print("12: Prison")
    print("13: Home")
    print("14: Home Already With Flag")
    print("15: Oppo Home")
    print("16: Barrier")
    print("17: Blank")
    print("18: Flag")
    print("19: Oppo Flag")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
