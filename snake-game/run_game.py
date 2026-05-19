#!/usr/bin/env python3
"""
贪吃蛇游戏启动器
让用户选择运行经典版或增强版
"""

import subprocess
import sys
import os

def main():
    print("🐍 贪吃蛇游戏启动器")
    print("=" * 40)
    print("请选择游戏版本:")
    print("1. 增强版 (推荐) - 精美图标和草坪背景")
    print("2. 经典版 - 简洁界面")
    print("=" * 40)
    
    while True:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            print("\n正在启动增强版...")
            try:
                subprocess.run([sys.executable, "snake_game_enhanced.py"], check=True)
            except subprocess.CalledProcessError:
                print("启动失败，请检查 pygame 是否已安装")
            except KeyboardInterrupt:
                print("\n游戏已退出")
            break
        elif choice == "2":
            print("\n正在启动经典版...")
            try:
                subprocess.run([sys.executable, "snake_game.py"], check=True)
            except subprocess.CalledProcessError:
                print("启动失败，请检查 pygame 是否已安装")
            except KeyboardInterrupt:
                print("\n游戏已退出")
            break
        else:
            print("无效选择，请输入 1 或 2")

if __name__ == "__main__":
    main()