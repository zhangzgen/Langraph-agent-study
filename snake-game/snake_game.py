"""
🐍 贪吃蛇游戏
使用 Python 和 Pygame 制作的经典贪吃蛇游戏
"""

import pygame
import random
import sys
from enum import Enum

# 初始化 Pygame
pygame.init()

# 游戏常量
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // CELL_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // CELL_SIZE
FPS = 10

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

# 方向枚举
class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Snake:
    """蛇类"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置蛇的状态"""
        # 初始位置在屏幕中央
        start_x = GRID_WIDTH // 2
        start_y = GRID_HEIGHT // 2
        
        # 蛇的初始身体（3个格子）
        self.body = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ]
        
        self.direction = Direction.RIGHT
        self.grow = False
    
    def move(self):
        """移动蛇"""
        head_x, head_y = self.body[0]
        
        # 根据方向计算新的头部位置
        if self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        
        # 在头部插入新位置
        self.body.insert(0, new_head)
        
        # 如果不需要增长，移除尾部
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
    
    def change_direction(self, new_direction):
        """改变蛇的方向（防止180度转弯）"""
        # 检查是否是相反方向
        if (self.direction == Direction.UP and new_direction == Direction.DOWN) or \
           (self.direction == Direction.DOWN and new_direction == Direction.UP) or \
           (self.direction == Direction.LEFT and new_direction == Direction.RIGHT) or \
           (self.direction == Direction.RIGHT and new_direction == Direction.LEFT):
            return
        
        self.direction = new_direction
    
    def check_collision(self):
        """检查碰撞"""
        head_x, head_y = self.body[0]
        
        # 检查是否撞墙
        if head_x < 0 or head_x >= GRID_WIDTH or head_y < 0 or head_y >= GRID_HEIGHT:
            return True
        
        # 检查是否撞到自己
        if (head_x, head_y) in self.body[1:]:
            return True
        
        return False
    
    def eat(self):
        """吃食物"""
        self.grow = True

class Food:
    """食物类"""
    
    def __init__(self):
        self.position = (0, 0)
        self.randomize()
    
    def randomize(self):
        """随机生成食物位置"""
        self.position = (
            random.randint(0, GRID_WIDTH - 1),
            random.randint(0, GRID_HEIGHT - 1)
        )
    
    def draw(self, surface):
        """绘制食物"""
        rect = pygame.Rect(
            self.position[0] * CELL_SIZE,
            self.position[1] * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
        pygame.draw.rect(surface, RED, rect)
        pygame.draw.rect(surface, DARK_GREEN, rect, 1)

class Game:
    """游戏类"""
    
    def __init__(self):
        # 创建游戏窗口
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("🐍 贪吃蛇游戏")
        
        # 创建时钟
        self.clock = pygame.time.Clock()
        
        # 创建字体
        self.font_large = pygame.font.Font(None, 74)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # 游戏对象
        self.snake = Snake()
        self.food = Food()
        
        # 游戏状态
        self.score = 0
        self.high_score = 0
        self.game_over = False
        self.paused = False
        
        # 确保食物不在蛇身上
        self.food.randomize()
        while self.food.position in self.snake.body:
            self.food.randomize()
    
    def handle_events(self):
        """处理游戏事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                # 游戏结束时的按键处理
                if self.game_over:
                    if event.key == pygame.K_r:
                        self.restart_game()
                    elif event.key == pygame.K_ESCAPE:
                        return False
                    continue
                
                # 暂停/继续
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    continue
                
                # 方向控制
                if not self.paused:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.snake.change_direction(Direction.UP)
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.snake.change_direction(Direction.DOWN)
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.snake.change_direction(Direction.LEFT)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.snake.change_direction(Direction.RIGHT)
        
        return True
    
    def update(self):
        """更新游戏状态"""
        if self.game_over or self.paused:
            return
        
        # 移动蛇
        self.snake.move()
        
        # 检查是否吃到食物
        if self.snake.body[0] == self.food.position:
            self.snake.eat()
            self.score += 10
            
            # 更新最高分
            if self.score > self.high_score:
                self.high_score = self.score
            
            # 生成新食物
            self.food.randomize()
            while self.food.position in self.snake.body:
                self.food.randomize()
        
        # 检查碰撞
        if self.snake.check_collision():
            self.game_over = True
    
    def draw(self):
        """绘制游戏画面"""
        # 清屏
        self.screen.fill(BLACK)
        
        # 绘制游戏区域边框
        pygame.draw.rect(self.screen, WHITE, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 2)
        
        # 绘制网格（可选，增加视觉效果）
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (WINDOW_WIDTH, y))
        
        # 绘制食物
        self.food.draw(self.screen)
        
        # 绘制蛇
        for i, (x, y) in enumerate(self.snake.body):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            # 蛇头用不同颜色
            if i == 0:
                pygame.draw.rect(self.screen, YELLOW, rect)
            else:
                pygame.draw.rect(self.screen, GREEN, rect)
            
            pygame.draw.rect(self.screen, DARK_GREEN, rect, 1)
        
        # 绘制分数
        score_text = self.font_small.render(f"分数: {self.score}", True, WHITE)
        high_score_text = self.font_small.render(f"最高分: {self.high_score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(high_score_text, (WINDOW_WIDTH - high_score_text.get_width() - 10, 10))
        
        # 绘制暂停提示
        if self.paused:
            pause_text = self.font_large.render("游戏暂停", True, WHITE)
            pause_rect = pause_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(pause_text, pause_rect)
            
            continue_text = self.font_small.render("按空格键继续", True, WHITE)
            continue_rect = continue_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            self.screen.blit(continue_text, continue_rect)
        
        # 绘制游戏结束画面
        if self.game_over:
            # 半透明遮罩
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            # 游戏结束文字
            game_over_text = self.font_large.render("游戏结束", True, RED)
            game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
            self.screen.blit(game_over_text, game_over_rect)
            
            # 最终分数
            final_score_text = self.font_medium.render(f"最终分数: {self.score}", True, WHITE)
            final_score_rect = final_score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)
            
            # 重新开始提示
            restart_text = self.font_small.render("按 R 键重新开始", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 70))
            self.screen.blit(restart_text, restart_rect)
            
            # 退出提示
            exit_text = self.font_small.render("按 ESC 键退出", True, WHITE)
            exit_rect = exit_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 110))
            self.screen.blit(exit_text, exit_rect)
        
        # 更新显示
        pygame.display.flip()
    
    def restart_game(self):
        """重新开始游戏"""
        self.snake.reset()
        self.food.randomize()
        while self.food.position in self.snake.body:
            self.food.randomize()
        self.score = 0
        self.game_over = False
        self.paused = False
    
    def run(self):
        """运行游戏主循环"""
        running = True
        
        while running:
            # 处理事件
            running = self.handle_events()
            
            # 更新游戏状态
            self.update()
            
            # 绘制画面
            self.draw()
            
            # 控制帧率
            self.clock.tick(FPS)
        
        # 退出 Pygame
        pygame.quit()
        sys.exit()

def main():
    """主函数"""
    print("🐍 贪吃蛇游戏")
    print("=" * 40)
    print("游戏控制:")
    print("  方向键 或 WASD: 控制蛇的移动方向")
    print("  空格键: 暂停/继续游戏")
    print("  R 键: 重新开始游戏")
    print("  ESC 键: 退出游戏")
    print("=" * 40)
    print("正在启动游戏...")
    
    # 创建并运行游戏
    game = Game()
    game.run()

if __name__ == "__main__":
    main()