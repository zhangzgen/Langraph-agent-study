"""
🐍 贪吃蛇游戏 - 增强版
使用 Python 和 Pygame 制作的经典贪吃蛇游戏
包含精美图标和草坪背景
"""

import pygame
import random
import sys
import math
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
BROWN = (139, 69, 19)
LIGHT_GREEN = (144, 238, 144)
DARK_BROWN = (101, 67, 33)
SKY_BLUE = (135, 206, 235)

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
        # 创建蛇头图标
        self.head_icon = self.create_snake_head_icon()
        # 创建蛇身图标
        self.body_icon = self.create_snake_body_icon()
    
    def create_snake_head_icon(self):
        """创建蛇头图标"""
        surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        
        # 绘制蛇头（圆形）
        pygame.draw.circle(surface, YELLOW, (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 2 - 2)
        pygame.draw.circle(surface, DARK_GREEN, (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 2 - 2, 2)
        
        # 绘制眼睛
        eye_size = 4
        # 左眼
        pygame.draw.circle(surface, BLACK, (CELL_SIZE // 2 - 4, CELL_SIZE // 2 - 3), eye_size)
        pygame.draw.circle(surface, WHITE, (CELL_SIZE // 2 - 4, CELL_SIZE // 2 - 3), eye_size - 2)
        # 右眼
        pygame.draw.circle(surface, BLACK, (CELL_SIZE // 2 + 4, CELL_SIZE // 2 - 3), eye_size)
        pygame.draw.circle(surface, WHITE, (CELL_SIZE // 2 + 4, CELL_SIZE // 2 - 3), eye_size - 2)
        
        # 绘制舌头
        tongue_points = [
            (CELL_SIZE // 2, CELL_SIZE // 2 + 5),
            (CELL_SIZE // 2 - 3, CELL_SIZE // 2 + 10),
            (CELL_SIZE // 2 + 3, CELL_SIZE // 2 + 10)
        ]
        pygame.draw.polygon(surface, RED, tongue_points)
        
        return surface
    
    def create_snake_body_icon(self):
        """创建蛇身图标"""
        surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        
        # 绘制蛇身（圆角矩形）
        rect = pygame.Rect(2, 2, CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(surface, GREEN, rect, border_radius=5)
        pygame.draw.rect(surface, DARK_GREEN, rect, 2, border_radius=5)
        
        # 添加鳞片纹理
        for i in range(3):
            for j in range(3):
                x = 5 + i * 5
                y = 5 + j * 5
                pygame.draw.circle(surface, DARK_GREEN, (x, y), 2)
        
        return surface
    
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
    
    def draw(self, surface):
        """绘制蛇"""
        for i, (x, y) in enumerate(self.body):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            # 蛇头用不同图标
            if i == 0:
                # 根据方向旋转蛇头
                rotated_head = self.head_icon.copy()
                if self.direction == Direction.UP:
                    rotated_head = pygame.transform.rotate(rotated_head, 90)
                elif self.direction == Direction.DOWN:
                    rotated_head = pygame.transform.rotate(rotated_head, -90)
                elif self.direction == Direction.LEFT:
                    rotated_head = pygame.transform.rotate(rotated_head, 180)
                # 右向不需要旋转
                
                surface.blit(rotated_head, rect)
            else:
                surface.blit(self.body_icon, rect)

class Food:
    """食物类"""
    
    def __init__(self):
        self.position = (0, 0)
        self.randomize()
        # 创建食物图标
        self.icon = self.create_food_icon()
        # 动画相关
        self.animation_timer = 0
        self.animation_speed = 0.1
    
    def create_food_icon(self):
        """创建食物图标（苹果）"""
        surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        
        # 绘制苹果主体
        apple_rect = pygame.Rect(3, 5, CELL_SIZE - 6, CELL_SIZE - 8)
        pygame.draw.ellipse(surface, RED, apple_rect)
        pygame.draw.ellipse(surface, DARK_GREEN, apple_rect, 2)
        
        # 绘制苹果高光
        highlight_rect = pygame.Rect(6, 8, 6, 4)
        pygame.draw.ellipse(surface, (255, 200, 200), highlight_rect)
        
        # 绘制苹果茎
        stem_points = [
            (CELL_SIZE // 2, 5),
            (CELL_SIZE // 2 + 2, 2),
            (CELL_SIZE // 2 - 2, 2)
        ]
        pygame.draw.polygon(surface, BROWN, stem_points)
        
        # 绘制叶子
        leaf_points = [
            (CELL_SIZE // 2 + 2, 3),
            (CELL_SIZE // 2 + 6, 1),
            (CELL_SIZE // 2 + 4, 5)
        ]
        pygame.draw.polygon(surface, GREEN, leaf_points)
        
        return surface
    
    def randomize(self):
        """随机生成食物位置"""
        self.position = (
            random.randint(0, GRID_WIDTH - 1),
            random.randint(0, GRID_HEIGHT - 1)
        )
    
    def update(self):
        """更新食物动画"""
        self.animation_timer += self.animation_speed
    
    def draw(self, surface):
        """绘制食物"""
        # 计算动画偏移（轻微上下浮动）
        offset_y = math.sin(self.animation_timer) * 2
        
        rect = pygame.Rect(
            self.position[0] * CELL_SIZE,
            self.position[1] * CELL_SIZE + offset_y,
            CELL_SIZE,
            CELL_SIZE
        )
        
        # 绘制阴影
        shadow_rect = pygame.Rect(
            self.position[0] * CELL_SIZE + 2,
            self.position[1] * CELL_SIZE + CELL_SIZE - 2,
            CELL_SIZE - 4,
            4
        )
        pygame.draw.ellipse(surface, (0, 0, 0, 50), shadow_rect)
        
        # 绘制食物
        surface.blit(self.icon, rect)

class GrassBackground:
    """草坪背景类"""
    
    def __init__(self):
        self.grass_colors = [
            (34, 139, 34),   # 森林绿
            (0, 128, 0),     # 绿色
            (50, 205, 50),   # 酸橙绿
            (144, 238, 144), # 浅绿
            (0, 100, 0),     # 深绿
        ]
        self.grass_patches = self.generate_grass_patches()
        self.flowers = self.generate_flowers()
    
    def generate_grass_patches(self):
        """生成草地块"""
        patches = []
        for _ in range(100):
            x = random.randint(0, WINDOW_WIDTH)
            y = random.randint(0, WINDOW_HEIGHT)
            size = random.randint(5, 15)
            color = random.choice(self.grass_colors)
            patches.append((x, y, size, color))
        return patches
    
    def generate_flowers(self):
        """生成花朵"""
        flowers = []
        flower_colors = [
            (255, 192, 203), # 粉色
            (255, 255, 0),   # 黄色
            (255, 165, 0),   # 橙色
            (255, 0, 0),     # 红色
            (255, 255, 255), # 白色
        ]
        
        for _ in range(20):
            x = random.randint(0, WINDOW_WIDTH)
            y = random.randint(0, WINDOW_HEIGHT)
            size = random.randint(3, 8)
            color = random.choice(flower_colors)
            flowers.append((x, y, size, color))
        
        return flowers
    
    def draw(self, surface):
        """绘制草坪背景"""
        # 绘制基础草地颜色
        surface.fill((124, 179, 66))  # 草地绿色
        
        # 绘制草地块
        for x, y, size, color in self.grass_patches:
            pygame.draw.circle(surface, color, (x, y), size)
        
        # 绘制草地纹理
        for i in range(0, WINDOW_WIDTH, 20):
            for j in range(0, WINDOW_HEIGHT, 20):
                # 随机草地纹理
                if random.random() > 0.7:
                    grass_height = random.randint(5, 15)
                    grass_color = random.choice(self.grass_colors)
                    pygame.draw.line(surface, grass_color, (i, j), (i, j - grass_height), 2)
        
        # 绘制花朵
        for x, y, size, color in self.flowers:
            # 花朵中心
            pygame.draw.circle(surface, YELLOW, (x, y), size // 2)
            # 花朵花瓣
            for angle in range(0, 360, 60):
                rad = math.radians(angle)
                petal_x = x + int(size * math.cos(rad))
                petal_y = y + int(size * math.sin(rad))
                pygame.draw.circle(surface, color, (petal_x, petal_y), size // 2)

class Game:
    """游戏类"""
    
    def __init__(self):
        # 创建游戏窗口
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("🐍 贪吃蛇游戏 - 增强版")
        
        # 创建时钟
        self.clock = pygame.time.Clock()
        
        # 创建字体
        self.font_large = pygame.font.Font(None, 74)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # 游戏对象
        self.snake = Snake()
        self.food = Food()
        self.background = GrassBackground()
        
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
        
        # 更新食物动画
        self.food.update()
        
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
        # 绘制草坪背景
        self.background.draw(self.screen)
        
        # 绘制游戏区域边框（木栅栏效果）
        self.draw_fence_border()
        
        # 绘制食物
        self.food.draw(self.screen)
        
        # 绘制蛇
        self.snake.draw(self.screen)
        
        # 绘制分数（带背景框）
        self.draw_score_board()
        
        # 绘制暂停提示
        if self.paused:
            self.draw_pause_screen()
        
        # 绘制游戏结束画面
        if self.game_over:
            self.draw_game_over_screen()
        
        # 更新显示
        pygame.display.flip()
    
    def draw_fence_border(self):
        """绘制木栅栏边框"""
        fence_color = BROWN
        fence_dark = DARK_BROWN
        
        # 上边框
        for x in range(0, WINDOW_WIDTH, 20):
            pygame.draw.rect(self.screen, fence_color, (x, 0, 20, 10))
            pygame.draw.rect(self.screen, fence_dark, (x, 0, 20, 10), 1)
            # 木纹效果
            pygame.draw.line(self.screen, fence_dark, (x + 5, 2), (x + 5, 8), 1)
            pygame.draw.line(self.screen, fence_dark, (x + 15, 2), (x + 15, 8), 1)
        
        # 下边框
        for x in range(0, WINDOW_WIDTH, 20):
            pygame.draw.rect(self.screen, fence_color, (x, WINDOW_HEIGHT - 10, 20, 10))
            pygame.draw.rect(self.screen, fence_dark, (x, WINDOW_HEIGHT - 10, 20, 10), 1)
            pygame.draw.line(self.screen, fence_dark, (x + 5, WINDOW_HEIGHT - 8), (x + 5, WINDOW_HEIGHT - 2), 1)
            pygame.draw.line(self.screen, fence_dark, (x + 15, WINDOW_HEIGHT - 8), (x + 15, WINDOW_HEIGHT - 2), 1)
        
        # 左边框
        for y in range(0, WINDOW_HEIGHT, 20):
            pygame.draw.rect(self.screen, fence_color, (0, y, 10, 20))
            pygame.draw.rect(self.screen, fence_dark, (0, y, 10, 20), 1)
            pygame.draw.line(self.screen, fence_dark, (2, y + 5), (8, y + 5), 1)
            pygame.draw.line(self.screen, fence_dark, (2, y + 15), (8, y + 15), 1)
        
        # 右边框
        for y in range(0, WINDOW_HEIGHT, 20):
            pygame.draw.rect(self.screen, fence_color, (WINDOW_WIDTH - 10, y, 10, 20))
            pygame.draw.rect(self.screen, fence_dark, (WINDOW_WIDTH - 10, y, 10, 20), 1)
            pygame.draw.line(self.screen, fence_dark, (WINDOW_WIDTH - 8, y + 5), (WINDOW_WIDTH - 2, y + 5), 1)
            pygame.draw.line(self.screen, fence_dark, (WINDOW_WIDTH - 8, y + 15), (WINDOW_WIDTH - 2, y + 15), 1)
    
    def draw_score_board(self):
        """绘制分数板"""
        # 分数背景框
        score_bg = pygame.Surface((200, 60))
        score_bg.set_alpha(200)
        score_bg.fill((139, 69, 19))  # 棕色背景
        pygame.draw.rect(score_bg, DARK_BROWN, (0, 0, 200, 60), 3)
        self.screen.blit(score_bg, (10, 10))
        
        # 分数文字
        score_text = self.font_small.render(f"分数: {self.score}", True, WHITE)
        high_score_text = self.font_small.render(f"最高分: {self.high_score}", True, YELLOW)
        self.screen.blit(score_text, (20, 15))
        self.screen.blit(high_score_text, (20, 40))
    
    def draw_pause_screen(self):
        """绘制暂停画面"""
        # 半透明遮罩
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # 暂停文字
        pause_text = self.font_large.render("游戏暂停", True, WHITE)
        pause_rect = pause_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 30))
        self.screen.blit(pause_text, pause_rect)
        
        # 继续提示
        continue_text = self.font_small.render("按空格键继续", True, LIGHT_GREEN)
        continue_rect = continue_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30))
        self.screen.blit(continue_text, continue_rect)
    
    def draw_game_over_screen(self):
        """绘制游戏结束画面"""
        # 半透明遮罩
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # 游戏结束文字
        game_over_text = self.font_large.render("游戏结束", True, RED)
        game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 80))
        self.screen.blit(game_over_text, game_over_rect)
        
        # 最终分数
        final_score_text = self.font_medium.render(f"最终分数: {self.score}", True, WHITE)
        final_score_rect = final_score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20))
        self.screen.blit(final_score_text, final_score_rect)
        
        # 最高分
        high_score_text = self.font_medium.render(f"最高分: {self.high_score}", True, YELLOW)
        high_score_rect = high_score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30))
        self.screen.blit(high_score_text, high_score_rect)
        
        # 重新开始提示
        restart_text = self.font_small.render("按 R 键重新开始", True, LIGHT_GREEN)
        restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 80))
        self.screen.blit(restart_text, restart_rect)
        
        # 退出提示
        exit_text = self.font_small.render("按 ESC 键退出", True, LIGHT_GREEN)
        exit_rect = exit_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 120))
        self.screen.blit(exit_text, exit_rect)
    
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
    print("🐍 贪吃蛇游戏 - 增强版")
    print("=" * 50)
    print("✨ 新特性:")
    print("  - 精美的蛇头和蛇身图标")
    print("  - 动态苹果食物图标")
    print("  - 草坪背景和木栅栏边框")
    print("  - 花朵装饰和草地纹理")
    print("  - 增强的分数显示界面")
    print("=" * 50)
    print("游戏控制:")
    print("  方向键 或 WASD: 控制蛇的移动方向")
    print("  空格键: 暂停/继续游戏")
    print("  R 键: 重新开始游戏")
    print("  ESC 键: 退出游戏")
    print("=" * 50)
    print("正在启动游戏...")
    
    # 创建并运行游戏
    game = Game()
    game.run()

if __name__ == "__main__":
    main()