from __future__ import annotations

import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

from .Cartpole import Cart, Pole, Track
from .event_handler import pygame_events


class DoubleCartpoleEnv(gym.Env):
    """与Gymnasium兼容的双倒立摆环境。

    ``render_sim`` 为了与原始Gym包兼容而保留。
    新代码应优先使用 ``render_mode="human"`` 或 ``render_mode="rgb_array"``。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_sim: bool | None = None,
        render_mode: str | None = None,
        n_steps: int = 1000,
        force_scale: float = 1200.0,
    ):
        super().__init__()

        if render_mode is None and render_sim:
            render_mode = "human"
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"unsupported render_mode={render_mode!r}")

        self.render_mode = render_mode
        self.render_sim = render_mode is not None
        self.max_time_steps = int(n_steps)
        self.force_scale = float(force_scale)
        self.dt = 1.0 / self.metadata["render_fps"]

        self.screen_width = 800
        self.screen_height = 800
        self.screen = None
        self.clock = None
        self.draw_options = None

        # 定义动作空间：[-1, 1] 之间的连续值，表示施加在小车上的力的比例
        self.min_action = np.array([-1], dtype=np.float32)
        self.max_action = np.array([1], dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, dtype=np.float32
        )

        # 定义观测空间：6个归一化到[-1, 1]的连续值
        self.min_observation = np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32)
        self.max_observation = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.min_observation, high=self.max_observation, dtype=np.float32
        )

        self.force = 0.0
        self.current_time_step = 0
        self.init_pymunk()

    def init_pygame(self):
        """初始化Pygame用于渲染。"""
        if self.screen is not None:
            return
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        else:
            # 如果是 "rgb_array" 模式，则创建一个离屏表面
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        pygame.display.set_caption("Double Cartpole Environment")
        self.clock = pygame.time.Clock()

        # 设置窗口图标
        script_dir = os.path.dirname(__file__)
        icon_path = os.path.join("img", "icon.png")
        icon_path = os.path.join(script_dir, icon_path)
        if self.render_mode == "human" and os.path.exists(icon_path):
            pygame.display.set_icon(pygame.image.load(icon_path))

    def init_pymunk(
        self,
        *,
        pole_1_angle: float | None = None,
        pole_2_angle: float | None = None,
    ):
        """初始化Pymunk物理引擎和环境对象。"""
        rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random.default_rng()

        self.force = 0.0
        self.current_time_step = 0
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -1000)

        if self.render_mode is not None:
            self.init_pygame()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            pymunk.pygame_util.positive_y_is_up = True

        initial_x = 400
        initial_y = 400
        self.target = 400  # 目标位置

        # 创建轨道
        self.track = Track(0, 400, 800, 400, 0, self.space)

        # 创建小车
        self.cart_mass = 1
        self.cart = Cart(
            initial_x,
            initial_y,
            80,
            40,
            self.cart_mass,
            (33, 93, 191),
            self.space,
        )

        # 创建第一个摆杆
        self.pole_1_mass = 1
        self.pole_1_length = 160
        pole_1_thickness = 15

        if pole_1_angle is None:
            # 缩小初始角度范围，降低初始难度
            pole_1_angle = float(
                rng.uniform(89.0 * np.pi / 180, 90.0 * np.pi / 180)
            )
        alpha = pole_1_angle

        pole_1_x = initial_x + self.pole_1_length * np.cos(alpha)
        pole_1_y = initial_y + self.pole_1_length * np.sin(alpha)
        self.pole_1 = Pole(
            pole_1_x,
            pole_1_y,
            initial_x,
            initial_y,
            pole_1_thickness,
            self.pole_1_mass,
            (66, 135, 245),
            self.space,
        )

        # 创建第二个摆杆
        self.pole_2_mass = 1
        self.pole_2_length = 160
        pole_2_thickness = 15

        if pole_2_angle is None:
            # 缩小初始角度范围，降低初始难度
            pole_2_angle = float(
                rng.uniform(89.0 * np.pi / 180, 90.0 * np.pi / 180)
            )
        betha = pole_2_angle

        pole_2_x = pole_1_x + self.pole_2_length * np.cos(betha)
        pole_2_y = pole_1_y + self.pole_2_length * np.sin(betha)
        self.pole_2 = Pole(
            pole_2_x,
            pole_2_y,
            pole_1_x,
            pole_1_y,
            pole_2_thickness,
            self.pole_2_mass,
            (119, 169, 248),
            self.space,
        )

        # 创建约束
        # 小车在轨道上的滑动约束
        self.left_slider = pymunk.GrooveJoint(
            self.track.body, self.cart.body, (0, 400), (800, 400), (-20, 0)
        )
        self.left_slider.error_bias = 0
        self.left_slider.collide_bodies = False
        self.space.add(self.left_slider)

        self.right_slider = pymunk.GrooveJoint(
            self.track.body, self.cart.body, (0, 400), (800, 400), (20, 0)
        )
        self.right_slider.error_bias = 0
        self.right_slider.collide_bodies = False
        self.space.add(self.right_slider)

        # 小车和第一个摆杆的枢轴关节
        self.pivot_1 = pymunk.PivotJoint(
            self.cart.body,
            self.pole_1.body,
            (0, 0),
            (-self.pole_1_length / 2, 0),
        )
        self.pivot_1.error_bias = 0
        self.pivot_1.collide_bodies = False
        self.space.add(self.pivot_1)

        # 第一个摆杆和第二个摆杆的枢轴关节
        self.pivot_2 = pymunk.PivotJoint(
            self.pole_1.body,
            self.pole_2.body,
            (self.pole_1_length / 2, 0),
            (-self.pole_2_length / 2, 0),
        )
        self.pivot_2.error_bias = 0
        self.pivot_2.collide_bodies = False
        self.space.add(self.pivot_2)

    def step(self, action):
        """执行一个环境步骤。"""
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.min_action, self.max_action)
        self.force = float(action[0]) * self.force_scale
        self.cart.body.apply_force_at_local_point((self.force, 0), (0, 0))

        # 模拟摩擦力
        pymunk.Body.update_velocity(self.pole_1.body, Vec2d(0, 0), 0.999, self.dt)
        pymunk.Body.update_velocity(self.pole_2.body, Vec2d(0, 0), 0.999, self.dt)
        pymunk.Body.update_velocity(self.cart.body, Vec2d(0, 0), 0.9999, self.dt)

        self.space.step(self.dt)
        self.current_time_step += 1

        # 奖励函数
        obs = self.get_observation()
        # 基础奖励为1，并根据离目标点的距离给予额外奖励
        reward = float(1 + 0.5 * (-np.abs(obs[5]) + 1))

        # 失去平衡的惩罚
        terminated = False
        if np.abs(obs[1]) >= 1 or np.abs(obs[2]) >= 1:
            terminated = True
            reward = -50

        # 小车撞墙的惩罚
        if np.abs(obs[5]) >= 1:
            terminated = True
            reward = -50

        # 时间耗尽则截断
        truncated = self.current_time_step >= self.max_time_steps

        if self.render_mode == "human":
            self.render()

        info = {"force": self.force}
        return obs, reward, terminated, truncated, info

    def get_observation(self):
        """获取并归一化环境观测值。"""
        # 小车速度
        cart_velocity = np.clip(self.cart.body.velocity_at_local_point((0, 0))[0]/610, -1, 1)
        # 第一个摆杆的角度
        pole_1_angle = -9*self.pole_1.body.angle/np.pi + 4.5
        pole_1_angle = np.clip(pole_1_angle, -1, 1)
        # 第二个摆杆的角度
        pole_2_angle = -9*self.pole_2.body.angle/np.pi + 4.5
        pole_2_angle = np.clip(pole_2_angle, -1, 1)
        # 第一个摆杆的角速度
        pole_1_angular_velocity = np.clip(self.pole_1.body.angular_velocity/15, -1, 1)
        # 第二个摆杆的角速度
        pole_2_angular_velocity = np.clip(self.pole_2.body.angular_velocity/15, -1, 1)

        # 计算离目标线的距离
        x = self.cart.body.position[0]
        if x < self.target:
            distance_x = np.clip((x/(self.target-40) - self.target/(self.target-40)) , -1, 0)
        else:
            distance_x = np.clip((-x/(self.target-760) + self.target/(self.target-760)) , 0, 1)

        return np.array(
            [
                cart_velocity,
                pole_1_angle,
                pole_2_angle,
                pole_1_angular_velocity,
                pole_2_angular_velocity,
                distance_x,
            ],
            dtype=np.float32,
        )

    def _to_screen(self, point):
        """将Pymunk世界坐标转换为Pygame屏幕坐标。"""
        return int(round(point[0])), int(round(self.screen_height - point[1]))

    def _draw_poly_shape(self, shape, color):
        """按Pymunk刚体当前姿态绘制多边形。"""
        vertices = [
            self._to_screen(shape.body.local_to_world(vertex))
            for vertex in shape.get_vertices()
        ]
        pygame.draw.polygon(self.screen, color, vertices)

    def render(self):
        """渲染环境的一帧。"""
        if self.render_mode is None:
            return None
        if self.screen is None:
            self.init_pygame()

        cart_pos = self.cart.body.position
        pivot_1 = cart_pos
        pivot_2 = self.pole_1.body.local_to_world((self.pole_1_length / 2, 0))
        pole_2_tip = self.pole_2.body.local_to_world((self.pole_2_length / 2, 0))
        scale = 1.0 / 12

        if self.render_mode == "human":
            pygame_events()

        self.screen.fill((243, 243, 243))

        pygame.draw.line(
            self.screen,
            (85, 94, 100),
            self._to_screen((0, 400)),
            self._to_screen((800, 400)),
            3,
        )
        # 绘制目标线
        pygame.draw.line(self.screen, (149, 165, 166), (400, 0), (400, 800), 1)

        # 绘制第二根摆杆的右侧角度限制参考线。
        x_prim = pivot_2[0] + (self.pole_2_length + 25) * np.cos(11 * np.pi / 18)
        y_prim = pivot_2[1] + (self.pole_2_length + 25) * np.sin(11 * np.pi / 18)
        pygame.draw.line(
            self.screen,
            (255, 26, 26),
            self._to_screen(pivot_2),
            self._to_screen((x_prim, y_prim)),
            4,
        )

        # 直接使用Pygame绘制主体，避免debug_draw坐标系和窗口后端差异导致黑屏。
        self._draw_poly_shape(self.cart.shape, (33, 93, 191))
        self._draw_poly_shape(self.pole_1.shape, (66, 135, 245))
        self._draw_poly_shape(self.pole_2.shape, (119, 169, 248))

        # 绘制枢轴点
        pygame.draw.circle(self.screen, (20, 52, 110), self._to_screen(pivot_1), 6)
        pygame.draw.circle(self.screen, (32, 80, 180), self._to_screen(pivot_2), 6)
        pygame.draw.circle(self.screen, (55, 120, 215), self._to_screen(pole_2_tip), 5)

        # 绘制最大力范围的可视化条
        force_y = self._to_screen(cart_pos)[1] + 1
        pygame.draw.line(
            self.screen,
            (179, 179, 179),
            (cart_pos.x - scale * self.force_scale, force_y),
            (cart_pos.x + scale * self.force_scale, force_y),
            4,
        )

        # 绘制当前施加的力的可视化条
        if self.force != 0:
            pygame.draw.line(
                self.screen,
                (255, 0, 0),
                (cart_pos.x, force_y),
                (cart_pos.x + scale * self.force, force_y),
                4,
            )

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        # 如果是 "rgb_array" 模式，返回图像数组
        return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态。"""
        super().reset(seed=seed)
        options = options or {}
        self.init_pymunk(
            pole_1_angle=options.get("pole_1_angle"),
            pole_2_angle=options.get("pole_2_angle"),
        )
        observation = self.get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, {}

    def close(self):
        """关闭环境并清理资源。"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        self.screen = None
        self.clock = None
        self.draw_options = None
