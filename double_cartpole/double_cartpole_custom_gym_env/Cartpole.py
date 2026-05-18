import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d


class Cart:
    """表示环境中的小车。"""

    def __init__(self, x, y, a, b, mass, RGB, space):
        """
        初始化小车对象。

        :param x: 小车中心的x坐标
        :param y: 小车中心的y坐标
        :param a: 小车的宽度
        :param b: 小车的高度
        :param mass: 小车的质量
        :param RGB: 小车的颜色 (R, G, B)
        :param space: Pymunk物理空间
        """
        self.shape = pymunk.Poly.create_box(None, size=(a, b))
        self.moment_of_inertia = pymunk.moment_for_poly(mass, self.shape.get_vertices())
        self.body = pymunk.Body(mass, self.moment_of_inertia, body_type=pymunk.Body.DYNAMIC)
        self.shape.body = self.body
        self.body.position = x, y
        self.shape.color = pygame.Color(RGB)
        self.shape.sensor = True  # 设置为传感器，使其不与其他对象发生物理碰撞
        space.add(self.body, self.shape)

class Pole:
    """表示环境中的摆杆。"""

    def __init__(self, x1, y1, x2, y2, thickness, mass, RGB, space):
        """
        初始化摆杆对象。

        :param x1, y1: 摆杆一端的坐标
        :param x2, y2: 摆杆另一端的坐标
        :param thickness: 摆杆的厚度
        :param mass: 摆杆的质量
        :param RGB: 摆杆的颜色 (R, G, B)
        :param space: Pymunk物理空间
        """
        x = (x1+x2)/2
        y = (y1+y2)/2
        length = abs(Vec2d(x1-x2, y1-y2))
        v_vec = Vec2d(x1-x, y1-y)
        x_vec = Vec2d(1, 0)
        angle = x_vec.get_angle_between(v_vec)

        self.shape = pymunk.Poly.create_box(None, size=(length+thickness, thickness))
        self.moment_of_inertia = pymunk.moment_for_poly(mass, self.shape.get_vertices())
        self.body = pymunk.Body(mass, self.moment_of_inertia, body_type=pymunk.Body.DYNAMIC)
        self.shape.body = self.body
        self.body.position = x, y
        self.body.angle = angle
        self.shape.color = pygame.Color(RGB)
        self.shape.sensor = True # 设置为传感器，使其不与其他对象发生物理碰撞
        space.add(self.body, self.shape)

class Track:
    """表示小车运动的轨道。"""

    def __init__(self, x1, y1, x2, y2, thickness, space):
        """
        初始化轨道对象。

        :param x1, y1: 轨道起点的坐标
        :param x2, y2: 轨道终点的坐标
        :param thickness: 轨道的厚度
        :param space: Pymunk物理空间
        """
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, (x1, y1), (x2, y2), thickness)
        self.shape.sensor = True # 设置为传感器，使其不与其他对象发生物理碰撞
        space.add(self.body, self.shape)
