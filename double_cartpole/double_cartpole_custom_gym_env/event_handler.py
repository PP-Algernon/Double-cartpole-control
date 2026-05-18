import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import sys

def pygame_events():
    """处理Pygame事件，主要是退出事件。"""
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            pygame.quit()
            sys.exit()

