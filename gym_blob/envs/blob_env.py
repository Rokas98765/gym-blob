import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pygame
import time
import random
from pygame.locals import *


class BlobEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 30
                }

    def __init__(self):
        self.action_space = spaces.Discrete(6)
        pygame.init()
        self.clock = pygame.time.Clock()  # sets a clock
        self.display_width = 1002
        self.display_height = 720
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))
        self.pos_x = self.display_width / 1.2
        self.pos_y = self.display_height / 1.2
        self.bullets = []
        self.bullets2 = []
        self.x = (self.display_width * 0.08)
        self.y = (self.display_height * 0.2)
        self.x_change = 0
        self.y_change = 0
        self.blob_speed = 2
        self.velocity = [2, 2]
        self.score = 0
        self.lives = 3
        self.bullet_width = 36
        self.bullet_height = 15
        self.blob_width = 51
        self.blob_height = 51
        self.previous_time = 100
        self.previous_time2 = 100
        self.step_counter = 0

    def step(self, action):
        reward = 0.0
        self.step_counter += 1

        self.pos_x += self.velocity[0]
        self.pos_y += self.velocity[1]

        if self.pos_x + self.blob_width > self.display_width or self.pos_x < 601:
            self.velocity[0] = -self.velocity[0]

        if self.pos_y + self.blob_height > self.display_height or self.pos_y < 0:
            self.velocity[1] = -self.velocity[1]

        for b in range(len(self.bullets2)):
            self.bullets2[b][0] -= 6

        for bullet in self.bullets2:
            if bullet[0] < 0:
                self.bullets2.remove(bullet)

        if self.step_counter - self.previous_time2 > 50:
            self.previous_time2 = self.step_counter
            self.bullets2.append([self.pos_x + 25, self.pos_y + 24])

        for b in range(len(self.bullets)):
            self.bullets[b][0] += 6

        for bullet in self.bullets:
            if bullet[0] > 1005:
                self.bullets.remove(bullet)

        if action == "FIRE":
            if self.step_counter - self.previous_time > 50:
                self.previous_time = self.step_counter
                self.bullets.append([self.x + 25, self.y + 24])

        # If the player is holding down one key or the other the blob moves in that direction
        if self.x < 0:
            self.x = 0
        if self.x > 401 - self.blob_width:
            self.x = 401 - self.blob_width

        if self.y < 0:
            self.y = 0
        if action == "UP":
            self.y_change = -self.blob_speed
        if self.y > self.display_height - self.blob_height:
            self.y = self.display_height - self.blob_height
        if action == "DOWN":
            self.y_change = self.blob_speed

        # Reset x and y to new position
        self.x += self.x_change
        self.y += self.y_change

        for bullet in self.bullets:
            if bullet[0] > self.pos_x and bullet[0] < self.pos_x + self.blob_width:
                if bullet[1] > self.pos_y and bullet[1] < self.pos_y + self.blob_height or bullet[
                    1] + self.bullet_height > self.pos_y and bullet[
                    1] + self.bullet_height < self.pos_y + self.blob_height:
                    self.bullets.remove(bullet)
                    self.score += 1
                    reward += 2

        for bullet in self.bullets2:
            if bullet[0] + self.bullet_width < self.x + self.blob_width and bullet[0] > self.x:
                if bullet[1] > self.y and bullet[1] < self.y + self.blob_height or bullet[
                    1] + self.bullet_height > self.y and bullet[1] + self.bullet_height < self.y + self.blob_height:
                    self.bullets2.remove(bullet)
                    self.lives -= 1
                    reward -= 2
        env.render()
        state = pygame.surfarray.array2d(pygame.transform.scale(self.gameDisplay, (100, 100))).flatten()
        return state, reward, self.lives < 0, {"lives": self.lives, "score": self.score}

    def reset(self):
        # all the following code seems to define the initial game state
        self.bullets = []
        self.bullets2 = []
        self.x = (self.display_width * 0.08)
        self.y = (self.display_height * 0.2)
        self.x_change = 0
        self.y_change = 0
        self.blob_speed = 2
        self.velocity = [2, 2]
        self.score = 0
        self.lives = 3
        self.pos_x = self.display_width / 1.2
        self.pos_y = self.display_height / 1.2
        self.previous_time = 100
        self.previous_time2 = 100
        self.step_counter = 0
        env.render()
        state = pygame.surfarray.array2d(pygame.transform.scale(self.gameDisplay, (100, 100))).flatten()
        return state, 0, self.lives < 0, {"lives": self.lives, "score": self.score}

    def render(self, mode='human', close=False):
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.blue = (53, 155, 255)

        self.blobImage = pygame.image.load('blob2.png')
        self.bulletpicture = pygame.image.load("bullet.png")
        self.bullet_width = 36
        self.bullet_height = 15
        self.blob_width = 51
        self.blob_height = 51

        self.gameDisplay.fill(self.blue)  # changes background surface
        pygame.draw.line(self.gameDisplay, self.black, (601, self.display_height), (601, 0), 3)
        pygame.draw.line(self.gameDisplay, self.black, (401, self.display_height), (401, 0), 3)
        self.gameDisplay.blit(self.blobImage, (self.pos_x, self.pos_y))
        self.gameDisplay.blit(self.blobImage, (self.x, self.y))

        for bullet in self.bullets:
            self.gameDisplay.blit(pygame.transform.scale(self.bulletpicture, (self.bullet_width, self.bullet_height)),
                                  pygame.Rect(bullet[0], bullet[1], 0, 0))
        for bullet in self.bullets2:
            self.gameDisplay.blit(pygame.transform.scale(self.bulletpicture, (self.bullet_height, self.bullet_width)),
                                  pygame.Rect(bullet[0], bullet[1], 0, 0))

        self.gameDisplay.blit(pygame.transform.scale(self.gameDisplay, (100, 100)), (0, 0))
        pygame.display.update()  # update screen
        self.clock.tick(120)  # moves frame on (fps in parameters)


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "DOWN",
    4: "UPFIRE",
    5: "DOWNFIRE",
}
