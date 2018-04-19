import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pygame
import time
import random
from pygame.locals import *

class blobEnv(gym.Env):
    """
    gym.Env encapsulates the environment with arbitrary behind-the-scenes dynamics.
    It acts as a "wrapper" around the already-existing game, forming the bridge
    between reinforcement learning and the game itself.
    """
    # ‘human’ means the program should render to the current display,
    # it returns nothing but allows a human to see the program visually.
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Any one-time setup is run in this constructor method
        """

        # The Space object corresponding to valid actions
        self.action_space = spaces.Discrete(4)

        # Initialises all imported pygame modules
        pygame.init()

        # Sets the values for screen size
        self.display_width = 1002
        self.display_height = 720

        # Stores the game display window
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))

        # Sets positions for the starting location of the hard-coded entity
        self.pos_x = self.display_width / 1.2
        self.pos_y = self.display_height / 1.2

        # Creates a list to store bullets fired by the agent
        self.bullets = []

        # Creates a list to store bullets fired by the hard-coded entity
        self.bullets2 = []

        # Sets positions for the starting location of the agent
        self.x = (self.display_width * 0.08)
        self.y = (self.display_height * 0.2)

        # A variable used for resetting the y co-ordinate
        self.y_change = 0

        # Speed of agent's movements
        self.blob_speed = 2

        # Speed of hard-coded entities movements
        self.velocity = [2, 2]

        # Tracks the amount of times agent hits opposing entity
        self.score = 0

        # Tracks the amount of times the agent was hit by a bullet
        self.lives = 3

        # Variable for the size of the bullet
        self.bullet_width = 36
        self.bullet_height = 15

        # Variable for the size of the agent and opposing entity
        self.blob_width = 51
        self.blob_height = 51

        # Timer for bullet firing delay
        self.previous_time = 100
        self.previous_time2 = 100

        # Counter for bullet firing delay
        self.step_counter = 0

    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. When the end of the episode
        is reached, reset() is called to reset the environment's internal state

        :param action: an action provided by the environment
        :return outputs: observation (state), reward, done, info
        """

        # Variable to be fed into the neural network
        reward = 0.0

        # Increments 1 to bullet firing counter
        self.step_counter += 1

        # Makes hard-coded entity move
        self.pos_x += self.velocity[0]
        self.pos_y += self.velocity[1]

        # If the left wall is hit, the x-axis along which the hard-coded entity is
        # travelling is reversed
        if self.pos_x + self.blob_width > self.display_width or self.pos_x < 601:
            self.velocity[0] = -self.velocity[0]

        # If the top of the screen is hit, the y-axis along which the hard-coded entity is
        # travelling is reversed
        if self.pos_y + self.blob_height > self.display_height or self.pos_y < 0:
            self.velocity[1] = -self.velocity[1]

        # Sets the speed at which the hard-coded entities bullets travel
        for b in range(len(self.bullets2)):
            self.bullets2[b][0] -= 6

        # Removes bullet if the x-axis it is travelling along falls below 0
        for bullet in self.bullets2:
            if bullet[0] < 0:
                self.bullets2.remove(bullet)

        # Sets a delay for when hard-coded entity can fire a bullet
        if self.step_counter - self.previous_time2 > 90:
            self.previous_time2 = self.step_counter
            self.bullets2.append([self.pos_x + 25, self.pos_y])

        # Sets the speed at which the agents' bullets travel
        for b in range(len(self.bullets)):
            self.bullets[b][0] += 6

        # Removes bullet if it passes 1005 on the x-axis
        for bullet in self.bullets:
            if bullet[0] > 1005:
                self.bullets.remove(bullet)

        # If the actions chosen if 0, agent fires a bullet.
        # A delay is set between when each bullet can be fired
        if action == 0:
            if self.step_counter - self.previous_time > 90:
                self.previous_time = self.step_counter
                self.bullets.append([self.x + 25, self.y + 24])

        # NOOP (no-operation) if the action chosen is 3
        if action == 3:
            None

        # Prevents agent from moving beyond the visible area at the top of the screen
        if self.y < 0:
            self.y = 0

        # If the action chosen is 1, the agent moves up the screen
        if action == 1:
            self.y_change = -self.blob_speed

        # Prevents agent from moving beyond the visible area at the bottom of the screen
        if self.y > self.display_height - self.blob_height:
            self.y = self.display_height - self.blob_height

        # If the action chosen is 2, the agent moves down the screen
        if action == 2:
            self.y_change = self.blob_speed

        # Reset y to new position
        self.y += self.y_change

        # Bullet collision detection, if bullet collides with hard-coded entity,
        # 1 is incremented to the score, and 2 is incremented to the reward
        # while the bullet is removed
        for bullet in self.bullets:
            if bullet[0] > self.pos_x and bullet[0] < self.pos_x + self.blob_width:
                if bullet[1] > self.pos_y and bullet[1] < self.pos_y + self.blob_height or bullet[
                    1] + self.bullet_height > self.pos_y and bullet[
                    1] + self.bullet_height < self.pos_y + self.blob_height:
                    self.bullets.remove(bullet)
                    self.score += 1
                    reward += 2

        # Bullet collision detection, if bullet collides with agent,
        # 1 is taken away from lives, and the reward value is reduced
        # by 2, while the bullet is removed
        for bullet in self.bullets2:
            if bullet[0] + self.bullet_width < self.x + self.blob_width and bullet[0] > self.x:
                if bullet[1] > self.y and bullet[1] < self.y + self.blob_height or bullet[
                    1] + self.bullet_height > self.y and bullet[1] + self.bullet_height < self.y + self.blob_height:
                    self.bullets2.remove(bullet)
                    self.lives -= 1
                    reward -= 2

        # Makes calls to the event queue to prevent display window from crashing
        pygame.event.pump()

        # Calls the render() method inside the class
        self.render()

        # A 3 dimensional array acting as the environment observation for a minimised version of the display
        state = pygame.surfarray.array3d(pygame.transform.scale(self.gameDisplay, (100, 100)))

        return state, reward, self.lives <= 0, {"lives": self.lives, "score": self.score}

    def reset(self):
        """
        Resets the state of the environment, returning an initial state corresponding to
        an observation
        :return state: the initial observation of the space
        """

        # Resets the previously defined variables and objects below
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
        self.render()
        state = pygame.surfarray.array3d(pygame.transform.scale(self.gameDisplay, (100, 100)))
        return state, 0, self.lives < 0, {"lives": self.lives, "score": self.score}

    def render(self, mode='human'):
        """
        Contains the graphics of the game
        :param mode: 'human' means program renders to the current display
        :return:
        """

        # Defines a colour using 'RGB' colour values
        self.black = (0, 0, 0)
        self.blue = (53, 155, 255)

        # Loads an image to represent the entity and agent
        self.blobImage = pygame.image.load('blob2.png')

        # Loads an image to represent the bullets
        self.bulletpicture = pygame.image.load("bullet.png")

        # Changes background surface to blue
        self.gameDisplay.fill(self.blue)

        # Displays 2 walls near the middle of the display screen
        pygame.draw.line(self.gameDisplay, self.black, (601, self.display_height), (601, 0), 3)
        pygame.draw.line(self.gameDisplay, self.black, (401, self.display_height), (401, 0), 3)

        # Displays the hard-coded entity on the screen
        self.gameDisplay.blit(self.blobImage, (self.pos_x, self.pos_y))

        # Displays the agent on the screen
        self.gameDisplay.blit(self.blobImage, (self.x, self.y))

        # Re-sizes the agents bullets shape to allow the neural network and motion tracer
        # to differentiate between bullets fired by the agents and bullets fired by the
        # hard coded entity
        for bullet in self.bullets:
            self.gameDisplay.blit(pygame.transform.scale(self.bulletpicture, (self.bullet_width, self.bullet_height)),
                                  pygame.Rect(bullet[0], bullet[1], 0, 0))
        for bullet in self.bullets2:
            self.gameDisplay.blit(pygame.transform.scale(self.bulletpicture, (self.bullet_height, self.bullet_width)),
                                  pygame.Rect(bullet[0], bullet[1], 0, 0))

        # A minimised version of the display screen is shown on the top left of the display window
        self.gameDisplay.blit(pygame.transform.scale(self.gameDisplay, (100, 100)), (0, 0))

        # Updates the screen
        pygame.display.update()

#Dictionary to track what number correlates to which action
ACTION_MEANING = {
    0: "FIRE",
    1: "UP",
    2: "DOWN",
    3: "NOOP"
}
