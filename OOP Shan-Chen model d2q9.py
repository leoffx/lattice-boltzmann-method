import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.signal as signal


class newFluid:

    G = -6.
    wiX = [[1. / 36., 0., -1. / 36.], [1. / 9., 0, -1. / 9.],
           [1. / 36., 0., -1. / 36.]]

    wiY = [[1. / 36., 1. / 9., 1. / 36.], [0, 0, 0],
           [-1. / 36., -1. / 9., -1. / 36.]]
    omega = .8

    def __init__(self, height, width):
        self.heigth = height
        self.width = width
        self.u = np.zeros((2, height, width))
        self.fin = np.zeros((9, height, width))
        self.rho = .2 * np.ones(
            (height, width))  #+ .1 * np.random.rand(height, width)
        for i in range(height):
            for j in range(width):
                if (i - height / 2)**2 + (j - 50)**2 < 500:
                    self.rho[i, j] = 2.

    def collision(self):
        ##shan-chen
        psi = 1 - np.exp(-1 * self.rho)

        somX = signal.convolve2d(psi, self.wiX, 'same', 'wrap')
        somY = signal.convolve2d(psi, self.wiY, 'same', 'wrap')

        Fx = -self.G * psi * somX
        Fy = -self.G * psi * somY

        print(self.rho.max(), self.rho.min())

        self.u[0, :, :] = (self.fin[1, :, :] + self.fin[5, :, :] +
                           self.fin[8, :, :] - self.fin[3, :, :] -
                           self.fin[7, :, :] - self.fin[6, :, :]) / self.rho
        self.u[1, :, :] = (self.fin[2, :, :] + self.fin[5, :, :] +
                           self.fin[6, :, :] - self.fin[4, :, :] -
                           self.fin[7, :, :] - self.fin[8, :, :]) / self.rho

        #incorporating the force
        self.u[0, :, :] += Fx / (self.omega * self.rho)
        self.u[1, :, :] += Fy / (self.omega * self.rho)

        self.u[0, :, :] += .0005 / (self.omega * self.rho)

        u2 = self.u[0, :, :]**2 + self.u[1, :, :]**2
        uxuy = self.u[0, :, :] * self.u[1, :, :]
        um32u2 = 1. - 1.5 * u2  #1 minus 3/2 of u**2

        self.fin[0, :, :] = self.fin[0, :, :] * (
            1. - self.omega) + self.omega * 4. / 9. * self.rho * (um32u2)
        self.fin[1, :, :] = self.fin[1, :, :] * (
            1. - self.omega) + self.omega * 1. / 9. * self.rho * (
                3. * self.u[0, :, :] + 4.5 * self.u[0, :, :]**2 + um32u2)
        self.fin[2, :, :] = self.fin[2, :, :] * (
            1. - self.omega) + self.omega * 1. / 9. * self.rho * (
                3. * self.u[1, :, :] + 4.5 * self.u[1, :, :]**2 + um32u2)
        self.fin[3, :, :] = self.fin[3, :, :] * (
            1. - self.omega) + self.omega * 1. / 9. * self.rho * (
                -3. * self.u[0, :, :] + 4.5 * self.u[0, :, :]**2 + um32u2)
        self.fin[4, :, :] = self.fin[4, :, :] * (
            1. - self.omega) + self.omega * 1. / 9. * self.rho * (
                -3. * self.u[1, :, :] + 4.5 * self.u[1, :, :]**2 + um32u2)
        self.fin[5, :, :] = self.fin[5, :, :] * (
            1. - self.omega) + self.omega * 1. / 36. * self.rho * (
                3. * (self.u[0, :, :] + self.u[1, :, :]) + 4.5 *
                (u2 + 2. * uxuy) + um32u2)
        self.fin[6, :, :] = self.fin[6, :, :] * (
            1. - self.omega) + self.omega * 1. / 36. * self.rho * (
                3. * (-self.u[0, :, :] + self.u[1, :, :]) + 4.5 *
                (u2 - 2. * uxuy) + um32u2)
        self.fin[7, :, :] = self.fin[7, :, :] * (
            1. - self.omega) + self.omega * 1. / 36. * self.rho * (
                3. * (-self.u[0, :, :] - self.u[1, :, :]) + 4.5 *
                (u2 + 2. * uxuy) + um32u2)
        self.fin[8, :, :] = self.fin[8, :, :] * (
            1. - self.omega) + self.omega * 1. / 36. * self.rho * (
                3. * (self.u[0, :, :] - self.u[1, :, :]) + 4.5 *
                (u2 - 2. * uxuy) + um32u2)

    def streaming(self):
        self.fin[2, :, :] = np.roll(self.fin[2, :, :], 1, axis=0)
        self.fin[5, :, :] = np.roll(self.fin[5, :, :], 1, axis=0)
        self.fin[6, :, :] = np.roll(self.fin[6, :, :], 1, axis=0)

        self.fin[4, :, :] = np.roll(self.fin[4, :, :], -1, axis=0)
        self.fin[8, :, :] = np.roll(self.fin[8, :, :], -1, axis=0)
        self.fin[7, :, :] = np.roll(self.fin[7, :, :], -1, axis=0)

        self.fin[1, :, :] = np.roll(self.fin[1, :, :], 1, axis=1)
        self.fin[5, :, :] = np.roll(self.fin[5, :, :], 1, axis=1)
        self.fin[8, :, :] = np.roll(self.fin[8, :, :], 1, axis=1)

        self.fin[3, :, :] = np.roll(self.fin[3, :, :], -1, axis=1)
        self.fin[7, :, :] = np.roll(self.fin[7, :, :], -1, axis=1)
        self.fin[6, :, :] = np.roll(self.fin[6, :, :], -1, axis=1)

        self.rho = np.sum(self.fin, axis=0)


fluid = newFluid(150, 300)

# PLOT LOOP
theFig = plt.figure(figsize=(8, 3))


def nextStep(arg):
    for _ in range(10):
        fluid.collision()
        fluid.streaming()

    fluidImage = plt.imshow(fluid.rho)

    return (fluidImage, )


animate = matplotlib.animation.FuncAnimation(
    theFig, nextStep, interval=1, blit=True)
plt.show()
