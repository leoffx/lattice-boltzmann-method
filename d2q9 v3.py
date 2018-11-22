import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.signal as signal

# DEFINE:
height = 150
width = 150
omega = 1.  # omega = 1/tau

u = np.zeros((2, height, width))
fin = np.zeros((9, height, width))

rho = 200. * np.ones((height, width)) + np.random.rand(height, width)

#shan-chen
G = -75.
wiX = np.flip(
    np.flip([[-1. / 36., 0., 1. / 36.], [-1. / 9., 0, 1. / 9.],
             [-1. / 36., 0., 1. / 36.]],
            axis=0),
    axis=1)
wiY = np.flip(
    np.flip([[1. / 36., 1. / 9., 1. / 36.], [0, 0, 0],
             [-1. / 36., -1. / 9., -1. / 36.]],
            axis=0),
    axis=1)


# streaming step
def streaming():
    global fin, rho

    fin[2, :, :] = np.roll(fin[2, :, :], 1, axis=0)
    fin[5, :, :] = np.roll(fin[5, :, :], 1, axis=0)
    fin[6, :, :] = np.roll(fin[6, :, :], 1, axis=0)

    fin[4, :, :] = np.roll(fin[4, :, :], -1, axis=0)
    fin[8, :, :] = np.roll(fin[8, :, :], -1, axis=0)
    fin[7, :, :] = np.roll(fin[7, :, :], -1, axis=0)

    fin[1, :, :] = np.roll(fin[1, :, :], 1, axis=1)
    fin[5, :, :] = np.roll(fin[5, :, :], 1, axis=1)
    fin[8, :, :] = np.roll(fin[8, :, :], 1, axis=1)

    fin[3, :, :] = np.roll(fin[3, :, :], -1, axis=1)
    fin[7, :, :] = np.roll(fin[7, :, :], -1, axis=1)
    fin[6, :, :] = np.roll(fin[6, :, :], -1, axis=1)
    rho = np.sum(fin, axis=0)


# collision step
def collision():
    global fin, rho

    ##shan-chen
    psi = 4. * np.exp(-200. / rho)
    somX = signal.convolve2d(psi, wiX, 'same', 'wrap')
    somY = signal.convolve2d(psi, wiY, 'same', 'wrap')
    Fx = -G * psi * somX
    Fy = -G * psi * somY

    u[0, :, :] = (fin[1, :, :] + fin[5, :, :] + fin[8, :, :] - fin[3, :, :] -
                  fin[7, :, :] - fin[6, :, :]) / rho
    u[1, :, :] = (fin[2, :, :] + fin[5, :, :] + fin[6, :, :] - fin[4, :, :] -
                  fin[7, :, :] - fin[8, :, :]) / rho

    #incorporating the force
    u[0, :, :] += Fx / (omega * rho)
    u[1, :, :] += Fy / (omega * rho)

    u2 = u[0, :, :]**2 + u[1, :, :]**2
    uxuy = u[0, :, :] * u[1, :, :]
    um32u2 = 1. - 1.5 * u2  #1 minus 3/2 of u**2

    fin[0, :, :] = fin[0, :, :] * (1. - omega) + omega * 4. / 9. * rho * (
        um32u2)
    fin[1, :, :] = fin[1, :, :] * (1. - omega) + omega * 1. / 9. * rho * (
        3. * u[0, :, :] + 4.5 * u[0, :, :]**2 + um32u2)
    fin[2, :, :] = fin[2, :, :] * (1. - omega) + omega * 1. / 9. * rho * (
        3. * u[1, :, :] + 4.5 * u[1, :, :]**2 + um32u2)
    fin[3, :, :] = fin[3, :, :] * (1. - omega) + omega * 1. / 9. * rho * (
        -3. * u[0, :, :] + 4.5 * u[0, :, :]**2 + um32u2)
    fin[4, :, :] = fin[4, :, :] * (1. - omega) + omega * 1. / 9. * rho * (
        -3. * u[1, :, :] + 4.5 * u[1, :, :]**2 + um32u2)
    fin[5, :, :] = fin[5, :, :] * (1. - omega) + omega * 1. / 36. * rho * (
        3. * (u[0, :, :] + u[1, :, :]) + 4.5 * (u2 + 2. * uxuy) + um32u2)
    fin[6, :, :] = fin[6, :, :] * (1. - omega) + omega * 1. / 36. * rho * (
        3. * (-u[0, :, :] + u[1, :, :]) + 4.5 * (u2 - 2. * uxuy) + um32u2)
    fin[7, :, :] = fin[7, :, :] * (1. - omega) + omega * 1. / 36. * rho * (
        3. * (-u[0, :, :] - u[1, :, :]) + 4.5 * (u2 + 2. * uxuy) + um32u2)
    fin[8, :, :] = fin[8, :, :] * (1. - omega) + omega * 1. / 36. * rho * (
        3. * (u[0, :, :] - u[1, :, :]) + 4.5 * (u2 - 2. * uxuy) + um32u2)


# PLOT LOOP
theFig = plt.figure(figsize=(8, 3))


def nextStep(arg):
    global rho
    for _ in range(10):
        collision()
        streaming()

    fluidImage = plt.imshow(rho)

    return (fluidImage, )


animate = matplotlib.animation.FuncAnimation(
    theFig, nextStep, interval=1, blit=True)
plt.show()
