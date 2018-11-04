import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.signal as signal
import random

# DEFINIR:
height = 100
width = 200
viscosity = 0.1  # viscosidade // proporcional ao inverso de reynolds // não diminuir muito
omega = np.ones(
    (height, width)) / (3 * viscosity + .5)  # parametro de relaxamento

#shan chen
psi_0 = 4.
rho_0 = 200.

G = -120.
wi = np.array([[1. / 36, 1. / 9, 1. / 36], [1. / 9, 0, 1. / 9],
               [1. / 36, 1. / 9, 1. / 36]])

u0 = 0.1  # velocidade inicial // não aumentar muito

u = np.zeros((2, height, width))
fin = np.zeros((9, height, width))

# inicialização da densidade, dado u0
fin[0, :, :] = 4. / 9. * (1 - 1.5 * u0**2)
fin[1, :, :] = 1. / 9. * (1 + 1.5 * u0**2)
fin[2, :, :] = 1. / 9. * (1 - 1.5 * u0**2)
fin[3, :, :] = 1. / 9. * (1 + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[4, :, :] = 1. / 9. * (1 - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[5, :, :] = 1. / 36. * (1 + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[6, :, :] = 1. / 36. * (1 + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[7, :, :] = 1. / 36. * (1 - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[8, :, :] = 1. / 36. * (1 - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
rho = np.sum(fin, axis=0)  # densidade macroscopica

#outro fluido
fluid = np.zeros((height, width), bool)
for i in range(height):
    for j in range(width):
        if (i - height / 2)**2 + (j - 50)**2 < 100:
            fluid[i, j] = True
v_f = .01
omega[fluid] = 1 / (3 * v_f + .5)
"""fin1 = fin[1, 0, 0]
fin3 = fin[3, 0, 0]
fin5 = fin[5, 0, 0]
fin6 = fin[6, 0, 0]
fin7 = fin[7, 0, 0]
fin8 = fin[8, 0, 0]"""


# streaming step
def streaming(fin):
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
    fin[6, :, :] = np.roll(fin[6, :, :], -1, axis=1)
    fin[7, :, :] = np.roll(fin[7, :, :], -1, axis=1)


# collision step
def collision(fin):
    global rho

    rho = np.sum(fin, axis=0)
    psi = psi_0 * np.exp(-rho_0 / rho)

    som = signal.convolve2d(psi, wi, 'same', 'wrap')

    F = -G * psi * som

    u[0, :, :] = (fin[1, :, :] + fin[5, :, :] + fin[8, :, :] - fin[3, :, :] -
                  fin[6, :, :] - fin[7, :, :]) / rho
    u[1, :, :] = (fin[2, :, :] + fin[5, :, :] + fin[6, :, :] - fin[4, :, :] -
                  fin[8, :, :] - fin[7, :, :]) / rho

    u[0, :, :] += F / omega * rho
    u[1, :, :] += F / omega * rho

    u2 = u[0, :, :]**2 + u[1, :, :]**2
    uxuy = u[0, :, :] * u[1, :, :]
    um32u2 = 1 - 1.5 * (u2)  #um menos 3/2 de u**2

    fin[0, :, :] = fin[0, :, :] * (1 - omega) + omega * 4 / 9 * rho * (um32u2)
    fin[2, :, :] = fin[2, :, :] * (1 - omega) + omega * 1 / 9 * rho * (
        3 * u[1, :, :] + 4.5 * u[1, :, :]**2 + um32u2)
    fin[4, :, :] = fin[4, :, :] * (1 - omega) + omega * 1 / 9 * rho * (
        -3 * u[1, :, :] + 4.5 * u[1, :, :]**2 + um32u2)
    fin[1, :, :] = fin[1, :, :] * (1 - omega) + omega * 1 / 9 * rho * (
        3 * u[0, :, :] + 4.5 * u[0, :, :]**2 + um32u2)
    fin[3, :, :] = fin[3, :, :] * (1 - omega) + omega * 1 / 9 * rho * (
        -3 * u[0, :, :] + 4.5 * u[0, :, :]**2 + um32u2)
    fin[5, :, :] = fin[5, :, :] * (1 - omega) + omega * 1 / 36 * rho * (
        3 * (u[0, :, :] + u[1, :, :]) + 4.5 * (u2 + 2 * uxuy) + um32u2)
    fin[8, :, :] = fin[8, :, :] * (1 - omega) + omega * 1 / 36 * rho * (
        3 * (u[0, :, :] - u[1, :, :]) + 4.5 * (u2 - 2 * uxuy) + um32u2)
    fin[6, :, :] = fin[6, :, :] * (1 - omega) + omega * 1 / 36 * rho * (
        3 * (-u[0, :, :] + u[1, :, :]) + 4.5 * (u2 - 2 * uxuy) + um32u2)
    fin[7, :, :] = fin[7, :, :] * (1 - omega) + omega * 1 / 36 * rho * (
        3 * (-u[0, :, :] - u[1, :, :]) + 4.5 * (u2 + 2 * uxuy) + um32u2)

    # fluxo no inicio do quadro
    """fin[1, :, :][:, 0] = fin1
    fin[3, :, :][:, 0] = fin3
    fin[5, :, :][:, 0] = fin5
    fin[8, :, :][:, 0] = fin8
    fin[6, :, :][:, 0] = fin6
    fin[7, :, :][:, 0] = fin7"""


# PLOT LOOP
theFig = plt.figure(figsize=(8, 3))
u_plot = np.abs(u[0, :, :], u[1, :, :])


def nextStep(arg):
    global rho
    for _ in range(20):  #quantos calculos vão ser feitos por passo
        streaming(fin)
        collision(fin)

    u_plot = np.abs(u[0, :, :], u[1, :, :])

    fluidImage = plt.imshow(u_plot)

    return (fluidImage, )


animate = matplotlib.animation.FuncAnimation(
    theFig, nextStep, interval=1, blit=True)
plt.show()