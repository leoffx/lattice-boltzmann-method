import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

# DEFINIR:
height = 80
width = 200
viscosity = 0.1  # viscosidade
omega = 1 / (3 * viscosity + 0.5)  # parametro de relaxamento
u0 = 0.04  # velocidade incial

u = np.zeros((2, height, width))
fin = np.zeros((9, height, width))

# inicialização da densidade, dado u0
fin[0, :, :] = 4. / 9. * (np.ones((height, width)) - 1.5 * u0**2)
fin[1, :, :] = 1. / 9. * (np.ones((height, width)) - 1.5 * u0**2)
fin[2, :, :] = 1. / 9. * (np.ones((height, width)) - 1.5 * u0**2)
fin[3, :, :] = 1. / 9. * (np.ones(
    (height, width)) + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[4, :, :] = 1. / 9. * (np.ones(
    (height, width)) - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[5, :, :] = 1. / 36. * (np.ones(
    (height, width)) + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[6, :, :] = 1. / 36. * (np.ones(
    (height, width)) + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[7, :, :] = 1. / 36. * (np.ones(
    (height, width)) - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
fin[8, :, :] = 1. / 36. * (np.ones(
    (height, width)) - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
rho = np.sum(fin, axis=0)  # densidade macroscopica

u[0, :, :] = (
    fin[3, :, :] + fin[5, :, :] + fin[6, :, :] - fin[4, :, :] - fin[7, :, :] -
    fin[8, :, :]) / rho  # velocidade macroscopica em X
u[1, :, :] = (
    fin[1, :, :] + fin[5, :, :] + fin[7, :, :] - fin[2, :, :] - fin[6, :, :] -
    fin[8, :, :]) / rho  # velocidade macroscopica em Y

# obstáculo / objeto
obj = np.zeros((height, width), bool)
obj[32:48, 20:50] = True
obj[0, :] = True
obj[-1, :] = True
objN = np.roll(obj, 1, axis=0)
objS = np.roll(obj, -1, axis=0)
objE = np.roll(obj, 1, axis=1)
objW = np.roll(obj, -1, axis=1)
objNE = np.roll(objN, 1, axis=1)
objNW = np.roll(objN, -1, axis=1)
objSE = np.roll(objS, 1, axis=1)
objSW = np.roll(objS, -1, axis=1)


# streaming step
def streaming():
    global fin
    fin[1, :, :] = np.roll(fin[1, :, :], 1, axis=0)
    fin[5, :, :] = np.roll(fin[5, :, :], 1, axis=0)
    fin[7, :, :] = np.roll(fin[7, :, :], 1, axis=0)
    fin[2, :, :] = np.roll(fin[2, :, :], -1, axis=0)
    fin[6, :, :] = np.roll(fin[6, :, :], -1, axis=0)
    fin[8, :, :] = np.roll(fin[8, :, :], -1, axis=0)
    fin[3, :, :] = np.roll(fin[3, :, :], 1, axis=1)
    fin[5, :, :] = np.roll(fin[5, :, :], 1, axis=1)
    fin[6, :, :] = np.roll(fin[6, :, :], 1, axis=1)
    fin[4, :, :] = np.roll(fin[4, :, :], -1, axis=1)
    fin[7, :, :] = np.roll(fin[7, :, :], -1, axis=1)
    fin[8, :, :] = np.roll(fin[8, :, :], -1, axis=1)
    # condiçao de contorno no objeto
    fin[1, :, :][objN] = fin[2, :, :][obj]
    fin[2, :, :][objS] = fin[1, :, :][obj]
    fin[3, :, :][objE] = fin[4, :, :][obj]
    fin[4, :, :][objW] = fin[3, :, :][obj]
    fin[5, :, :][objNE] = fin[8, :, :][obj]
    fin[7, :, :][objNW] = fin[6, :, :][obj]
    fin[6, :, :][objSE] = fin[7, :, :][obj]
    fin[8, :, :][objSW] = fin[5, :, :][obj]


# collision step
def collision():
    global rho, u, fin
    rho = np.sum(fin, axis=0)
    u[0, :, :] = (fin[3, :, :] + fin[5, :, :] + fin[6, :, :] - fin[4, :, :] -
                  fin[7, :, :] - fin[8, :, :]) / rho
    u[1, :, :] = (fin[1, :, :] + fin[5, :, :] + fin[7, :, :] - fin[2, :, :] -
                  fin[6, :, :] - fin[8, :, :]) / rho

    u2 = u[0, :, :]**2 + u[1, :, :]**2
    omu215 = 1 - 1.5 * u2
    uxuy = u[0, :, :] * u[1, :, :]
    fin[0, :, :] = (1 - omega) * fin[0, :, :] + omega * 4. / 9. * rho * omu215
    fin[1, :, :] = (1 - omega) * fin[1, :, :] + omega * 1. / 9. * rho * (
        omu215 + 3 * u[1, :, :] + 4.5 * u[1, :, :]**2)
    fin[2, :, :] = (1 - omega) * fin[2, :, :] + omega * 1. / 9. * rho * (
        omu215 - 3 * u[1, :, :] + 4.5 * u[1, :, :]**2)
    fin[3, :, :] = (1 - omega) * fin[3, :, :] + omega * 1. / 9. * rho * (
        omu215 + 3 * u[0, :, :] + 4.5 * u[0, :, :]**2)
    fin[4, :, :] = (1 - omega) * fin[4, :, :] + omega * 1. / 9. * rho * (
        omu215 - 3 * u[0, :, :] + 4.5 * u[0, :, :]**2)
    fin[5, :, :] = (1 - omega) * fin[5, :, :] + omega * 1. / 36. * rho * (
        omu215 + 3 * (u[0, :, :] + u[1, :, :]) + 4.5 * (u2 + 2 * uxuy))
    fin[7, :, :] = (1 - omega) * fin[7, :, :] + omega * 1. / 36. * rho * (
        omu215 + 3 * (-u[0, :, :] + u[1, :, :]) + 4.5 * (u2 - 2 * uxuy))
    fin[6, :, :] = (1 - omega) * fin[6, :, :] + omega * 1. / 36. * rho * (
        omu215 + 3 * (u[0, :, :] - u[1, :, :]) + 4.5 * (u2 - 2 * uxuy))
    fin[8, :, :] = (1 - omega) * fin[8, :, :] + omega * 1. / 36. * rho * (
        omu215 + 3 * (-u[0, :, :] - u[1, :, :]) + 4.5 * (u2 + 2 * uxuy))
    # fluxo no fim do quadro
    fin[3, :, :][:, 0] = 1. / 9. * (1 + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
    fin[4, :, :][:, 0] = 1. / 9. * (1 - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
    fin[5, :, :][:, 0] = 1. / 36. * (1 + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
    fin[6, :, :][:, 0] = 1. / 36. * (1 + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
    fin[7, :, :][:, 0] = 1. / 36. * (1 - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
    fin[8, :, :][:, 0] = 1. / 36. * (1 - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)


# rotacional
def curl(ux, uy):
    return np.roll(
        uy, -1, axis=1) - np.roll(
            uy, 1, axis=1) - np.roll(
                ux, -1, axis=0) + np.roll(
                    ux, 1, axis=0)


u_plot = np.sqrt(u[0, :, :]**2 + u[1, :, :]**2)

# PLOT LOOP
theFig = plt.figure(figsize=(8, 3))
fluidImage = plt.imshow(
    u_plot,
    origin='lower',
    norm=plt.Normalize(-.1, .1),
    cmap=plt.get_cmap('jet'),
    interpolation='none')
bImageArray = np.zeros((height, width, 4), np.uint8)
bImageArray[obj, 3] = 255
objImage = plt.imshow(bImageArray, origin='lower', interpolation='none')


def nextStep(arg):
    for i in range(20):  #quantos calculos vão ser feitos por passo
        streaming()
        collision()
    u_plot = np.sqrt(u[0, :, :]**2 + u[1, :, :]**2)
    fluidImage.set_array(u_plot)

    return (fluidImage, objImage)


animate = matplotlib.animation.FuncAnimation(
    theFig, nextStep, interval=0.1, blit=True)
plt.show()