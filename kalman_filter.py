import numpy as np
import matplotlib.pyplot as plt
from filterpy import kalman
from filterpy import common

#def fx(x, dt):
#    F = np.array([[1, dt, 0, 0],
#                  [0, 1, 0, 0],
#                  [0, 0, 1, dt],
#                  [0, 0, 0, 1]], dtype=np.float)
#
#    return np.dot(F, x)
#
#
#def hx(x):
#    return np.array([x[0], x[2]])
#
#dt = 0.1
#
#points = kalman.MerweScaledSigmaPoints(4, alpha=0.1, beta=0.2, kappa=-1)
#
#kf = kalman.UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
#
#kf.x = np.array([-1., 1., -1., 1.])
#kf.P *= 0.2
#
#z_std = 0.1
#
#kf.R = np.diag([z_std**2, z_std**2])
#kf.Q = common.Q_discrete_white_noise(dim=4, dt=dt, var=0.1**2, block_size=2)
#
#zs = [[i + np.random.rand()*z_std, i + np.random.rand()*z_std] for i in range(50)]
#
#for z in zs:
#    kf.predict()
#    kf.update(z)
#    print(kf.x,  "log-likelihodd", kf.log_likelihood)


#def fx(x, dt):
#     # state transition function - predict next state based
#     # on constant velocity model x = vt + x_0
#     F = np.array([[1, dt, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, dt],
#                   [0, 0, 0, 1]], dtype=float)
#     return np.dot(F, x)
#
#def hx(x):
#   # measurement function - convert state into a measurement
#   # where measurements are [x_pos, y_pos]
#   return np.array([x[0], x[2]])
#dt = 0.1
## create sigma points to use in the filter. This is standard for Gaussian processes
#points = kalman.MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
#kf = kalman.UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
#kf.x = np.array([-1., 1., -1., 1]) # initial state
#kf.P *= 0.2 # initial uncertainty
#z_std = 0.1
#kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
#kf.Q = common.Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)
#zs = [[i+np.random.randn()*z_std, i+np.random.randn()*z_std] for i in range(50)] # measurements
#for z in zs:
#    kf.predict()
#    kf.update(z)
#    print(kf.x, 'log-likelihood', kf.log_likelihood)


def fx(x, dt):
    F = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1]], dtype=np.float)

    return np.dot(F, x)


def hx(x):
    return np.array([x[0]])

dt = 0.1

z_std = 0.25

a = np.linspace(0, 30, int(30./dt), True)

zs = np.sin(a)

zs_noise = [z + np.random.normal(scale=z_std) for z in zs]

points = kalman.MerweScaledSigmaPoints(n=3, alpha=.1, beta=2., kappa=-1)

kf = kalman.UnscentedKalmanFilter(dim_x=3, dim_z=1, dt=dt, fx=fx, hx=hx, points=points)

kf.x = np.array([0, 1., 0])
kf.P *= 0.1
kf.R = np.array([z_std**2])
kf.Q = common.Q_discrete_white_noise(dim=3, dt=dt, var=0.01**2, block_size=1)
#print(zs)
zs_f = []
for z in zs:
    kf.predict()
    kf.update(z)
    zs_f.append(kf.x[0])

plt.plot(a, zs, "b")
plt.plot(a, zs_noise, "r")
plt.plot(a, zs_f, "y")
plt.grid()
plt.show()

zs_pre = []
for i in range(100):
    kf.predict()
    zs_pre.append(kf.x[0])




#dt = 0.1
#kf = kalman.KalmanFilter(dim_x=3, dim_z=1)
#
#kf.x = np.array([0., 1., 0.])
#
#kf.F = F = np.array([[1, dt, 0.5 * dt ** 2],
#                     [0, 1, dt],
#                     [0, 0, 1]], dtype=np.float)
#
#kf.P *= 1.0
#z_std = 0.15
#kf.R = z_std
#
#kf.Q = common.Q_discrete_white_noise(dim=3, dt=dt, var=0.1)
#
#a = np.linspace(0, 20, int(20./dt), True)
#
#zs = np.sin(a)
#
#zs_noise = [z + np.random.normal(scale=z_std) for z in zs]
#zs_f = []
#for z in zs:
#    kf.predict()
#    kf.update(z)
#    zs_f.append(kf.x[0])
#
#plt.plot(a, zs, "b")
#plt.plot(a, zs_noise, "r")
#plt.plot(a, zs_f, "y")
#plt.grid()
#plt.show()
