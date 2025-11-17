"""

Path tracking simulation with LQR steering control and PID speed control.

author Atsushi Sakai (@Atsushi_twi)

"""
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
sys.path.append("../../PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise


Kp = 1.0  # speed proportional gain

# LQR parameter
# Q = np.eye(4) * 1
Q = np.diag([0.02, 0.0, 2.5, 0.0])
# R = np.eye(1) * 1
R = np.diag([5.0])

# parameters
dt = 0.01  # time tick[s]
L = 2.469  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(60.0)  # maximum steering angle[rad]

show_animation = True
#  show_animation = False


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):

    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state


def PIDControl(target, current):
    a = Kp * (target - current)

    return a


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T @ X @ A - A.T @ X @ B @ \
            la.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if (abs(Xn - X)).max() < eps:
            break
        X = Xn

    return Xn


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eigVals, eigVecs = la.eig(A - B @ K)

    return K, X, eigVals


def lqr_steering_control(state, cx, cy, cyaw, ck, pe, pth_e, d_t):
    ind, e, match, match_d = calc_nearest_index(state, cx, cy, cyaw) # 这里的e是距离误差
    dt = d_t
    # print("dt: ", dt)
    # print("index is", ind, e)
    # print("ck: ", ck)
    k = ck[match]
    v = state.v
    th_e = pi_2_pi(state.yaw - cyaw[match])  # 航向误差 坐标范围是0-pi,0-(-pi)

    A = np.zeros((4, 4))  # 创建一个4*4的矩阵
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt
    # print(A)

    B = np.zeros((4, 1))
    B[3, 0] = v / L

    K, _, _ = dlqr(A, B, Q, R)

    x = np.zeros((4, 1))

    x[0, 0] = e
    x[1, 0] = (e - pe) / dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt

    ff = math.atan2(L * k, 1)  # 前馈控制量
    fb = pi_2_pi((-K @ x)[0, 0])  # 反馈控制量

    # delta = ff + fb  # 控制量，前轮转角
    delta = fb
    # print("ff 前馈 is:", ff)
    # print("fb 反馈 is:", fb)

    return delta, ind, e, th_e


def calc_nearest_index(state, cx, cy, cyaw):
    # 计算匹配点索引及距离误差
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)
    # print("length: ", len(d))
    match = min(ind + 30, len(d)-1)
    match_d = d[match]

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    # return ind, mind
    return ind, mind, match, match_d


def closed_loop_prediction(cx, cy, cyaw, ck, speed_profile, goal):
    T = 500.0  # max simulation time
    goal_dis = 0.3
    stop_speed = 0.05

    state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]

    e, e_th = 0.0, 0.0

    while T >= time:
        dl, target_ind, e, e_th = lqr_steering_control(
            state, cx, cy, cyaw, ck, e, e_th)  # dl就是前轮转角

        ai = PIDControl(speed_profile[target_ind], state.v)
        state = update(state, ai, dl)

        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.hypot(dx, dy) <= goal_dis:
            print("Goal")
            break

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if target_ind % 1 == 0 and show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.0001)

    return t, x, y, yaw, v


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)  # 目标速度变为与cx等长度的列表

    direction = 1.0

    # Set stop point
    for i in range(len(cx) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])  # 计算间隔两个点之间的航向差的绝对值
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0  # 两点之间的航向误差在45°-90°之间为True,否则为false

        if switch:
            direction *= -1  # 用于判断一下是正方向行进还是反方向

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    speed_profile[-1] = 0.0  # 最后一个点目标速度为0

    return speed_profile


def main():
    print("LQR steering control tracking start!!")
    ax = [0.0, 6.0, 12.5, 10.0, 7.5, 3.0, -1.0]
    ay = [0.0, -3.0, -5.0, 6.5, 3.0, 5.0, -2.0]
    goal = [ax[-1], ay[-1]]  # 目标点

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)  # 计算样条插值点 s 是插值点之间的间隔

    # print("cx is:", len(cx), s)

    target_speed = 10.0 / 3.6  # simulation parameter km/h -> m/s

    sp = calc_speed_profile(cx, cy, cyaw, target_speed)  # sp是目标速度列表

    t, x, y, yaw, v = closed_loop_prediction(cx, cy, cyaw, ck, sp, goal)

    if show_animation:  # pragma: no cover
        plt.close()
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="input")
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots(1)
        plt.plot(s, [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="yaw")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("yaw angle[deg]")

        plt.subplots(1)
        plt.plot(s, ck, "-r", label="curvature")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("curvature [1/m]")

        plt.show()


if __name__ == '__main__':
    main()
