from typing import List

from dsSplines import DsSplineIterator
from ConstrainedState import ConstrainedState
from TrajectoryState import *
import numpy as np


# pretty much directly taken from WPILIB, Java code here:
# https://github.com/wpilibsuite/allwpilib/blob/4630191fa4acc6e8be5a176167c952dea6e163bb/wpimath/src/main/java/edu/wpi/first/wpilibj/trajectory/TrajectoryParameterizer.java
def generateTrajectory(splineIterator: DsSplineIterator, maxVel: float, maxAcc: float, startVel: float = None, endVel: float = None):
    if startVel is None:
        startVel = 0
    if endVel is None:
        endVel = maxVel

    times = []  # timestamp of each point
    ss = []  # total distance covered at each point
    for t, s in splineIterator:
        times.append(t)
        ss.append(s)
    states = []
    predecessor = ConstrainedState(splineIterator(0), 0, splineIterator.curvature(0), startVel, -maxAcc, maxAcc)

    # go forward on the path, constraining each point by acceleration and velocity
    for i, time in enumerate(times):
        state = ConstrainedState(splineIterator(time), ss[i], splineIterator.curvature(time), maxVel, -maxAcc, maxAcc)

        ds = state.accumPos - predecessor.accumPos

        # iterate on the acceleration and velocity constraints
        while True:
            maxVelFromPredecessor = np.sqrt(predecessor.velocity * predecessor.velocity + 2 * predecessor.maxACC * ds)
            state.velocity = min(maxVel, maxVelFromPredecessor)
            state.minACC = -maxAcc
            state.maxACC = maxAcc

            # the only constraint I care about here is net acceleration
            # the full acceleration budget is centripetal acceleration + forward acceleration
            centAccel = state.velocity * state.velocity * state.curvature

            # if the centripetal acceleration exceeds the maximum acceleration, we need to slow down here
            if np.abs(centAccel) > maxAcc:
                # compute the max velocity achievable with the current curvature
                # taken from WPIlib: https://github.com/wpilibsuite/allwpilib/blob/f57c188f2e3d0e9cf0de80e2a1cbb76f10a3e8fa/wpimath/src/main/java/edu/wpi/first/wpilibj/trajectory/constraint/CentripetalAccelerationConstraint.java
                state.velocity = np.sqrt(maxAcc / np.abs(state.curvature))
                centAccel = maxAcc

            # now take that much acceleration away from the maximum acceleration we can give to speeding up
            state.maxACC = maxAcc - centAccel
            state.minACC = -1 * state.maxACC

            if i == 0:
                break

            actualAccel = (state.velocity * state.velocity - predecessor.velocity * predecessor.velocity) / (2 * ds)

            # print(i, time, actualAccel, state)
            # if we need more acceleration than we can get, modify the predecessor's max acceleration?
            if actualAccel > state.maxACC:
                predecessor.maxACC = state.maxACC
            else:
                # this records the actual acceleration we're going to use into the "maxAcc" field
                if actualAccel > predecessor.minACC:
                    predecessor.maxACC = actualAccel
                # if we need to brake harder than we can, the backwards pass will fix that
                # we no longer violate any constraints, so can move on to the next point
                break

        states.append(state)
        predecessor = state

    successor = ConstrainedState(splineIterator(times[-1]), ss[-1], splineIterator.curvature(times[-1]), endVel, -maxAcc, maxAcc)

    # now go backwards on the trajectory, maintaining the minimum acceleration constraints
    for i in range(len(states) - 1, -1, -1):
        state = states[i]
        # ds will be negative here
        ds = state.accumPos - successor.accumPos

        while True:
            maxVelFromSucessor = np.sqrt(successor.velocity * successor.velocity + successor.minACC * ds * 2.0)

            # the min accel doesn't violate the velocity
            if maxVelFromSucessor > state.velocity:
                break

            state.velocity = maxVelFromSucessor

            # the only constraint I care about here is net acceleration
            # the full acceleration budget is centripetal acceleration + forward acceleration
            centAccel = state.velocity * state.velocity * state.curvature

            # if the centripetal acceleration exceeds the maximum acceleration, we need to slow down here
            if np.abs(centAccel) > maxAcc:
                # compute the max velocity achievable with the current curvature
                # taken from WPIlib: https://github.com/wpilibsuite/allwpilib/blob/f57c188f2e3d0e9cf0de80e2a1cbb76f10a3e8fa/wpimath/src/main/java/edu/wpi/first/wpilibj/trajectory/constraint/CentripetalAccelerationConstraint.java
                state.velocity = np.sqrt(maxVel / np.abs(state.curvature))
                centAccel = maxAcc

            # now take that much acceleration away from the maximum acceleration we can give to speeding up
            state.maxACC = np.sqrt(maxAcc * maxAcc - centAccel * centAccel)
            state.minACC = -1 * state.maxACC

            if i == len(times) - 1:
                break

            actualAccel = (state.velocity * state.velocity - successor.velocity * successor.velocity) / (2 * ds)

            if actualAccel < state.minACC:
                successor.minACC = state.minACC
            else:
                successor.minACC = actualAccel
                break

        successor = state


    # we now have keypoints with constrained velocities and accelerations
    # now we have to reparameterize into physical time units (seconds)
    finalStates: List[TrajectoryState] = []
    t = 0
    s = 0
    vel = 0

    for i, state in enumerate(states):
        ds = state.accumPos - s

        accel = (state.velocity * state.velocity - vel * vel) / (2 * ds)

        dt = 0
        if i > 0:
            finalStates[i - 1].acc = accel
            if np.abs(accel) > 1e-6:
                dt = (state.velocity - vel) / accel
            elif vel > 1e-6:
                dt = ds / vel
            else:
                raise ValueError("ended up not moving at state {}".format(repr(state)))

        vel = state.velocity
        s = state.accumPos

        t += dt

        finalStates.append(TrajectoryState(t, vel, accel, state.pos, state.curvature, times[i]))

    return finalStates


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    import scipy.interpolate as spi


    try:
        get_ipython().magic('matplotlib')
    except AttributeError:
        pass

    t = np.array([0, 1, 2, 3])
    poss = np.array([[0, 0],
                     [1, 1],
                     [2, 1],
                     [1, 2]])

    interp = spi.CubicSpline(t, poss)

    iterator = DsSplineIterator(interp, 0.01)

    smallt = np.linspace(0, 3, 1000)
    smallo = interp(smallt)

    iterT = []
    for t, s in iterator:
        print(t, s)
        iterT.append(t)
    iterO = interp(iterT)

    # plt.figure()
    # plt.plot(smallo[:, 0], smallo[:, 1])
    # plt.plot(poss[:, 0], poss[:, 1], 'o')
    # plt.plot(iterO[:, 0], iterO[:, 1], '*')

    traj = generateTrajectory(iterator, 1, 1, 0, 1)

    ts = [tr.time for tr in traj]
    vels = [tr.vel for tr in traj]
    accs = [tr.acc for tr in traj]

    # plt.subplot(2, 1, 1)
    # plt.plot(ts, vels)
    # plt.subplot(2, 1, 2)
    # plt.plot(ts, accs)

    interpolator = TrajectoryInterpolator(traj, iterator)

    smallT = np.linspace(0, traj[-1].time, 1000)
    fineStates = [interpolator.sample(t) for t in smallT]

    finets = [tr.time for tr in fineStates]
    finevels = [tr.vel for tr in fineStates]
    fineaccs = [tr.acc for tr in fineStates]
    finenetaccs = [tr.getNetAcceleration() for tr in fineStates]

    plt.figure()
    plt.plot(finets, finevels)
    plt.title('Velocity along path')
    plt.xlabel('time (sec)')
    plt.ylabel('velocity (m/s)')
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(finets, fineaccs)
    plt.title('Acceleration along path')
    plt.xlabel('time (sec)')
    plt.ylabel('acceleration (m/s^2)')
    plt.subplot(2, 1, 2)
    plt.plot(finets, finenetaccs)
    plt.title('Net acceleration magnitude')
    plt.xlabel('time (sec)')
    plt.ylabel('acceleration (m/s^2)')

    finex = [tr.pos[0] for tr in fineStates]
    finey = [tr.pos[1] for tr in fineStates]

    plt.figure()
    plt.plot(finex, finey)
    plt.title('path with time to points labelled')
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')

    for i in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]:
        print('{}: {}'.format(i, finets[i]))
        plt.plot(finex[i], finey[i], '.')
        plt.text(finex[i], finey[i], 't={:.1f}'.format(finets[i]))
