from copy import copy
from dataclasses import dataclass
import numpy as np
from dsSplines import DsSplineIterator
from typing import List


@dataclass
class TrajectoryState:
    time: float
    vel: float
    acc: float
    pos: np.ndarray
    curvature: float
    splineT: float

    def getNetAcceleration(self):
        # we know the "acc" acceleration is entirely in the direction of forward motion
        # and curvature is, by definition, perpendicular to that
        # so the total acceleration is just the pythagorean theorem of the two
        centAccel = self.vel * self.vel * self.curvature
        return np.sqrt(self.acc * self.acc + centAccel * centAccel)


class TrajectoryInterpolator:

    def __init__(self, states: List[TrajectoryState], spline: DsSplineIterator):
        self.states = states
        self.spline = spline
        self.lastIndex = 0

    # we could binary search, but because we expect this to be used linearly it will be faster to just search around the last sampled point
    def sample(self, time: float):
        if time > self.states[self.lastIndex].time:
            while self.lastIndex < len(self.states) - 1 and time > self.states[self.lastIndex + 1].time:
                self.lastIndex += 1
        else:
            while self.lastIndex >= 0 and time < self.states[self.lastIndex].time:
                self.lastIndex -= 1

        if self.lastIndex >= len(self.states) - 1:
            # we're at the end of the trajectory, assume the realT -> splineT relation holds from the last two points
            splineTperRealT = (self.states[-1].splineT - self.states[-2].splineT) / (
                        self.states[-1].time - self.states[-2].time)
            oTraj = copy(self.states[-1])
            splineT = self.states[-1].splineT + (time - self.states[-1].time) * splineTperRealT
            oTraj.pos = self.spline(splineT)
            oTraj.curvature = self.spline.curvature(splineT)
            oTraj.splineT = splineT
            oTraj.time = time
            return oTraj
        else:
            splineTperRealT = (self.states[self.lastIndex + 1].splineT - self.states[self.lastIndex].splineT) / (
                    self.states[self.lastIndex + 1].time - self.states[self.lastIndex].time)
            oTraj = copy(self.states[self.lastIndex])
            splineT = self.states[self.lastIndex].splineT + (time - self.states[self.lastIndex].time) * splineTperRealT
            oTraj.pos = self.spline(splineT)
            oTraj.curvature = self.spline.curvature(splineT)
            oTraj.splineT = splineT
            dv = self.states[self.lastIndex + 1].vel - self.states[self.lastIndex].vel
            dt = self.states[self.lastIndex + 1].time - self.states[self.lastIndex].time
            # lerp the velocity
            oTraj.vel = self.states[self.lastIndex].vel + dv * (time - self.states[self.lastIndex].time) / dt
            oTraj.time = time
            return oTraj
