import time
from pprint import pprint
from typing import List, Optional

import airsimdroneracinglab as adrl
from airsimdroneracinglab import Vector3r

from BaseRacer import BaseRacer
import numpy as np
import scipy.interpolate as spi

from TrajectoryState import TrajectoryInterpolator, TrajectoryState
from dsSplines import DsSplineIterator
from generateTrajectory import generateTrajectory
import threading


class SplineRacer(BaseRacer):
    def __init__(self, maxVel, maxAcc):
        super().__init__()

        self.maxVel = maxVel
        self.maxAcc = maxAcc

        self.trajWaypoints: Optional[List[TrajectoryState]] = None
        self.trajInterp: Optional[TrajectoryInterpolator] = None

        self.follower_thread = threading.Thread(target=self.loopFollower)
        self.follow = False
        self.startTime = 0

        # eye roll emoji
        self.airsim_client.race_tier = None

    def generateTrajectory(self):
        self.get_ground_truth_gate_poses()

        # make the spline of the entire course
        # evenly space the t between gates
        # start with the drone's current position
        t = np.arange(len(self.gate_poses_ground_truth) + 1)
        xyz = np.empty((len(t), 3))
        pos = self.position
        xyz[0, 0] = pos.x_val
        xyz[0, 1] = pos.y_val
        xyz[0, 2] = pos.z_val - 2  # get it off the starting block
        for i, pose in enumerate(self.gate_poses_ground_truth):
            xyz[i + 1, 0] = pose.position.x_val
            xyz[i + 1, 1] = pose.position.y_val
            xyz[i + 1, 2] = pose.position.z_val
        spline = DsSplineIterator(spi.CubicSpline(t, xyz), 0.1)

        # max velocity, max acceleration, starting velocity, (max) ending velocity
        self.trajWaypoints = generateTrajectory(spline, self.maxVel, self.maxAcc, 0, self.maxVel)

        self.trajInterp = TrajectoryInterpolator(self.trajWaypoints, spline)

    def plotCourse(self, n=5000):
        points = []
        traj = []
        ts = np.linspace(0, self.trajWaypoints[-1].time, n)
        for t in ts:
            pt = self.trajInterp.sample(t).pos
            traj.append(self.trajInterp.sample(t))
            points.append(Vector3r(pt[0], pt[1], pt[2]))
        self.airsim_client.simPlotLineStrip(points, color_rgba=[1.0, 1.0, 0.0, 0.5], is_persistent=True)

    def goToTrajPoint(self, t):
        trajPt = self.trajInterp.sample(t)
        lookAheadPt = self.trajInterp.sample(t + 0.5)
        # print('going to point {}'.format(trajPt))

        ptVec = Vector3r(trajPt.pos[0], trajPt.pos[1], trajPt.pos[2])
        vel = self.trajInterp.spline.unitVel(t) * trajPt.vel
        velVec = Vector3r(vel[0], vel[1], vel[2])

        lptVec = Vector3r(lookAheadPt.pos[0], lookAheadPt.pos[1], lookAheadPt.pos[2])
        vel = self.trajInterp.spline.unitVel(t + 0.5) * lookAheadPt.vel
        lvelVec = Vector3r(vel[0], vel[1], vel[2])


        #self.airsim_client.simPlotPoints([ptVec], size=30, duration=0.1)
        # self.airsim_client.cancelLastTask(self.drone_name)

        # return self.airsim_client.moveOnSplineVelConstraintsAsync(
        #     [ptVec, lptVec],
        #     [velVec, lvelVec],
        #     vel_max=10.0,
        #     acc_max=4.0,
        #     add_position_constraint=False,
        #     add_velocity_constraint=False,
        #     add_acceleration_constraint=False,
        #     replan_from_lookahead=False,
        #     viz_traj=True,
        #     vehicle_name=self.drone_name
        # )

        # return self.airsim_client.moveOnSplineAsync(
        #     [ptVec, lptVec],
        #     vel_max=10.0,
        #     acc_max=4.0,
        #     add_position_constraint=False,
        #     add_velocity_constraint=False,
        #     add_acceleration_constraint=False,
        #     replan_from_lookahead=False,
        #     viz_traj=True,
        #     vehicle_name=self.drone_name
        # )

        return self.airsim_client.moveToPositionAsync(
            ptVec.x_val, ptVec.y_val, ptVec.z_val, 30.0,
            drivetrain=adrl.DrivetrainType.ForwardOnly,
            yaw_mode=adrl.YawMode(is_rate=False, yaw_or_rate=0))

    def followSplinePath(self):
        pts = []
        vels = []
        ts = np.linspace(0, self.trajWaypoints[-1].time, 150)
        for t in ts:
            trajPt = self.trajInterp.sample(t)
            ptVec = Vector3r(trajPt.pos[0], trajPt.pos[1], trajPt.pos[2])
            vel = self.trajInterp.spline.unitVel(t) * trajPt.vel
            velVec = Vector3r(vel[0], vel[1], vel[2])
            pts.append(ptVec)
            vels.append(velVec)

        return self.airsim_client.moveOnSplineAsync(
            pts,
            vel_max=100.0,
            acc_max=30.0,
            add_position_constraint=False,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            replan_from_lookahead=False,
            viz_traj=True,
            vehicle_name=self.drone_name
        )


    def loopFollower(self):
        t = 0
        while self.follow:
            t = time.time() - self.startTime
            self.goToTrajPoint(t)
            time.sleep(0.3)
            t = t + 1

    def startFollowingPath(self):
        if not self.follow:
            self.follow = True
            self.follower_thread.start()
            self.startTime = time.time()
            print('started following trajectory at time {}s'.format(self.startTime))



if __name__ == "__main__":
    spl = SplineRacer(1, 4)

    spl.load_level('Soccer_Field_Medium')
    spl.initialize_drone()
    spl.odometry_callback()
    spl.start_odometry_callback_thread()
    spl.generateTrajectory()
    # pprint(spl.trajWaypoints)
    spl.plotCourse()
    spl.start_race(1)
    spl.takeoff_with_moveOnSpline()
    # spl.startFollowingPath()
    spl.followSplinePath()
