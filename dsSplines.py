import numpy as np
import scipy.integrate as integ
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import matplotlib
from IPython import get_ipython


class DsSplineIterator:
    def __init__(self, spline: spi.CubicSpline, step=0.1):
        self.spline = spline
        self.dspline = self.spline.derivative()
        self.ddspline = self.dspline.derivative()
        # desired ds step
        self.dstep = step
        # tolerance in error fraction
        self.tol = 0.01
        # last returned t
        self.curT = 0
        # total accumulated distance
        self.curDist = 0
        # last target distance, so we don't accumulate error
        self.curTargDist = 0
        # last step required to get ds step
        self.tstep = step
        # last delta needed to fix the tstep
        self.tstepstep = step / 2
        # flag to return 0 the first time it's iterated over
        self.zero = True

    def __iter__(self):
        self.curT = 0
        self.curDist = 0
        self.curTargDist = 0
        self.zero = True
        return self

    def __call__(self, *args, **kwargs):
        return self.spline(*args, **kwargs)

    def _arclenIntegrand(self, t):
        return np.sqrt(np.sum(np.square(self.dspline(t))))

    def arclength(self, t1, t2):
        return integ.quad_vec(self._arclenIntegrand, t1, t2)[0]

    def curvature(self, t):
        # taken from https://en.wikipedia.org/wiki/Curvature
        try:
            dx, dy, dz = self.dspline(t)
            ddx, ddy, ddz = self.ddspline(t)
        except ValueError:  # throws a ValueError if attempting to unpack a 2d curve into 3d
            dx, dy = self.dspline(t)
            ddx, ddy = self.ddspline(t)
            dz = 0
            ddz = 0
        return np.sqrt((ddz * dy - ddy * dz) ** 2 + (ddx * dz - ddz * dx) ** 2 + (ddy * dx - ddx * dy) ** 2) \
            / np.power(dx ** 2 + dy ** 2 + dz ** 2, 3 / 2)

    def unitVel(self, t):
        vel = self.dspline(t)
        return vel / np.sqrt(np.sum(np.square(vel)))

    def unitAcc(self, t):
        acc = self.ddspline(t)
        return acc / np.sqrt(np.sum(np.square(acc)))

    def __next__(self):
        if self.zero:
            self.zero = False
            return 0, 0
        # if we've overrun the end of the spline stop iterating
        if self.curT > self.spline.x[-1]:
            raise StopIteration
        self.curTargDist += self.dstep
        # start by guessing the same time step as worked the last time
        tstep = self.tstep
        nextT = self.curT + tstep
        integral = self.arclength(self.curT, nextT)
        dist = self.curDist + integral
        lastLow = None
        lastHigh = None
        # do a binary search over tstep until we get to a dist that's within tolerance
        while np.abs(self.curTargDist - dist) > self.dstep * self.tol:
            if dist > self.curTargDist:
                lastHigh = tstep
                if lastLow is None:
                    # just go down an arbitrary (and increasing) amount
                    tstep -= self.tstepstep
                    self.tstepstep *= 1.5
                else:
                    # do the binary search thing, guessing halfway between high and low
                    tstep = (lastHigh + lastLow) / 2
            else:
                lastLow = tstep
                if lastHigh is None:
                    tstep += self.tstepstep
                    self.tstepstep *= 1.5
                else:
                    tstep = (lastHigh + lastLow) / 2
            nextT = self.curT + tstep
            integral = self.arclength(self.curT, nextT)
            dist = self.curDist + integral
        self.curT = nextT
        # don't update tstepstep if it would become 0
        if self.tstep != tstep:
            # update the last delta magnitude required
            self.tstepstep = np.abs(self.tstep - tstep)
        # update the last timestep required
        self.tstep = tstep
        self.curDist = dist
        return self.curT, self.curDist


if __name__ == '__main__':
    # set up interactive matplotlib that actually works properly
    get_ipython().magic('matplotlib')

    t = np.array([0, 1, 2, 3])
    poss = np.array([[0, 0],
                     [1, 1],
                     [2, 1],
                     [1, 2]])

    interp = spi.CubicSpline(t, poss)

    smallt = np.linspace(0, 3, 1000)
    smallo = interp(smallt)

    iterator = DsSplineIterator(interp, 0.01)

    iterT = []
    for t, s in iterator:
        print(t, s)
        iterT.append(t)
    iterO = interp(iterT)

    plt.figure()
    plt.plot(smallo[:, 0], smallo[:, 1])
    plt.plot(poss[:, 0], poss[:, 1], 'o')
    plt.plot(iterO[:, 0], iterO[:, 1], '*')
