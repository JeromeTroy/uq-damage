# Building step functions for integrating over domain

from fenics import *
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal

# import Energy as ener

# function whose support is only a specied angular
# ...(UserExpression) indicates this extends the
# fenics class UserExpression.
# all defined methods are polymorphic to those of UserExpression
# which I presume is a subclass of Expression
class RingStepFunction(UserExpression):
    """
    This function is zero everywhere except on a
    specified angular range
    """

    def __init__(self, start_angle, angle_range):
        """
        Builds the expression

        Input:
            start_angle (radians) - which angle to start at
            angle_range (radians) - the angular width

        Eg. start_angle = pi, angle_range = pi/2
        This will be zero everywhere except for
        angles between pi and 3pi/2
        """

        # assign values
        self._start_angle = start_angle
        self._stop_angle = start_angle + angle_range

        # build everything else the same as usual expression
        super().__init__()

    def set_start_angle(self, angle):
        self._start_angle = angle

    def set_angle_range(self, val):
        self._stop_angle = self._start_angle + val

    def eval(self, value, x):
        """
        Evaluation of the step function

        Input:
            value - output variable for result
                (replaces return value)
            x - the input for the function
        Output:
            put in value
            will be 1, if polar angle of x is in specified range
            0 otherwise
        """

        # determine the polar angle of x
        # note arctan2 casts the angle to the correct quadrant
        angle = np.arctan2(x[1], x[0])

        # piecewise expression
        if (self._start_angle <= angle <= self._stop_angle) or (self._start_angle-2*np.pi <= angle <= self._stop_angle-2*np.pi):
            # in allowed domain
            value[0] = 1.0
        else:
            # outside allowed domain
            value[0] = 0.0


    def value_shape(self):
        """
        Tells FEniCS the shape of the output

        Should not need to call this manually
        """

        # scalar return - no shape
        return ()

# performing integrations
def angle_averaged_damage(damage, θ, width=5*np.pi/180):
    """
    Average damage over annular segments

    Input:
        damage : Fenics Function object
            damage function
        θ : numpy array shape (n,)
            starting angles for annular segments (in radians)
        width : float > 0, optional
            width (in radians) of the annular segments.
            The default is 5 degrees
    Output:
        avg_damage : numpy array shape (n,) 
            damage values averaged over the segments
    """

    # compute inner and outer radii
    V = damage.function_space()
    coords = V.mesh().coordinates().T
    r = np.linalg.norm(coords, axis=1)
    inner_rad = np.min(r)
    outer_rad = np.max(r)

    num_segments = len(θ)

    # base area
    area = 0.5 * width * (outer_rad**2 - inner_rad**2)

    # allocate space for average damages
    avg_damage = np.zeros_like(θ)

    # initialize the step function
    step_function = RingStepFunction(θ[0], width)

    # loop for angle averaging
    for i in range(num_segments):
        start_angle = θ[i]

        # update step function
        step_function.set_start_angle(start_angle)
        step_function.set_angle_range(width)

        step = project(step_function, V)

        # perform integration
        avg_damage[i] = assemble(damage * step * dx) / area

    return avg_damage
