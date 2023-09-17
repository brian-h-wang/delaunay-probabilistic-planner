"""
Brian Wang
bhw45@cornell.edu

Module for computing control commands for the robot.
"""
import numpy as np

def feedback_lin(vx, vy, theta, epsilon=0.2):
    """
    Perform feedback linearization to convert a Vx, Vy command in
    the inertial frame to a command in terms of robot forward and
    angular velocity.
    (This code adapted from MAE 5180 Autonomous Mobile Robots)
    vx: float
    vy: float
        Commanded velocity in x- and y- directions
    theta: float
        angle of the robot
    epsilon: float
        Lookahead distance. A smaller epsilon value will cause
        the robot to turn more quickly when following a path.
    Returns
    -------
    cmdV
        Forward velocity command
    cmdW
        Angular velocity command
    """
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, s], [-s, c]])
    V = np.array([vx, vy]).reshape((2,1))
    cmd = (np.diag([1, 1/epsilon]).dot(R).dot(V)).flatten()
    return cmd[0], cmd[1]