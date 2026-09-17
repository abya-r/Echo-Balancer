'''
*****************************************************************************************
*
*        =================================================
*             Echo Balancer (EB) Theme (eYRC 2026-27)
*        =================================================
*
*  This script is to implement Task 1A of Echo Balancer (EB) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          3213
# Author List:      Aditri Khanna, Kritika Raj, Abya Rao, Anamika Kumari
# Filename:         Task1A.py
# Functions:        find_equilibrium_points, find_A_B_matrices,
#                   find_eigen_values, compute_lqr_gain
# Global variables: theta, omega, u, theta_dot, omega_dot, STATES

import sympy as sp
import numpy as np
import control


############################################################################
#                        THE SYSTEM  -  fill this in                       #
############################################################################
# The pendulum-with-torque system from the "Modeling of non-linear Dynamical
# Systems" Learnings page, with one added damping term:
#
#     theta_dot = omega
#     omega_dot = -10*sin(theta) - omega + u
#
#   theta = angle from vertical   omega = angular rate   u = applied torque
#
# Use sympy symbols, not plain Python numbers, so these expressions stay
# differentiable.

# Define the symbolic variables
theta, omega, u = sp.symbols('theta omega u')

# Define the differential equations
theta_dot = omega
omega_dot = -10*sp.sin(theta) - omega + u

# The order of the states. Keep it as angle first, then its rate.
STATES = [theta, omega]

############################################################################


def find_equilibrium_points():
    '''
    Find every point where the pendulum is not accelerating: switch off the
    input, set both derivatives to zero, and solve for theta and omega
    together.

    Returns:
    ---
    `equi_points`: [ list of tuples ] one (theta, omega) pair per equilibrium

    Hint: sp.solve() takes a list of expressions (assumed equal to zero) and
    a list of unknowns. sin(theta) = 0 has infinitely many roots in theory;
    only two are physically distinct.
    '''
    theta_eq = theta_dot.subs(u, 0)
    omega_eq = omega_dot.subs(u, 0)

    equi_points = sp.solve(
        [theta_eq, omega_eq],
        [theta, omega],
        dict=True
    )
    ###### WRITE YOUR CODE HERE ################
    # HINT: sp.solve() takes a list of expressions (each taken to equal
    # zero) and a list of unknowns, e.g.
    #     sp.solve([x + y - 2, x - y], [x, y])       # -> [(1, 1)]

    ############################################

    return equi_points


def find_A_B_matrices(eq_points):
    '''
    Linearise the system at every equilibrium point: f.jacobian(STATES)
    gives A and f.jacobian([u]) gives B, where f = [theta_dot, omega_dot].
    Using jacobian() instead of writing each partial derivative by hand is
    what lets this same code work unchanged on Task 1B's four-state model.
    At each point, substitute the state values and u = 0.

    Input Arguments:
    ---
    `eq_points`: [ list of tuples ] the points from find_equilibrium_points()

    Returns:
    ---
    `A_matrices`, `B_matrices`: [ lists of sympy Matrix ]

    Note: B does not depend on theta here, so it is the same at both points;
    that is expected, not a mistake.
    '''
    f = sp.Matrix([theta_dot, omega_dot])

    A_matrices, B_matrices = [], []

    ###### WRITE YOUR CODE HERE ################
    # HINT: a column Matrix has a .jacobian() method, e.g.
    #     sp.Matrix([x * y, x + k]).jacobian([x, y])   # w.r.t. the states
    #     sp.Matrix([x * y, x + k]).jacobian([k])      # w.r.t. the input

    A_symbolic = f.jacobian(STATES)
    B_symbolic = f.jacobian([u])

    for point in eq_points:
        values = {
            STATES[0]: point[0],
            STATES[1]: point[1],
            u: 0
        }

        A = A_symbolic.subs(values)
        B = B_symbolic.subs(values)

        A_matrices.append(A)
        B_matrices.append(B)
    
    ############################################

    return A_matrices, B_matrices


def find_eigen_values(A_matrices):
    '''
    Work out whether the pendulum, left alone, falls back or falls away at
    each equilibrium: call .eigenvals() on each A matrix, then mark it
    'Stable' if every eigenvalue has a strictly negative real part, else
    'Unstable'.

    Input Arguments:
    ---
    `A_matrices`: [ list of sympy Matrix ] from find_A_B_matrices()

    Returns:
    ---
    `eigen_values`: [ list ] one .eigenvals() dict per point
    `stability`:    [ list of str ] 'Stable' or 'Unstable' per point

    Hint: sp.re(value) gives the real part. A complex pair with a negative
    real part is still stable; it just oscillates on the way back.
    '''
    eigen_values = []
    stability = []

    ###### WRITE YOUR CODE HERE ################
    # HINT: matrix.eigenvals() returns a dict of {eigenvalue: multiplicity},
    # e.g. sp.Matrix([[0, 1], [-4, -2]]).eigenvals()

    for A in A_matrices:
    
        ev_dict = A.eigenvals()
        eigen_values.append(ev_dict)
    
        is_stable = True
        for val in ev_dict.keys():
            if sp.re(val).evalf() >= 0:
                is_stable = False
                break
    
        if is_stable:
            stability.append('Stable')
        else:
            stability.append('Unstable')

    ############################################

    return eigen_values, stability


def compute_lqr_gain(A_matrices, B_matrices, stability):
    '''
    Design the controller for the one equilibrium that needs it: hanging
    down settles back on its own, balanced upright does not.

    Input Arguments:
    ---
    `A_matrices`, `B_matrices`: [ lists of sympy Matrix ]
    `stability`: [ list of str ] from find_eigen_values()

    Returns:
    ---
    `K`: [ numpy array ] the LQR gain, one entry per state

    Steps: pick out the A and B at the unstable equilibrium, convert both to
    float numpy arrays (control.lqr() will not take sympy Matrix objects),
    then call control.lqr(A, B, Q, R) and keep the first of its three
    return values.

    Note: do not change Q or R; they are fixed for Task 1A so every team's
    K is comparable.
    '''
    # Define the Q and R matrices
    Q = np.eye(2)        # State weighting matrix
    R = np.array([1])    # Control weighting matrix

    ###### WRITE YOUR CODE HERE ################
    # HINT: control.lqr() takes plain numpy arrays and returns three
    # things, the gain first, e.g.
    #     K, _, _ = control.lqr(A, B, Q, R)

    ############################################

    return K


def main_function():    # Don't change anything in this function
    eq_points = find_equilibrium_points()

    if not eq_points:
        print("No equilibrium points found.")
        return None, None, None, None, None, None

    A_matrices, B_matrices = find_A_B_matrices(eq_points)
    eigen_values, stability = find_eigen_values(A_matrices)
    K = compute_lqr_gain(A_matrices, B_matrices, stability)

    return eq_points, A_matrices, B_matrices, eigen_values, stability, K


def task1a_output(eq_points, A_matrices, B_matrices, eigen_values, stability, K):
    '''
    This function prints the results you have obtained.
    '''
    print("Equilibrium Points:")
    for i, point in enumerate(eq_points):
        print(f"  Point {i + 1}: theta = {point[0]}, omega = {point[1]}")

    print("\nA Matrices at Equilibrium Points:")
    for i, matrix in enumerate(A_matrices):
        print(f"  At Point {i + 1}:")
        print(sp.pretty(matrix, use_unicode=False))

    print("\nB Matrices at Equilibrium Points:")
    for i, matrix in enumerate(B_matrices):
        print(f"  At Point {i + 1}: {sp.Matrix(matrix).T.tolist()[0]}")

    print("\nEigenvalues at Equilibrium Points:")
    for i, eigvals in enumerate(eigen_values):
        eigvals_str = ', '.join([f"{val}: {count}" for val, count in eigvals.items()])
        print(f"  At Point {i + 1}: {eigvals_str}")

    print("\nStability of Equilibrium Points:")
    for i, status in enumerate(stability):
        print(f"  At Point {i + 1}: {status}")

    print("\nLQR Gain Matrix K at the unstable Equilibrium Point:")
    print(K)


if __name__ == "__main__":
    results = main_function()
    eq_points, A_matrices, B_matrices, eigen_values, stability, K = results
    task1a_output(eq_points, A_matrices, B_matrices, eigen_values, stability, K)
