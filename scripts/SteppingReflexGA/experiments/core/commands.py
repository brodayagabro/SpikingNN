"""Angle-to-current map; the selected neural/muscle controller stays fixed."""
import numpy as np
from .afferents import afferent_response
from .dynamics import A1, A2

COMMAND_SIZE = 6
COEFFICIENT_LIMIT = 100.


def decode_command(unit):
    unit = np.asarray(unit, dtype=float)
    if unit.shape != (COMMAND_SIZE,) or not np.isfinite(unit).all() or np.any((unit < 0) | (unit > 1)):
        raise ValueError("Expected six finite command genes in [0, 1]")
    return ((2 * unit - 1) * COEFFICIENT_LIMIT).reshape(2, 3)


def input_currents(controller, coefficients, target_rad):
    """Static full-afferent length compensation plus a learned cubic residual.

No measured limb state is used. x=theta*/45 degrees; coefficients multiply
[x,x²,x³]. At theta*=0 both applied currents remain exactly zero.
"""
    if not np.isfinite(target_rad) or abs(target_rad) > np.pi/4 + 1e-12:
        raise ValueError("Command angle must be inside [-45, 45] degrees")
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (2, 3) or not np.isfinite(coefficients).all():
        raise ValueError("Expected a finite 2 x 3 coefficient matrix")
    length0 = np.hypot(A1, A2)
    lengths = np.sqrt(length0**2 + 2*A1*A2*np.sin(target_rad)*np.array([-1., 1.]))
    delta = np.zeros(6)
    for i in range(2):
        args = (controller[42+i], controller[44+i])
        # Only static length terms differ; activation and force are zero here.
        delta[3*i:3*i+3] = (afferent_response(0., length0, 0., 0., *args)
                            - afferent_response(0., lengths[i], 0., 0., *args))
    x = target_rad / (np.pi/4)
    return controller[46:48] + controller[30:42].reshape(2, 6) @ delta + coefficients @ np.array([x, x*x, x*x*x])


def commanded_parameters(controller, coefficients, target_rad):
    result = controller.copy()
    result[46:48] = input_currents(controller, coefficients, target_rad)
    return result


def polynomial_currents(coefficients, target_rad):
    """Pure polynomial command, I_i=sum_k c[i,k]*(theta*/45deg)**k.

    Column 0 is a constant current (including at the upright command), column 1
    is the linear term, column 2 the quadratic term, and column 3 the cubic
    term. The measured limb angle/speed never enters this command. Degree zero
    is useful for calibrating the two currents at one prescribed target.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if (coefficients.ndim != 2 or coefficients.shape[0] != 2 or
            coefficients.shape[1] not in (1, 3, 4) or not np.isfinite(coefficients).all()):
        raise ValueError("Expected finite 2 x (1, 3 or 4) polynomial coefficients")
    if not np.isfinite(target_rad) or abs(target_rad) > np.pi/4 + 1e-12:
        raise ValueError("Command angle must be inside [-45, 45] degrees")
    x = target_rad / (np.pi/4)
    return coefficients @ (x ** np.arange(coefficients.shape[1]))


def polynomial_parameters(controller, coefficients, target_rad):
    """Copy the fixed physical controller and replace only its two input currents."""
    result = controller.copy()
    result[46:48] = polynomial_currents(coefficients, target_rad)
    return result
