"""Explicit, reproducible banks; each row is [target, theta0, omega0, kick_t, kick_dw]."""
import numpy as np

ANGLES = np.sort(np.r_[np.arange(-45., 46., 10.), 0.])
SEEDS = {"training": 20261008, "validation": 20261009, "test": 20261010,
         "controller_selection": 20261011}


def balance_cases(seed, count=24):
    if count < 1:
        raise ValueError("A case bank cannot be empty")
    rng = np.random.default_rng(seed)
    cases = np.zeros((count, 5))
    cases[:, 1] = np.deg2rad(rng.uniform(-8., 8., count))
    cases[:, 2] = rng.uniform(-.15, .15, count)
    cases[:, 3] = -1.
    cases[count//2:, 3] = rng.uniform(4., 6., count-count//2)
    cases[count//2:, 4] = rng.choice([-1., 1.], count-count//2) * rng.uniform(.15, .3, count-count//2)
    return cases


def setpoint_cases(seed, per_angle=3, angles=ANGLES):
    rng = np.random.default_rng(seed)
    return np.array([[np.deg2rad(a), 0. if i == 0 else np.deg2rad(a+rng.uniform(-8., 8.)),
                      rng.uniform(-.15, .15), -1. if i < 2 else rng.uniform(4., 6.),
                      0. if i < 2 else rng.choice([-1., 1.])*.2]
                     for a in angles for i in range(per_angle)])


def banks(experiment):
    if experiment == "angle_setpoint":
        return {name: setpoint_cases(SEEDS[name], per_angle=count,
                    angles=ANGLES if name != "test" else np.arange(-45., 46., 5.))
                for name, count in (("training", 3), ("validation", 4), ("test", 4))}
    if experiment not in ("equilibrium", "small_nm_weights"):
        raise ValueError(f"Unknown experiment: {experiment}")
    return {name: balance_cases(SEEDS[name], count)
            for name, count in (("training", 12), ("validation", 24), ("test", 48))}

