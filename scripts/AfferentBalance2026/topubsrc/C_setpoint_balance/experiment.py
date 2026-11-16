"""C: one controller with polynomial commands for multiple static targets.

The staged method uses table currents, bounded least squares, then joint search.
Only the final polynomial winner is tested on the held-out bank.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import lsq_linear
from genome import GenomeConfig, JointGenome, write_json
from model import balance_banks
from evolution import cli, finish_run, seed_coordinates, train_joint, train_phase

HERE = Path(__file__).resolve().parent


def make_banks(config, kind='polynomial'):
    """Cross independent initial-state banks with equally weighted target angles.

    Training always uses the five or more anchor targets. Polynomial validation
    and tests include configured intermediate targets; table stages use anchors
    only. Initial conditions are absolute angles, not errors around each target.
    """
    cfg = config if isinstance(config, GenomeConfig) else GenomeConfig(config)
    if kind not in ('polynomial','table'): raise ValueError('Unknown command representation')
    settings = cfg.data['setpoint']
    initial = balance_banks(cfg.data['simulation'])
    result = {}
    for name in ('training','validation','test'):
        targets = np.asarray(settings['targets_deg' if name=='training' or kind=='table'
                                      else f'{name}_targets_deg'], dtype=float)
        if (targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all()
                or len(np.unique(targets)) != len(targets) or np.any(abs(targets)>45)):
            raise ValueError('Target banks must contain unique finite angles within -45..45 degrees')
        if not set(settings['targets_deg']).issubset(set(targets)):
            raise ValueError('Each bank must include all training target angles')
        blocks=[]
        for target in targets:
            block=initial[name].copy(); block[:,0]=np.deg2rad(target); blocks.append(block)
        result[name]=np.concatenate(blocks)
    sets=[set(map(tuple,result[name])) for name in result]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Training, validation and test episodes must be disjoint')
    return result


def fit_table_polynomial(targets_deg, currents, degree, bound):
    """Fit two bounded ascending-power polynomials by least squares.

    This approximates currents, not dynamics. The returned residuals must be
    followed by fresh rollouts and joint refinement. No test outcomes enter fitting.
    """
    targets=np.asarray(targets_deg,dtype=float); currents=np.asarray(currents,dtype=float)
    if (degree not in (2,3) or targets.ndim!=1 or len(targets)<degree+1
            or len(np.unique(targets))!=len(targets) or currents.shape!=(2,len(targets))
            or not np.isfinite(targets).all() or not np.isfinite(currents).all()
            or not np.isfinite(bound) or bound<=0):
        raise ValueError('Invalid least-squares table, polynomial degree or coefficient bound')
    design=np.vander(targets/45.,degree+1,increasing=True)
    solutions=[lsq_linear(design,row,bounds=(-bound,bound),tol=1e-12) for row in currents]
    if not all(sol.success for sol in solutions):
        raise RuntimeError('Bounded least-squares solver failed')
    coefficients=np.array([sol.x for sol in solutions])
    residual=coefficients@design.T-currents
    return coefficients, dict(rmse_nA=float(np.sqrt(np.mean(residual**2))),
                              max_absolute_error_nA=float(abs(residual).max()),
                              solver='scipy.optimize.lsq_linear', degree=degree)



def train_experiment(config, settings, output, bank_factory, saved=None):
    """Use a shared total generation budget and restart annealing at each stage."""
    if settings.get('method', 'joint') == 'joint':
        return train_joint(config, settings, output, bank_factory, saved)
    if settings['generations'] < 2:
        raise ValueError('Staged training needs at least two generations')
    cfg = GenomeConfig(config)
    degree = cfg.data['setpoint']['degree']
    table, polynomial = JointGenome(cfg, 'table', degree), JointGenome(cfg, 'polynomial', degree)
    first = settings['generations']//2
    # Each phase is one ordinary GA run; the staged minimum applies to the total budget.
    phase_settings = {key:value for key,value in settings.items() if key != 'method'}
    table_winner = train_phase(table, bank_factory(config, 'table'),
                              {**phase_settings, 'generations':first}, output/'table',
                              center=seed_coordinates(table, saved))
    physical, values = table.decode(table_winner['unit'])
    coefficients, fit = fit_table_polynomial(table.targets, values, degree, polynomial.bound)
    if cfg.mirror_symmetry:
        parity = (-1.)**np.arange(degree+1)
        first_row = .5*(coefficients[0]+parity*coefficients[1])
        coefficients = np.array([first_row, parity*first_row])
    residual = coefficients@np.vander(table.targets/45.,degree+1,increasing=True).T-values
    write_json(output/'least_squares.json', dict(
        solver=fit['solver'], targets_deg=table.targets.tolist(), table_currents=values.tolist(),
        coefficients=coefficients.tolist(), rmse_model_units=float(np.sqrt(np.mean(residual**2))),
        mirror_projected=cfg.mirror_symmetry))
    banks = bank_factory(config, 'polynomial')
    winner = train_phase(polynomial, banks, {**phase_settings, 'generations':settings['generations']-first},
                         output/'polynomial', center=polynomial.encode(physical, coefficients))
    return finish_run(polynomial, winner, banks, output)


if __name__ == '__main__':
    cli('C_setpoint_balance', HERE, make_banks, train_experiment)
