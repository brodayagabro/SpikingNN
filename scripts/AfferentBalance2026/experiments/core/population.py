"""Population dynamics, shared serial/process evaluation, and controller selection."""
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
import multiprocessing as mp
import numpy as np
from .commands import commanded_parameters, polynomial_parameters
from .config import DT_MS, MODEL, POPULATION_SCHEMA
from .dynamics import rollout
from .fitness import fitness


def evaluate_individual(task):
    if len(task) not in (4, 5):
        raise ValueError("Expected a four-field legacy task or a five-field command task")
    controller, coefficients, cases, duration_s = task[:4]
    command_kind = task[4] if len(task) == 5 else "compensated_cubic"
    rows = []
    command = polynomial_parameters if command_kind == "polynomial" else commanded_parameters
    for case in cases:
        p = controller if coefficients is None else command(controller, coefficients, case[0])
        rows.append(rollout(p, case[1:], target_rad=case[0], duration_s=duration_s)[0])
    return np.asarray(rows)


class PopulationEvaluator:
    """One persistent pool per GA; parent owns all randomness and preserves order."""
    def __init__(self, workers=1, cache_size=0):
        if workers < 1 or cache_size < 0:
            raise ValueError("workers must be positive and cache_size nonnegative")
        self.workers = workers
        self.pool = None
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __enter__(self):
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"))
        return self

    def __exit__(self, *args):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def evaluate(self, controllers, coefficients, cases, duration_s=10., *, command_kind="compensated_cubic"):
        controllers = np.asarray(controllers, dtype=float)
        cases = np.asarray(cases, dtype=float)
        coefficients = None if coefficients is None else np.asarray(coefficients, dtype=float)
        if controllers.ndim != 2 or controllers.shape[1] != 52 or not len(controllers) or not np.isfinite(controllers).all():
            raise ValueError("Expected a nonempty N x 52 controller population")
        if cases.ndim != 2 or cases.shape[1] != 5 or not len(cases) or not np.isfinite(cases).all():
            raise ValueError("Expected a nonempty K x 5 case bank")
        if command_kind not in ("compensated_cubic", "polynomial"):
            raise ValueError("Unknown command law")
        if coefficients is not None:
            allowed = (1, 3, 4) if command_kind == "polynomial" else (3,)
            if (coefficients.ndim != 3 or coefficients.shape[:2] != (len(controllers), 2) or
                    coefficients.shape[2] not in allowed or not np.isfinite(coefficients).all()):
                raise ValueError("Expected one finite coefficient matrix per controller")
        tasks = [(p, None if coefficients is None else coefficients[i], cases, duration_s, command_kind)
                 for i, p in enumerate(controllers)]
        if self.cache_size:
            bank = (cases.shape, cases.tobytes(), duration_s, command_kind,
                    None if coefficients is None else coefficients.shape[1:])
            keys = [(bank, p.tobytes(), None if c is None else c.tobytes()) for p, c, _, _, _ in tasks]
            missing = OrderedDict((key, task) for key, task in zip(keys, tasks) if key not in self.cache)
            jobs = list(missing.values())
            calculated = self.pool.map(evaluate_individual, jobs) if self.pool else map(evaluate_individual, jobs)
            self.cache.update(zip(missing, calculated))
            metrics = np.array([self.cache[key] for key in keys])
            for key in keys:
                self.cache.move_to_end(key)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            results = self.pool.map(evaluate_individual, tasks) if self.pool else map(evaluate_individual, tasks)
            metrics = np.asarray(list(results))
        return np.array([fitness(m) for m in metrics]), metrics


def load_controller_population(path):
    """Old Simple_Afferents checkpoints must not silently seed the new experiment."""
    with np.load(path, allow_pickle=False) as data:
        required = {"schema", "model", "dt_ms", "physical_controllers", "experiment"}
        if not required <= set(data.files):
            raise ValueError("Expected a current final_population.npz; rerun the balance search")
        if str(data["schema"]) != POPULATION_SCHEMA or str(data["model"]) != MODEL or float(data["dt_ms"]) != DT_MS:
            raise ValueError("Incompatible population model or integration step")
        if str(data["experiment"]) not in ("equilibrium", "small_nm_weights"):
            raise ValueError("Choose a population from a balance experiment")
        experiment = str(data["experiment"])
        controllers = np.array(data["physical_controllers"])
    if controllers.ndim != 2 or controllers.shape[1] != 52 or not len(controllers) or not np.isfinite(controllers).all():
        raise ValueError("Invalid controller population")
    from .genome import ControllerDomain
    domain = ControllerDomain(small_weights=experiment == "small_nm_weights")
    for p in controllers:
        domain.encode(p)
    return controllers


def select_controller(controllers, cases, evaluator, duration_s=10.):
    scores, metrics = evaluator.evaluate(controllers, None, cases, duration_s)
    index = int(np.argmax(scores))  # Stable tie-breaking by population order.
    return controllers[index].copy(), index, scores, metrics
