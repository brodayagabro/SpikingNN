"""One real-coded GA with elitism, tournament selection and cosine annealing."""
from dataclasses import asdict, dataclass
import time
import numpy as np


def reflect(unit):
    folded = np.mod(unit, 2.)
    return np.where(folded > 1, 2-folded, folded)


@dataclass(frozen=True)
class Schedule:
    sigma_start: float = .20
    sigma_end: float = .015
    probability_start: float = .60
    probability_end: float = .10

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all() or not 0 < self.sigma_end <= self.sigma_start:
            raise ValueError("Expected finite 0 < sigma_end <= sigma_start")
        if not 0 <= self.probability_end <= self.probability_start <= 1:
            raise ValueError("Expected 0 <= probability_end <= probability_start <= 1")

    def at(self, generation, generations):
        if not 1 <= generation <= generations:
            raise ValueError("Expected 1 <= generation <= generations")
        cooling = .5*(1+np.cos(np.pi*(generation-1)/max(1, generations-1)))
        return (self.sigma_end+(self.sigma_start-self.sigma_end)*cooling,
                self.probability_end+(self.probability_start-self.probability_end)*cooling)


def initial_population(center, size, rng):
    if size < 3:
        raise ValueError("Population must contain at least three individuals")
    population = reflect(center + rng.normal(0., .1, (size, len(center))))
    population[0] = center
    immigrants = max(1, size//4)
    population[-immigrants:] = rng.random((immigrants, len(center)))
    return population


def reproduce(population, scores, rng, sigma, probability):
    size, dimensions = population.shape
    order = np.argsort(-scores, kind="stable")
    children = [population[i].copy() for i in order[:max(1, int(np.ceil(.05*size)))]]
    while len(children) < size:
        if rng.random() < .05:
            child = rng.random(dimensions)
        else:
            parents = []
            for _ in range(2):
                candidates = rng.integers(size, size=3)
                parents.append(population[candidates[np.argmax(scores[candidates])]])
            alpha = rng.uniform(-.15, 1.15, dimensions)
            child = alpha*parents[0] + (1-alpha)*parents[1]
            child += (rng.random(dimensions) < probability)*rng.normal(0., sigma, dimensions)
            child = reflect(child)
        children.append(child)
    return np.asarray(children)


def optimize(center, evaluate, *, size=30, generations=100, seed=42,
             schedule=Schedule(), on_generation=None, on_state=None,
             resume_state=None, schedule_generations=None, initial_candidates=None):
    if generations < 1 or seed < 0:
        raise ValueError("Positive generations and a nonnegative seed are required")
    rng = np.random.default_rng(seed)
    horizon = generations if schedule_generations is None else schedule_generations
    if horizon < generations:
        raise ValueError("Schedule horizon must cover all requested generations")
    if resume_state is None:
        population = initial_population(np.asarray(center), size, rng)
        if initial_candidates is not None:
            candidates = np.asarray(initial_candidates, dtype=float)
            if (candidates.ndim != 2 or candidates.shape[1] != len(center) or
                    len(candidates) >= size or not np.isfinite(candidates).all() or
                    np.any((candidates < 0) | (candidates > 1))):
                raise ValueError("Initial candidates must fit the population and normalized domain")
            population[1:1+len(candidates)] = candidates
        history, first, elapsed = [], 1, 0.
    else:
        population = np.array(resume_state["population"], copy=True)
        history = [dict(row) for row in resume_state["history"]]
        first = int(resume_state["generation"]) + 1
        if population.shape != (size, len(center)) or not 1 < first <= generations:
            raise ValueError("Incompatible population or generation in checkpoint")
        rng.bit_generator.state = resume_state["rng_state"]
        elapsed = history[-1]["elapsed_s"]
        population = reproduce(population, resume_state["scores"], rng,
                               *schedule.at(first, horizon))
    started = time.monotonic() - elapsed
    for generation in range(first, generations+1):
        scores, metrics = evaluate(population)
        leader = int(np.argmax(scores))
        sigma, probability = schedule.at(generation, horizon)
        row = dict(generation=generation, evaluations=generation*size,
                   best_fitness=float(scores[leader]), median_fitness=float(np.median(scores)),
                   best_success=float(metrics[leader, :, 5].mean()),
                   diversity=float(population.std(axis=0).mean()), mutation_sigma=sigma,
                   mutation_probability=probability, elapsed_s=time.monotonic()-started)
        history.append(row)
        if on_generation:
            on_generation(history)
        if on_state:
            on_state(generation, population, scores, metrics, history, rng.bit_generator.state)
        if generation < generations:
            population = reproduce(population, scores, rng, *schedule.at(generation+1, horizon))
    return population, scores, metrics, history, rng.bit_generator.state
