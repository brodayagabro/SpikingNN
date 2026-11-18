"""Real-coded GA: elitism, tournament selection, blend crossover and mutation.

Independent of the physical model. All randomness comes from a saved numpy RNG.
Scores are maximized. Generation 0 is evaluated; generations 1..G are offspring.
"""
from copy import deepcopy
import numpy as np


def reflect(values):
    folded = np.mod(values, 2.)
    return np.where(folded > 1, 2-folded, folded)


def schedule(generation, total, settings):
    cooling = .5*(1+np.cos(np.pi*(generation-1)/max(1, total-1)))
    sigma = settings['sigma_end'] + (settings['sigma_start']-settings['sigma_end'])*cooling
    probability = settings['mutation_probability']
    return sigma, probability


def initialize(center, size, rng, candidates=()):
    """Seeded center, nearby samples and 20% uniform immigrants."""
    population = reflect(center + rng.normal(0., .08, (size, len(center))))
    immigrants = max(1, size//5)
    population[-immigrants:] = rng.random((immigrants, len(center)))
    population[0] = center
    for i, candidate in enumerate(candidates):
        if i+1 >= size-immigrants:
            break
        population[i+1] = candidate
    return population


def reproduce(population, scores, rng, sigma, probability):
    size, dimensions = population.shape
    elite = max(1, int(np.ceil(.05*size)))
    order = np.argsort(-scores, kind='stable')
    children = np.empty_like(population)
    children[:elite] = population[order[:elite]]
    for i in range(elite, size):
        if rng.random() < .05:
            children[i] = rng.random(dimensions)
            continue
        parents = []
        for _ in range(2):
            candidates = rng.integers(size, size=3)
            parents.append(population[candidates[np.argmax(scores[candidates])]])
        alpha = rng.uniform(-.15, 1.15, dimensions)
        child = alpha*parents[0] + (1-alpha)*parents[1]
        child += (rng.random(dimensions) < probability)*rng.normal(0., sigma, dimensions)
        children[i] = reflect(child)
    return children


def optimize(center, evaluate, *, population, generations, seed, settings, callback=None,
             candidates=(), resume=None):
    """Deterministic optimization with complete evaluated checkpoints.

    Resume must use the same planned horizon, settings and evaluator. This
    contract is checked against the saved experiment manifest by train.py.
    """
    if population < 4 or generations < 1 or seed < 0:
        raise ValueError('Need population>=4, generations>=1, seed>=0')
    if not 0 < settings['sigma_end'] <= settings['sigma_start'] or not 0 <= settings['mutation_probability'] <= 1:
        raise ValueError('Invalid mutation schedule')
    rng = np.random.default_rng(seed)
    if resume is None:
        current = initialize(np.asarray(center), population, rng, candidates)
        first = 0
    else:
        current = np.asarray(resume['population'], dtype=float)
        if current.shape != (population, len(center)):
            raise ValueError('Checkpoint population shape mismatch')
        rng.bit_generator.state = deepcopy(resume['rng_state'])
        scores = np.asarray(resume['scores'])
        first = int(resume['generation'])+1
    for generation in range(first, generations+1):
        if generation:
            sigma, probability = schedule(generation, generations, settings)
            current = reproduce(current, scores, rng, sigma, probability)
        metrics = evaluate(current)
        scores = metrics[:, 0]
        if not np.isfinite(metrics).all():
            raise ValueError('Evaluator must return finite metrics, including failure penalties')
        state = dict(generation=generation, population=current, scores=scores, metrics=metrics,
                     rng_state=deepcopy(rng.bit_generator.state))
        if callback:
            callback(state)
    return state
