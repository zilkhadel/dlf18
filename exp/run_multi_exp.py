import itertools as it

from utils.reader import M_SUBJECTS, F_SUBJECTS
from exp.run_exp import run_experiment


def run_multi_experiment(gender, **kwargs):
    """
    Use run_exp to all possible 2-persons combinations within a gender.
    :param gender: the relevant gender
    """

    # generate a list of all possible 2-persons combinations, according to specified gender
    combs = it.combinations(M_SUBJECTS, 2) if gender == 'M' else it.combinations(F_SUBJECTS, 2) if gender == 'F' else []

    for pair in combs:
        run_experiment(gender=gender, subjects=pair, **kwargs)
