import utils.reader
import itertools as it
import exp.run_exp as rexp

# defining male and female lists
M_SUBJECTS = utils.reader.M_SUBJECTS
F_SUBJECTS = utils.reader.F_SUBJECTS

# generating a list of all the possible 2-persons combinations, separately for males and females
M_combs = list(it.combinations(M_SUBJECTS, 2))
F_combs = list(it.combinations(F_SUBJECTS, 2))


def run_multi_experiment(gend, **kwargs):
    if gend == 'M':
        combs = M_combs
    elif gend == 'F':
        combs = F_combs
    for pair in combs:
        rexp.run_experiment(gender=gend, subjects=pair, **kwargs)

    """
    use run_exp to all possible 2-persons combinations within a gender.
    :param gend: the relevant gender
    :param **kwargs: rest of run_exp arguments  
    """