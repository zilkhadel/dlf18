import os
import pickle
from os.path import exists as pe
from os.path import join as pj
from os.path import split as ps


class Paths(object):
    tmp_dir = '/Temp' if os.name == 'nt' else os.environ.get('TMP', '/tmp')

    dlf18_repo_dir = os.environ['DLF18_REPO']

    dlf18_resources_repo_dir = os.environ['DLF18_RESOURCES']
    data_dir = pj(dlf18_resources_repo_dir, 'data')
    models_dir = pj(dlf18_resources_repo_dir, 'models')
    experiments_dir = pj(models_dir, 'exp')
    pretrained_dir = pj(models_dir, 'pretrained')


def mkdirs(dir_path):
    try:
        if not pe(dir_path):
            os.makedirs(dir_path)
    except Exception as ex:
        if type(ex) is not FileExistsError:
            raise ex
    return dir_path


def dump(iterable, file_name, append=False, delimiter='\t'):
    with open(file_name, 'a' if append else 'w', encoding='utf-8') as f:
        f.writelines((delimiter.join(str(ll) for ll in l) if type(l) != str and hasattr(l, '__iter__') else str(l)).rstrip('\r\n') + '\n' for l in ([iterable] if type(iterable) == str else iterable))


def objdump(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def objload(path):
    return pickle.load(open(path, 'rb'))
