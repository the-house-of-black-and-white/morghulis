import logging


def create_dataset(dataset_name, data_dir):
    if dataset_name == 'widerface':
        from morghulis.widerface import Wider
        ds = Wider(data_dir)
    elif dataset_name == 'fddb':
        from morghulis.fddb import FDDB
        ds = FDDB(data_dir)
    elif dataset_name == 'afw':
        from morghulis.afw import AFW
        ds = AFW(data_dir)
    elif dataset_name == 'pascal_faces':
        from morghulis.pascal_faces import PascalFaces
        ds = PascalFaces(data_dir)
    elif dataset_name == 'mafa':
        from morghulis.mafa import Mafa
        ds = Mafa(data_dir)
    elif dataset_name == 'caltech':
        from morghulis.caltech_faces import CaltechFaces
        ds = CaltechFaces(data_dir)
    else:
        logging.error('Invalid dataset name %s', dataset_name)
        raise ValueError('Invalid dataset name %s' % dataset_name)
    return ds
