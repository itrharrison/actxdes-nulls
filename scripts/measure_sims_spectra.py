import pickle

from covmats import NullCovMat

config_fname = sys.argv[1]

nullcovmats = NullCovMat(config_fname=config_fname)

nullcovmats.get_spectra_set(sims_cross_mode='sims')
nullcovmats.make_covmat(plot_dir='./figures')