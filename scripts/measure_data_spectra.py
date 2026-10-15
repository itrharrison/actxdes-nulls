import pickle

from spectra import NullSpectrum

config_fname = sys.argv[1]

nullspectra = NullSpectrum(config_fname=config_fname)

nullspectra.setup_binning()
nullspectra.setup_tracer_fields()

for null_test in nullspectra.nulltests:

    print('Measuring data null spectra {}...'.format(null_test['name']))

    null_test_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls.pkl'.format(null_test['name']))

    null_test['kappa_field'] = nullspectra.setup_act_field(null_test)
    
    nullspectra.setup_workspaces(null_test)
    cl_decoupled = nullspectra.measure_cls(null_test)

    pickle.dump(cl_decoupled, open(null_test_cl_fname, 'wb'))