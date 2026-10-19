import os
import numpy as np
import yaml
import pickle
from scipy import stats

from get_spectra import NullSpectrum

from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots
nullspectra = NullSpectrum(config_fname='scripts/null_list_publicmaps.yaml')
# nullspectra = NullSpectrum(config_fname='scripts/lmin20_newsims_null_list_publicmaps.yaml')

cl_theory = np.loadtxt('./figures/cl_unbinned_actplanck.txt')
cl_theory = cl_theory.reshape([4, 1901])

cl_fid = {}

for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):

    cl_theory_bin = cl_theory[int(bin_tag.lstrip('bin')) - 1]
    cl_theory_ells = np.arange(len(cl_theory_bin))
    cls_to_bin = np.interp(np.arange(nullspectra.ell_bins.lmax + 1), cl_theory_ells, cl_theory_bin)
    cl_fid[bin_tag] = nullspectra.ell_bins.bin_cell(np.array([cls_to_bin]))[0]

ells = nullspectra.ell_bins.get_effective_ells()

figures = {}
axs = {}

all_ptes = []
all_ptes_bysim = []
all_chi2 = []
analysis_mask = (ells > nullspectra.analysis_range[0]) * (ells < nullspectra.analysis_range[1])

for null_test in nullspectra.nulltests:

    if 'transfer_function' in null_test.keys():
        transfer_dir = nullspectra.dirs['transfer_dir']
        transfer_name = null_test['transfer_function']
    
        transfer_function = np.loadtxt(os.path.join(nullspectra.dirs['root'], transfer_dir, transfer_name))
        transfer_function = np.nan_to_num(transfer_function, nan=1.0, posinf=1., neginf=1.) # first two values are numerically not defined

        transfer_function = nullspectra.ell_bins.bin_cell(transfer_function)

    else:
        transfer_function = np.ones_like(cl_fid[bin_tag])

    if 'transfer_function_baseline' in null_test.keys():
        transfer_dir = nullspectra.dirs['transfer_dir']
        transfer_name = null_test['transfer_function']
    
        transfer_function_baseline = np.loadtxt(os.path.join(nullspectra.dirs['root'], transfer_dir, transfer_name))
        transfer_function_baseline = np.nan_to_num(transfer_function_baseline, nan=1.0, posinf=1., neginf=1.) # first two values are numerically not defined

        transfer_function_baseline = nullspectra.ell_bins.bin_cell(transfer_function_baseline)

    else:
        transfer_function_baseline = np.ones_like(cl_fid[bin_tag])

    null_test_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls.pkl'.format(null_test['name']))

    # figures[null_test['name']], axs[null_test['name']] = plt.subplots(1, 2, figsize=(2 * 4.5, 3.75))
    figures[null_test['name']], axs[null_test['name']] = plt.subplots(1, 1, figsize=(1 * 4.5, 3.75))
    ax = [axs[null_test['name']]]

    # load the null map spectra
    cl_decoupled = pickle.load(open(null_test_cl_fname, 'rb'))

    # load the covmat
    null_test_covmat_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_covmat.pkl'.format(null_test['name']))
    covmat_list = pickle.load(open(null_test_covmat_fname, 'rb'))
    corrcoef_list = pickle.load(open(null_test_covmat_fname.replace('covmat', 'corrcoef'), 'rb'))

    # if spectra of a map space null, just plot
    cl_to_plot = {}
    err_bars = {}
    pte = {}
    chi2 = {}
    pte_text = ''
    if null_test['map_null']:
        for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):
            cl_to_plot[bin_tag] = cl_decoupled[bin_tag] #/ cl_fid[bin_tag]

            invcov = np.linalg.inv(covmat_list[bin_tag])
            dat = cl_to_plot[bin_tag][analysis_mask]
            chi2[bin_tag] = np.dot(dat.T, np.dot(invcov[np.ix_(analysis_mask, analysis_mask)], dat))
            pte[bin_tag] = 1. - stats.chi2.cdf(chi2[bin_tag], len(dat))
            all_ptes.append(pte[bin_tag])
            all_chi2.append(chi2[bin_tag])

            err_bars[bin_tag] = np.sqrt(np.diag(covmat_list[bin_tag]))
            # import pdb; pdb.set_trace()

        # then plot the subtracted spectra
        for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):
            # plt.subplot(1,4,ibin + 1)
            ax[0].plot(ells, 1.e9 * cl_to_plot[bin_tag], '.', c=f'C{ibin}') 
            # ax[0].plot(ells, 1.e9 * cl_fid[bin_tag], '-', c='k', alpha=0.4)
            # also plot the error bars
            ax[0].errorbar(ells, 1.e9 * cl_to_plot[bin_tag], yerr=1.e9*err_bars[bin_tag], c=f'C{ibin}', label='{} $\chi^2 = {:.2f},$ PTE$ = {:.4f}$'.format(bin_tag.replace('bin', 'Bin '), chi2[bin_tag], pte[bin_tag]))
            if bin_tag=='bin4':
                print(null_test['name'], '{:.4f}'.format(pte[bin_tag]))
            # pte_text = pte_text + f'{bin_tag}: {pte[bin_tag]:.3f}\n'

        # ax[0].text(1000, 2., pte_text)
        ax[0].legend(fontsize='x-small', loc='upper left')
            
    # else identify the correct baseline and load it
    else:
        baseline_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls.pkl'.format(null_test['baseline_name']))
        baseline_cl_decoupled = pickle.load(open(baseline_cl_fname, 'rb'))

        for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):
            cl_to_plot[bin_tag] = (cl_decoupled[bin_tag] / transfer_function - baseline_cl_decoupled[bin_tag] / transfer_function_baseline) / cl_fid[bin_tag]
            err_bars[bin_tag] = np.sqrt(np.diag(covmat_list[bin_tag]))

            invcov = np.linalg.inv(covmat_list[bin_tag])
            dat = cl_to_plot[bin_tag][analysis_mask]
            chi2[bin_tag] = np.dot(dat.T, np.dot(invcov[np.ix_(analysis_mask, analysis_mask)], dat))
            pte[bin_tag] = 1. - stats.chi2.cdf(chi2[bin_tag],len(dat))
            all_ptes.append(pte[bin_tag])
            all_chi2.append(chi2[bin_tag])

        # then plot the subtracted spectra
        for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):
            # plt.subplot(1,4,ibin + 1)
            ax[0].plot(ells, cl_to_plot[bin_tag], '.', c=f'C{ibin}') 
            ax[0].errorbar(ells, cl_to_plot[bin_tag], yerr=err_bars[bin_tag], c=f'C{ibin}', label='{} $\chi^2 = {:.2f},$ PTE $= {:.4f}$'.format(bin_tag.replace('bin', 'Bin '), chi2[bin_tag], pte[bin_tag]))
            
            # pte_text = pte_text + f'{bin_tag}: {pte[bin_tag]:.3f}\n'
        ax[0].legend(fontsize='x-small', loc='upper left')
        # ax.text(1000, 2., pte_text, c=f'C{ibin}')

    # ax[0].set_suptitle(null_test['name'])
    ax[0].set_xlim([0,ells.max()])
    if null_test['map_null']:
        # plt.suptitle(null_test['name'].split('-')[0])
        plt.suptitle(null_test['name'])
        # ax[0].set_ylim([-2.e-10, 2.e-10])
        # ax[0].set_ylim([-1, 1])
        # ax[0].set_ylim([-1e-8, 5.e-8])
        ax[0].set_ylim([-1., 2.5])
        # ax[0].set_ylabel('$C^{\kappa_{\\rm ' + null_test['name'].split('-')[0].replace('_', '\_') + '}\gamma}_{\ell}$')
        ax[0].set_ylabel('$C^{\kappa_{\\rm ' + null_test['name'].replace('_', '\_') + '}\gamma}_{\ell}$')
    else:
        plt.suptitle('{}$-${}'.format(null_test['name'].split('-')[0], null_test['baseline_name'].split('-')[0]))
        # ax[0].set_ylim([-1.e-7, 1.e-7])
        # ax[0].set_ylim([-2.e-11, 2.e-11])
        ax[0].set_ylim([-5., 5.])
        ax[0].set_ylabel('($C^{\kappa_{\\rm ' + null_test['name'].split('-')[0].replace('_', '\_') + '}\gamma}_{\ell} - C^{\kappa_{\\rm ' + null_test['baseline_name'].split('-')[0].replace('_', '\_') + '}\gamma}_{\ell}) / C^{\kappa\gamma, {\\rm th.}}_{\ell}$')

    ax[0].fill_between(np.arange(ells.max()), ax[0].get_ylim()[0], ax[0].get_ylim()[1],
                    where=(np.arange(ells.max()) > nullspectra.analysis_range[1]),
                    color='silver', alpha=0.5)
    ax[0].fill_between(np.arange(ells.max()), ax[0].get_ylim()[0], ax[0].get_ylim()[1],
                    where=(np.arange(ells.max()) < nullspectra.analysis_range[0]),
                    color='silver', alpha=0.5)

    ax[0].axhline(0.0, color='k')#, linestyle='dashed', alpha=0.4)
    ax[0].set_xlabel('$\ell$')
    ax[0].set_xlim([0, 3000])

    # Dinv = np.diag(1 / np.sqrt(np.diag(covmat_list['total'])))
    # corr = Dinv @ covmat_list['total'] @ Dinv
    corr = corrcoef_list['total']
    cov = covmat_list['total']

    # ax[1].imshow(corr, origin='lower', cmap='magma')
    # ncorrbinstep = corr.shape[1] // 4
    # # ax[1].axis('off')
    # ax[1].set_xticks([ncorrbinstep / 2, 3. * ncorrbinstep / 2, 5.* ncorrbinstep / 2, 7. * ncorrbinstep / 2], ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4'])
    # ax[1].set_yticks([ncorrbinstep / 2, 3. * ncorrbinstep / 2, 5.* ncorrbinstep / 2, 7. * ncorrbinstep / 2], ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4'], rotation=90)
    # ax[1].axhline(len(ells) - 0.5, color='k', alpha=0.4)
    # ax[1].axvline(len(ells) - 0.5, color='k', alpha=0.4)
    # ax[1].axhline(2*len(ells) - 0.5, color='k', alpha=0.4)
    # ax[1].axvline(2*len(ells) - 0.5, color='k', alpha=0.4)
    # ax[1].axhline(3*len(ells) - 0.5, color='k', alpha=0.4)
    # ax[1].axvline(3*len(ells) - 0.5, color='k', alpha=0.4)
    # ax[1].axhline(4*len(ells) - 0.5, color='k', alpha=0.4)
    # ax[1].axvline(4*len(ells) - 0.5, color='k', alpha=0.4)
    # plt.subplots_adjust(wspace=0.1)
    
    if null_test['map_null']:
        plt.savefig('./figures/nsims400_{}.png'.format(null_test['name']), dpi=300, bbox_inches='tight')
    else:
        plt.savefig('./figures/nsims400_{}-{}.png'.format(null_test['name'], null_test['baseline_name']), dpi=300, bbox_inches='tight')

    
plt.figure(figsize=(2 * 4.5, 3.75))
plt.subplot(121)
plt.plot(np.sort(all_ptes), np.linspace(0.25, 1.75, len(all_ptes)), 'd', ms=3, label='Individual PTEs (arbitrary $y$)', zorder=-1, color='C1')
plt.legend(loc='upper left', fontsize='small')
plt.hist(all_ptes, histtype='step', bins=np.linspace(0., 1., int(np.sqrt(len(all_ptes)))), density=True, color='C0')
# plt.text(0.625, 1.5, 'PTE$_{min} = '+f'{np.min(all_ptes):.2f}$'+'\nPTE$_{max} = '+f'{np.max(all_ptes):.2f}$')
plt.axhline(1.0, linestyle='dashed', alpha=0.4, color='k')
plt.xlabel('PTE')
plt.subplot(122)
plt.hist(all_chi2, histtype='step', bins=int(np.sqrt(len(all_ptes))), density=True)
plt.plot(np.linspace(0,25,128), stats.chi2.pdf(np.linspace(0,25,128), len(dat)))
# plt.yscale('log')
plt.xlabel('$\chi^2$')
plt.savefig('./figures/nsims400_pte_hist.png', dpi=300, bbox_inches='tight')

plt.close('all')
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

ntests = len(nullspectra.nulltests)
nbins = 4

fig, ax = plt.subplots(layout='constrained')

viridis = cm.get_cmap('PRGn', 1000)
newcolors = viridis(np.linspace(0, 1, 1000))
pink = np.array([248/256, 24/256, 148/256, 1])
orange = np.array([255/256, 127/256, 80/256, 1])
newcolors[:10, :] = orange
newcolors[-10:, :] = pink
newcmp = ListedColormap(newcolors)

ptes_grid = np.array(all_ptes).reshape(ntests,nbins).T

psm = ax.imshow(ptes_grid, cmap=newcmp, vmin=0.0, vmax=1.0, origin='lower')

ax.set_xticks(np.arange(ntests), [ns['name'] for ns in nullspectra.nulltests], rotation=90)
ax.set_yticks(np.arange(nbins), [f'Bin {ibin}' for ibin in np.arange(1,nbins+1)])

fig.colorbar(psm, orientation='horizontal', label='PTE', ax=ax, location='top', fraction=0.1)

# ax.text(3.65, -0.1, f'{ptes_grid.max():.3f}')
# ax.text(9.65, 2.9, f'{0.00984804:.3f}')
# ax.text(10.65, 2.9, f'{ptes_grid.min():.3f}')

plt.savefig('./figures/nsims400_pte_grid.png', dpi=300, bbox_inches='tight')
