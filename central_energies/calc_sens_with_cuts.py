#!/usr/bin/env python
import numpy as np
from time import time
import os, pickle, pwd, argparse

parser = argparse.ArgumentParser(
    description='Cetnral 90 energy construction'
    )
parser.add_argument(
    '--dec', type=float, default=None,
    help='Declination in degrees')
parser.add_argument(
    '--index', type=float, default=None,
    help='Spectral index')
args = parser.parse_args()

import csky as cy

dec = args.dec
gamma = args.index

delta_ts = np.logspace(3., 6.5, 8)
ra = 0.
mjd = 56293.0

username = pwd.getpwuid(os.getuid())[0]
ana_dir = cy.utils.ensure_dir(f'/home/{username}/central_energies/csky_cache/')
# Pick your favorite data sample, I used GFU for no particular reason
ana = cy.get_analysis(cy.selections.repo, 
    'version-002-p05', 
    cy.selections.GFUDataSpecs.GFU_IC86_2011_2018, 
    dir=ana_dir)

conf = {'ana': ana,
       'extended': True,
       'space': "ps",
        'time': "transient",
        'sig': 'transient',
       }

def get_sens(gamma, thresh, delta_t, low_e=0., high_e=np.inf):
    tr = cy.get_trial_runner(conf, ana=ana, src=src, 
        flux = cy.hyp.PowerLawFlux(gamma, energy_range=(low_e, high_e)))
    ntrials = 6000 if delta_t < 3e5 else 2000
    sens = tr.find_n_sig(thresh, 0.9,
                       batch_size=ntrials,
                       max_batch_size=ntrials,
                       logging=False)
    if low_e < 1e3:
        e0 = 1.
    else:
        e0 = 1e3
    return tr.to_E2dNdE(sens, E0=e0, unit=1e3)

central_en_dict = dict()

for delta_t in delta_ts:
    central_en_dict[delta_t] = dict()
    print(f"Beginning time window: {delta_t:.1e}")
    delta_t_days = delta_t / 86400.

    src = cy.utils.Sources(ra=ra,
        dec=np.radians(dec),
        mjd=mjd,
        sigma_t=0.,
        t_100=delta_t_days)

    cy.CONF['src'] = src
    cy.CONF['mp_cpus'] = 5

    tr = cy.get_trial_runner(conf, ana=ana, src=src)
    try:
        bg_trials = tr.get_many_fits(2500)
        chi2 = cy.dists.Chi2TSD(bg_trials)
        median = chi2.median()
    except:
        median = np.median(bg_trials.ts)

    sens_no_cut = get_sens(gamma, median, delta_t)
    central_en_dict[delta_t]['no_cut'] = sens_no_cut

    central_en_dict[delta_t]['low_cut'] = []
    if dec < 0.:
       low_cuts = np.logspace(3., 6., 25)
    else:
        low_cuts = np.logspace(1., 5., 33) 
    for low_cut in low_cuts:
        central_en_dict[delta_t]['low_cut'].append(
            (low_cut, get_sens(gamma, median, delta_t, low_e=low_cut))
            )
        
    central_en_dict[delta_t]['high_cut'] = []
    if dec < 0.:
        high_cuts = np.logspace(5., 9., 25)
    else:
        high_cuts = np.logspace(3., 7., 25)
    for high_cut in high_cuts:
        central_en_dict[delta_t]['high_cut'].append(
            (high_cut, get_sens(gamma, median, delta_t, high_e=high_cut))
            )

output_dir = cy.utils.ensure_dir(f'/home/{username}/central_dnde/results/')
output_f = output_dir + f'central_energy_results_dec_{dec:.1f}_gamma_{gamma:.1f}.pkl'
with open(output_f, 'wb') as fo:
    pickle.dump(central_en_dict, fo)
