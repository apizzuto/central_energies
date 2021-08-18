#!/usr/bin/env python
import numpy as np
from time import time
import os, pickle, pwd, argparse

parser = argparse.ArgumentParser(
    description='Sensitivity vs. Declination'
    )
parser.add_argument(
    '--dec', type=float, default=None,
    help='Declination in degrees')
args = parser.parse_args()

import csky as cy

dec = args.dec
delta_t = 1e5

gammas = np.linspace(1.5, 4., 21)
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

sens_dict = dict()

delta_t_days = delta_t / 86400.

src = cy.utils.Sources(ra=ra,
    dec=np.radians(dec),
    mjd=mjd,
    sigma_t=0.,
    t_100=delta_t_days)
tr = cy.get_trial_runner(conf, ana=ana, src=src)
try:
    bg_trials = tr.get_many_fits(5000)
    print(np.count_nonzero(bg_trials.ts))
    chi2 = cy.dists.Chi2TSD(bg_trials)
    median = chi2.median()
except:
    median = np.median(bg_trials.ts)

for gamma in gammas:
    print(f"Beginning gamma: {gamma:.3f}")

    cy.CONF['src'] = src
    cy.CONF['mp_cpus'] = 5

    sens_no_cut = get_sens(gamma, median, delta_t)
    sens_dict[gamma] = sens_no_cut

output_dir = cy.utils.ensure_dir(f'/home/{username}/central_dnde/central_energies/results/')
output_f = output_dir + f'sens_vs_gamma_dec_{dec:.1f}.pkl'
with open(output_f, 'wb') as fo:
    pickle.dump(sens_dict, fo)
