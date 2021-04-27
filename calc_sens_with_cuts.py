import numpy as np
from time import time
import pickle
import argparse

parser = argparse.ArgumentParser(description='Cetnral 90 energy construction')
parser.add_argument('--dec', type=float, default=None,
                    help='Declination in degrees')
parser.add_argument('--index', type=float, default=None,
                    help='Spectral index')
args = parser.parse_args()

t0 = time()
print("Importing csky")
import csky as cy
t1 = time()
print(f"Imported csky. Took {t1 - t0} seconds")

dec = args.dec
gamma = args.index

delta_ts = np.logspace(3., 7., 9)
ra = 0.
mjd = 57000.

ana_dir = cy.utils.ensure_dir('/home/apizzuto/csky_cache/')
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

def get_sens(gamma, thresh, low_e=0., high_e=np.inf):
    tr = cy.get_trial_runner(conf, ana=ana, src=src, 
        flux = cy.hyp.PowerLawFlux(gamma, energy_range=(low_e, high_e)))
    sens = tr.find_n_sig(thresh, 0.9,
                       batch_size=4000,
                       max_batch_size=4000,
                       logging=False)
    if low_e < 1e3:
        e0 = 1.
    else:
        e0 = 1e3
    return tr.to_E2dNdE(sens, E0=e0, unit=1e3)

central_en_dict = dict()

for delta_t in delta_ts:
    central_en_dict[delta_t] = dict()
    print(f"Begging time window: {delta_t:.1e}")
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
        bg_trials = tr.get_many_fits(1000)
        chi2 = cy.dists.Chi2TSD(bg_trials)
        median = chi2.median()
    except:
        median = np.median(bg_trials.ts)

    sens_no_cut = get_sens(gamma, median)
    central_en_dict[delta_t]['no_cut'] = sens_no_cut

    central_en_dict[delta_t]['low_cut'] = []
    for low_cut in np.logspace(1, 5., 33):
        central_en_dict[delta_t]['low_cut'].append(
            (low_cut, get_sens(gamma, median, low_e=low_cut))
            )
        
    central_en_dict[delta_t]['high_cut'] = []
    for high_cut in np.logspace(3., 7., 25):
        central_en_dict[delta_t]['high_cut'].append(
            (high_cut, get_sens(gamma, median, high_e=high_cut))
            )

with open(f'./central_energy_results_dec_{dec:.1f}_gamma_{gamma:.1f}.pkl', 'wb') as fo:
    pickle.dump(central_en_dict, fo)
