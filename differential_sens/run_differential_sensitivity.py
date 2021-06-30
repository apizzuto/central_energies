import numpy as np
import pickle, os, pwd
import seaborn as sns
import histlite as hl
from scipy.interpolate import PchipInterpolator
import argparse
import csky as cy

parser = argparse.ArgumentParser(description='Cetnral 90 energy construction')
parser.add_argument('--dec', type=float, default=None,
                    help='Declination in degrees')
parser.add_argument('--index', type=float, default=None,
                    help='Spectral index')
args = parser.parse_args()

timer = cy.timing.Timer()
time = timer.time

username = pwd.getpwuid(os.getuid())[0]
ana_dir = cy.utils.ensure_dir(f'/home/{username}/csky_cache')

try:
    ana = cy.get_analysis(cy.selections.repo, cy.selections.PSDataSpecs.IC86v4, dir=ana_dir)
    ana.save(ana_dir)
except:
    ana = cy.get_analysis(cy.selections.repo, cy.selections.PSDataSpecs.IC86v4)

cy.CONF['ana'] = ana
cy.CONF['mp_cpus'] = 5

dec = args.dec
gamma = args.index

src = cy.sources(0.0, dec, deg=True)
tr = cy.get_trial_runner(src=src)

with time('ps bg trials'):
    n_trials = 5000
    bg = cy.dists.Chi2TSD(tr.get_many_fits(n_trials))

def calc_power_law_sens(gamma):
    tr = cy.get_trial_runner(
        ana=ana, src=src, 
        flux=cy.hyp.PowerLawFlux(gamma)
    )
    sens = sens = tr.find_n_sig(
        bg.median(),
        0.9,
        n_sig_step=5,
        batch_size=100,
        tol=.05
    )
    return tr.to_E2dNdE(sens, E0=1e5, unit=1)

def calc_one_diff_sens_bin(bin_gamma, low_en, high_en):
    tr = cy.get_trial_runner(
        ana=ana, src=src, 
        flux=cy.hyp.PowerLawFlux(
            bin_gamma, 
            energy_range=(low_en, high_en)
        )
    )
    sens = sens = tr.find_n_sig(
        bg.median(),
        0.9,
        n_sig_step=5,
        batch_size=100,
        tol=.10,
        logging=False
    )
    return tr.to_E2dNdE(sens, E0=low_en, unit=1)

def calc_diff_sens(bins_per_decade=1, low=1e1, high=1e9, 
                   bin_gamma=2., logging=True):
    energy_bins = np.logspace(
        np.log10(low),
        np.log10(high),
        (np.log10(high) - np.log10(low)) * bins_per_decade + 1
    )
    diff_sens = {'bins': energy_bins, 'sens': [], 'gamma': bin_gamma}
    if logging:
        print('Running differential sens calculation')
    for low_en, high_en in zip(energy_bins[:-1], energy_bins[1:]):
        if logging:
            print(f'\t - doing bin from {low_en:.1e} to {high_en:.1e}')
        tmp_diff = calc_one_diff_sens_bin(bin_gamma, low_en, high_en)
        diff_sens['sens'].append(tmp_diff)
    return diff_sens

for n_bins_per_decade in [1, 2, 5]:
    res = calc_diff_sens(
        bins_per_decade=n_bins_per_decade, 
        low=1e2, 
        high=1e8, 
        bin_gamma=gamma)
    with open(
        f'results/dec_{dec:.1f}_index_{gamma:.1f}_bins_{n_bins_per_decade}_diff_sens.pkl', 
        'wb') as f:
        pickle.dump(res, f)