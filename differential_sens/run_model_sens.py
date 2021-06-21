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

src = cy.sources(0.0, dec, deg=True)
tr = cy.get_trial_runner(src=src)

with time('ps bg trials'):
    n_trials = 5000
    bg = cy.dists.Chi2TSD(tr.get_many_fits(n_trials))

def load_model(name='Crab_lowen'):
    tmp = np.loadtxt(f'ps_models/{name}.csv', delimiter=', ')
    ens = tmp.T[0]
    fls = tmp.T[1] * 1e3 # From TeV cm^-2 s^-1 to GeV cm^-2 s^-1
    return ens, fls

def make_log_spline(name='Crab_lowen'):
    ens, fls = load_model(name=name)
    log_e = np.log10(ens)
    log_f = np.log10(fls)
    log_spl = PchipInterpolator(log_e, log_f)
    return log_spl

def make_model_spline(name='Crab_lowen'):
    log_spl = make_log_spline(name=name)
    spl = lambda e: 10.**(log_spl(np.log10(e)))
    return spl

def make_all_splines():
    model_splines = dict()
    for name in ['3C273', 'Crab_highen', 
                 'Crab_lowen', 'G40_5-0_5', 
                 'Mrk421']:
        ens, fls = load_model(name=name)
        model_splines[name] = {'spl': make_model_spline(name=name),
                               'low_e': np.min(ens),
                               'high_e': np.max(ens)}
    return model_splines

model_splines = make_all_splines()

for name, model in model_splines.items():
    e_bins = np.logspace(
        np.log10(model['low_e']), 
        np.log10(model['high_e']),
        51
    )
    mids = 10.**(np.log10(e_bins)[:-1] + np.diff(np.log10(e_bins))/2.)
    binned_flux = model['spl'](mids) / (mids**2) # models are E^2 dN/dE, we want dN/dE
    flux_model = cy.hyp.BinnedFlux(e_bins, binned_flux)
    model_splines[name]['binned_flux'] = flux_model

def get_model_sensitivity(name):
    tr = cy.get_trial_runner(
        ana=ana, src=src, 
        flux=model_splines[name]['binned_flux']
    )
    sens = sens = tr.find_n_sig(
        bg.median(),
        0.9,
        n_sig_step=5,
        batch_size=100,
        tol=.10,
        logging=False
    )
    low_en = model_splines[name]['binned_flux'].bins_energy[0]
    return tr.to_dNdE(sens, E0=low_en, unit=1)

for name in model_splines.keys():
    model_sens = get_model_sensitivity(name)
    model_splines[name]['sens'] = model_sens

with open(f'results/dec_{dec:.1f}_model_sensitivities.pkl', 'wb') as f:
    pickle.dump(model_splines, f)