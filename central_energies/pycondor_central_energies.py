import pycondor, argparse, sys, os, pwd
from glob import glob
import numpy as np

username = pwd.getpwuid(os.getuid())[0]
if not os.path.exists(f'/scratch/{username}/'):
    os.mkdir(f'/scratch/{username}/')
if not os.path.exists(f'/scratch/{username}/central_dnde/'):
    os.mkdir(f'/scratch/{username}/central_dnde/')
if not os.path.exists(f'/scratch/{username}/central_dnde/condor/'):
    os.mkdir(f'/scratch/{username}/central_dnde/condor')

error = f'/scratch/{username}/central_dnde/condor/error'
output = f'/scratch/{username}/central_dnde/condor/output'
log = f'/scratch/{username}/central_dnde/condor/log'
submit = f'/scratch/{username}/central_dnde/condor/submit'

job = pycondor.Job('sensitivity_central_energy_constructions',
	# Pass this the path of where your executable lives
	'/home/apizzuto/central_dnde/calc_sens_with_cuts.py',
	error=error,
	output=output,
	log=log,
	submit=submit,
	getenv=True,
	universe='vanilla',
	verbose=2, 
	request_memory=8000,
	request_cpus=5,
	extra_lines= ['should_transfer_files = YES', 
		'when_to_transfer_output = ON_EXIT']
	)

for dec in [-30., 0., 30.]:
    for gamma in [2.0, 2.5, 3.0]:
        job.add_arg(f'--dec={dec} --index={gamma}')

dagman = pycondor.Dagman('central_energy_construction', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
