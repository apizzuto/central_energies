import pycondor, argparse, sys, os
from glob import glob
import numpy as np

error = '/scratch/apizzuto/fast_response/condor/error'
output = '/scratch/apizzuto/fast_response/condor/output'
log = '/scratch/apizzuto/fast_response/condor/log'
submit = '/scratch/apizzuto/fast_response/condor/submit'

job = pycondor.Job('sensitivity_central_energy_constructions',
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
		'when_to_transfer_output = ON_EXIT', 
		'Requirements =  (Machine != "node128.icecube.wisc.edu")']
	)

for dec in [-30., 0., 30.]:
    for gamma in [2.0, 2.5, 3.0]:
        job.add_arg(f'--dec={dec} --index={gamma}')

dagman = pycondor.Dagman('central_energy_construction', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
