#!/bin/bash
#
#SBATCH -A p71008
#SBATCH -J tle
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --array=1-5
#SBATCH --time 71:59:00
#SBATCH --licenses="gpfs@eodc"
####SBATCH --qos=devel_0128
#SBATCH --qos=normal_0256
####SBATCH --partition=mem_0128
#SBATCH --partition=mem_0256
##########SBATCH --mail-type=BEGIN    # first have to state the type of event to occur
##########SBATCH --mail-user=<email@address.at>   # and then your email address


echo "=================================================="
date
echo "--------------------------------------------------"
source /eodc/private/tuwgeo/users/tle/programs/miniconda2/bin/activate rt1_env_test
python -u /eodc/private/tuwgeo/users/tle/code_new/rt1_s1_processing/rt1_input_loadstackparallel_line.py /eodc/private/tuwgeo/users/tle/code_new/rt1_s1_processing/config_vsc.ini -totalarraynumber 5 -arraynumber ${SLURM_ARRAY_TASK_ID}
echo "--------------------------------------------------"
date
echo "=================================================="
