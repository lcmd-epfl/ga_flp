#Assume running this from the script directory

job_directory=$(pwd)
script=${1}
database=${2}
name="${1%%.py}"
output="${job_directory}/${name}.out"

echo "#!/bin/sh
#SBATCH --job-name=${name}
#SBATCH --mem=8gb
#SBATCH --cpus-per-task=4
#SBATCH --tasks=1
#SBATCH -o ${output}
hostname
python3 ${script} ${database} 
exit 0" > ${name}.job

sbatch ${name}.job



