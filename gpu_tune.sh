#!/bin/bash

# This is a script for submitting ray tune jobs on the HPC3 cluster.

# Set up sbatch arguments

#SBATCH --job-name=hlDNN_tunev2                   ## Name of the job.
#SBATCH -A kgreif                                 ## account to charge 
#SBATCH -p free-gpu                               ## partition/queue name
#SBATCH --cpus-per-task=2                        ## Only need 1 CPU for each task
#SBATCH --gpus-per-task=2                         ## Use only 1 GPU
#SBATCH --nodes=2                                 ## (-N) number of nodes to use
#SBATCH --mem-per-cpu=3G                          ## Should only need standard amount of memory
#SBATCH --ntasks-per-node 1

#SBATCH --time=00-08:00:00                      

#SBATCH --error=./outfiles/%x.err       ## error log file
#SBATCH --output=./outfiles/%x.out      ## output file

#SBATCH --mail-type=END,FAIL                      ## Send email
#SBATCH --mail-user=kgreif@uci.edu                ## to this address


# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# Start head node background task
port=6900
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &

# Start worker nodes
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done

# Set up directory tree before sbatch arguments
homedir=$(pwd)
trdir="${homedir}/tuning"
mkdir -p ${trdir}

# On to running the job!
# Start by printing out hostname and date of job
echo "Found a node, here's some info: "
hostname; date
echo $SLURM_JOB_NAME
echo $SLURM_JOB_ID
echo "================================"

# Command to run tuning script
command="python hp_train.py"

# Run command
echo "================================"
echo "Will run command ${command}"
$command
echo -e "\nDone!"

# Finally move output files from outfiles to trdir
echo "================================"
echo "Transferring output files..."
mv ${homedir}/outfiles/${SLURM_JOB_NAME}.out ${trdir}
mv ${homedir}/outfiles/${SLURM_JOB_NAME}.err ${trdir}
