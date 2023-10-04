submitter scslurm :: COMMANDS 
                  :: TASK REALIZATION CONFIGURATION {
    action run > exit_code {
        # Returns true iff the first parameter 
        # is the name of a defined variable
        function var_defined {
            eval '[[ -n ${'$1'+set} && ${'$1'} != "none" ]]'
        }

        # define script 
        wrapper="ducttape_job.sh"

        echo "#!/bin/bash" > $wrapper
        echo "set -e # stop on errors" >> $wrapper
        echo "set -o pipefail # stop on pipeline errors" >> $wrapper
        echo "set -u # stop on undeclared variables" >> $wrapper

        # print actual jobs 
        echo "$COMMANDS" >> $wrapper

        SLURM_ARGS="--job-name=$TASK"
        if (var_defined account); then
            SLURM_ARGS+=" -A $account"
        fi
        if (var_defined config); then
            SLURM_ARGS+=" -C ${config}"
        fi
        if (var_defined cpus); then
            SLURM_ARGS+=" --cpus-per-task=$cpus"
        fi
        if (var_defined mem); then
            SLURM_ARGS+=" --mem=$mem"
        fi
        if (var_defined gres); then
            SLURM_ARGS+=" --gres=$gres"
        fi
        if (var_defined time); then
            SLURM_ARGS+=" --time=$time"
        fi
        if (var_defined partition); then
            SLURM_ARGS+=" --partition=$partition"
        fi

        echo $SLURM_ARGS

        srun $SLURM_ARGS bash $wrapper
  }
}