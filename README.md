# SSD-LM

We present a demo of SSD-LM (**S**emi-autoregressive **S**implex-based **D**iffusion Language Model) in Google Colab [here](https://colab.research.google.com/drive/1vNKqvzzJQp3k89QPuns5ibsq-VNC9wGN?usp=sharing)!

For more details of SSD-LM, please check out our preprint [here](https://arxiv.org/abs/2210.17432).

Below introduces the steps to reproduce our experiments in the paper.

### Environment setup

We use Conda and provide a yaml file for our environment, `environment.yml` (env name: `ssdlm`). You can follow the steps [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) to set it up. We also use a Slurm system to train and evaluate our model. 

Note that for all shell and slurm scripts below and in the repository (ending with `.sh` or `.sbatch`), you may want to change the home directory paths within them (e.g., search and replace all `/private/home/xhan77`). 

For the slurm files, you may want to remove the placeholder and pass in your own partition name, device constraint (remove the line if not applicable), and job time limit (we use 72hrs but the longer the better). We also assume each compute node has 8 GPUs, but you can adjust according to your system as well. 

### Data processing

* `bash run_ssd_process_data.sh 2022`

This will download and process OpenWebText. 

### Model training

* `sbatch submit_template_ssd_model_train.sbatch`

This starts a distributed training for SSD-LM and saves to `logging/ssd_dbs25`. If the job is interrupted, simply resubmit the job via the command above (should automatically resume from the most recent recoverable checkpoint). Note that if the output directory is not empty and you want to start training from scratch again, you need to manually delete the non-empty output directory (`logging/ssd_dbs25`).

Alternatively, you can download files from [here](https://huggingface.co/xhan77/ssdlm/tree/main) and manually copy to `logging/ssd_dbs25`.

### Interactive inference

Please refer to the Colab [demo](https://colab.research.google.com/drive/1vNKqvzzJQp3k89QPuns5ibsq-VNC9wGN?usp=sharing) for details.

### Evaluation

#### Unconstrained generation

* `source loop_eval.sh`
* `source loop_eval_alt.sh`

Above starts unconstrained (prompted) generation and saves to the model directory `logging/ssd_dbs25`. The `alt` suffix means a sampling projection, and no suffix means multi-hot projection. 

* `source loop_scoring_eval.sh`

Above scores the generated continuations. The results can be found in files ending with `_ssd_eval.txt` and `_ssd_eval_sampling.txt`. 

* `source loop_baseline_gpt2.sh`

The GPT-2 baseline results can be obtained by running the command above. 

#### Controlled generation (via off-the-shelf sentiment classfiers)

* `source loop_eval_ctrsa.sh`
* `source loop_scoring_eval_ctrsa.sh`

This generates controlled continuations and scores the generations. The results are saved to files ending with `_ssd_ctrsa_eval.txt`. 

Please refer to our paper for more details about the experiment setup. 
