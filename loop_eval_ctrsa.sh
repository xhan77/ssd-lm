for TSTEP in 1000 2500 5000
do
  for TOPP in 0.2 0.5 0.9
  do
    for CTRLR in 0.0 100.0 500.0 2000.0
    do
      sbatch --export=ALL,TSTEP=$TSTEP,TOPP=$TOPP,LCTX=0,RTRUNC=150,DDEPTH=2,CTRLR=$CTRLR eval_template_ssd_model_ctrsa.sbatch
      sbatch --export=ALL,TSTEP=$TSTEP,TOPP=$TOPP,LCTX=0,RTRUNC=180,DDEPTH=1,CTRLR=$CTRLR eval_template_ssd_model_ctrsa.sbatch
      sbatch --export=ALL,TSTEP=$TSTEP,TOPP=$TOPP,LCTX=0,RTRUNC=188,DDEPTH=1,CTRLR=$CTRLR eval_template_ssd_model_ctrsa.sbatch
    done
  done
done
