for TSTEP in 1000 2500
do
  for TOPP in 0.0 0.1 0.2 0.5 0.7 0.9 0.95 0.99
  do
    sbatch --export=ALL,TSTEP=$TSTEP,TOPP=$TOPP,LCTX=25,RTRUNC=150,DDEPTH=1,CTRLR=0.0 eval_template_ssd_model.sbatch
    sbatch --export=ALL,TSTEP=$TSTEP,TOPP=$TOPP,LCTX=50,RTRUNC=100,DDEPTH=2,CTRLR=0.0 eval_template_ssd_model.sbatch
    sbatch --export=ALL,TSTEP=$TSTEP,TOPP=$TOPP,LCTX=100,RTRUNC=0,DDEPTH=4,CTRLR=0.0 eval_template_ssd_model.sbatch
  done
done
