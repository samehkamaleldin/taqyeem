from numpy.random import normal
from taqyeem import TaqKfoldCVExperiment

exp = TaqKfoldCVExperiment(name="simple_experiment", verbose=10)
exp.start_experiment()
for run_idx in range(5):
    for fold_idx in range(5):
        # train model >>>>>
        # eval model >>>>>
        alef_ap, alef_roc = normal(0.6, 0.1), normal(0.5, 0.1)
        geem_ap, geem_roc = normal(0.7, 0.1), normal(0.6, 0.1)
        exp.submit_cv_results("alef_model" , run_idx, fold_idx, metrics={"ap": alef_ap, "roc": alef_roc}, configs={"dataset": "nell"})
        exp.submit_cv_results("geem_model", run_idx, fold_idx, metrics={"ap": geem_ap, "roc": geem_roc}, configs={"dataset": "nell"})
exp.end_experiment()