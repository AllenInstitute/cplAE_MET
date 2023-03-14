import optuna
import numpy as np


def load_study(exp_name, results_folder="/home/fahimehb/Local/new_codes/cplAE_MET/data/results/"):
    storage = f'sqlite:///{results_folder}{exp_name}/{exp_name}.db'
    study = optuna.create_study(study_name=exp_name,
                                direction="maximize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage=storage,
                                load_if_exists=True)
    return study


def summarize_optuna(study, exp=False):
    '''Takes optuna study and summarize its contents
    Args:
    study: optuna study object
    exp: A boolian indicating whether the exp of the hyper-params were used'''

    print("Number of trials performed: ", len(study.trials))
    print("Best trial number: ", study.best_trial.number)
    print("Best trial value: ", study.best_trial.value)
    print("")
    if exp:
        print("Exp values of the best hyperparams")
        print(".........................................")
        for k, v in study.best_params.items():
            print(k, np.exp(v))
    else:
        print("Best hyper-params")
        print(".........................................")
        for k, v in study.best_params.items():
            print(k, v)