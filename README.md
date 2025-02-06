# Code for paper: Adapting to Evolving Adversaries with Regularized Continual Robust Training

Training code is located in the ``train`` directory, to run please run the commands below from within that directory.

To perform initial training on CIFAR-10 with WRN-28-10 architecture with L2 attacks ($\epsilon=0.5$) with adversarial l2 regularization (ALR) with $\lambda = 1$
```
python adv_train.py --arch wrn_28_10 --normalize --data_dir PATH_TO_DATA --model_dir PATH_TO_INITIALLY_TRAINED_RESULTS_DIR --dataset cifar10 --fname INITALLY_TRAINED_EXPERIMENT_NAME --procedure single --chkpt_iters 25 --epochs 100 --total_epochs 100 --latent_reg l2 --reg_noise adv --latent_reg_str 1 --attacks L2Attack --eps 0.5 --reg_num_iters 1
```

To perform initial training on CIFAR-10 with WRN-28-10 architecture with L2 attacks ($\epsilon=0.5$) with uniform regularization (UR) with $\sigma=2, \lambda = 1$
```
python adv_train.py --arch wrn_28_10 --normalize --data_dir PATH_TO_DATA --model_dir PATH_TO_INITIALLY_TRAINED_RESULTS_DIR --dataset cifar10 --fname INITALLY_TRAINED_EXPERIMENT_NAME --procedure single --chkpt_iters 25 --epochs 100 --total_epochs 100 --latent_reg l2 --reg_noise uniform --latent_reg_eps 2 --latent_reg_str 1 --attacks L2Attack --eps 0.5
```

To perform Croce et. al (2022)[1] fine-tuning from an initially trained model robust to L2Attacks ($\epsilon = 1$) with ALR ($\lambda=0.5$) to StAdv[2] attacks ($\epsilon=0.05$) on ImageNette:
```
python adv_train.py --arch ResNet18 --normalize --resume best --data_dir PATH_TO_DATA --model_dir PATH_TO_INITIALLY_TRAINED_RESULTS_DIR --dataset imagenette --model_dir_ft PATH_TO_FINETUNED_RESULTS_DIR --fname INITALLY_TRAINED_EXPERIMENT_NAME --ft_fname FINETUNED_EXPERIMENT_NAME --procedure finetune --chkpt_iters 10 --epochs 25 --total_epochs 25 --latent_reg l2 --attacks L2Attack StAdvAttack --eps 1 0.05 --lr_max 0.001 --lr_scheduler none  --reg_noise adv --latent_reg_str 0.5 --reg_num_iters 1
```

[1] Croce, Francesco, and Matthias Hein. "Adversarial Robustness against Multiple and Single $ l_p $-Threat Models via Quick Fine-Tuning of Robust Classifiers." *International Conference on Machine Learning*. PMLR, 2022.

[2] Xiao, Chaowei, et al. "Spatially Transformed Adversarial Examples." *International Conference on Learning Representations*. 2018.
