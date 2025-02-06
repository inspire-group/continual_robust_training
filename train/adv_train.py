# script for adversarial training for obtaining scores for plotting ideal curve
import argparse
import torch
import torch.nn as nn
import numpy as np
import models
from torchvision import datasets, transforms
import utils
import attacks
import random
import logging
import os
import time
import data
from finetuning_utils import EWC, count_layers, fix_end_layers
from reg_utils import Regularizer
from timm.models import create_model

from tensorboardX import SummaryWriter

def lr_schedule(t, epochs, max_lr):
    if t / epochs < 0.5:
        return max_lr
    elif t / epochs < 0.75:
        return max_lr / 10.
    else:
        return max_lr / 100.

def curriculum_schedule(t, epochs, curr_type, finetune_args=None):
    if curr_type == "step":
        if t / epochs < 0.3:
            return 0
        elif t / epochs < 0.4:
            return 1.0 / 3
        elif t / epochs < 0.5:
            return 2.0 / 3
        else:
            return 1
    elif curr_type == "linear":
        if t / epochs < 0.3:
            return 0
        elif t / epochs < 0.7:
            return (t - 30) / 40.0
        else:
            return 1
    else:
        return None
    
def curriculum_attack(atk, X, y, model, epoch_lambda, attack_iters):
    X_adv = X
    for _ in range(attack_iters):
        output = model(X_adv)
        prob_y = output[:, y].diag()
        # j should be the class other than y with the largest softmax probability,
        # but this gives an equivalent index when curriculum >= 0
        prob_j = output.max(1)[0]
        index = torch.where(prob_j - prob_y <= epoch_lambda)[0]
        new_adv = atk(X_adv, y, num_iterations=1)
        # only update examples where index is 1
        X_adv[index] = new_adv[index]
    return X_adv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'imagenette', 'imagenet100'])
    parser.add_argument('--metrics', nargs='+')
    parser.add_argument('--arch', type=str, help='model architecture', default='resnet18')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adamw'], default='sgd')
    parser.add_argument('--normalize', action='store_true', help='whether data is normalized before passing into model or not')
    parser.add_argument('--attacks', type=str, nargs='+', default=['NoAttack'])
    parser.add_argument('--eps', type=float, nargs='+', default=[0])
    parser.add_argument('--fname', type=str, help='name of save checkpoint', required=True)
    parser.add_argument('--ft_fname', type=str, default='', help='name of save checkpoint if using to finetune from a pretrain chkpt')
    parser.add_argument('--procedure', type=str, choices=['avg', 'max', 'random', 'finetune', 'single'], default='avg', help='multiattack training procedure')
    parser.add_argument('--ewc', action='store_true', help='use elastic weight consolidation')
    parser.add_argument('--ewc_obj', type=str, help='objective function for elastic weight', default='acc')
    parser.add_argument('--ewc_str', type=float, default=1, help='weight on ewc term')
    parser.add_argument('--ewc_use_fisher', action='store_true', help='use fisher for computing elastic weights (otherwise just uses weight consolidation with no elasticity)')
    parser.add_argument('--model_dir', type=str, help='path to directory of ckpts', default='adv_train_resnet')
    parser.add_argument('--model_dir_ft', type=str, default='')
    parser.add_argument('--latent_reg', type=str, default='none', choices=['none', 'l2', 'var_reg', 'contrastive', 'asym_contrastive'])
    parser.add_argument('--use_self_features', action='store_true')
    parser.add_argument('--hn_tau_plus', type=float, default=0.1, help="tau-plus in HN")
    parser.add_argument('--hn_beta', type=float, default=1.0, help="beta in HN")
    parser.add_argument('--contrastive_adv_weight', type=float, default=1, help="weight of adv loss")
    parser.add_argument('--contrastive_aug', action='store_true', help="also use contrastive augmentations")
    parser.add_argument('--include_orig', action='store_true', help="for contrastive loss, include original input")
    parser.add_argument('--latent_reg_eps', type=float, default=0)
    parser.add_argument('-t', '--nce_t', default=0.5, type=float,
                    help='temperature')
    parser.add_argument('--reg_noise', type=str, default='adv', choices=['adv', 'gaussian', 'uniform'])
    parser.add_argument('--reg_uniform_freq', type=float, default=0, help='frequency of using uniform instead of adv (is used only if reg_noise == adv)')
    parser.add_argument('--reg_num_iters', type=int, default=10, help="number of iterations for computing regularization if not random")
    parser.add_argument('--reg_step_size', type=int, default=0, help="step size used for optimization")
    parser.add_argument('--latent_reg_str', type=float, default=1.0, help='strength of feature regularization')
    parser.add_argument('--atk_scheduling', type=int, nargs='+', help='epoch to start training with next attack type', default=[])
    parser.add_argument('--fix_last', type=int, help='fix the last n layers when performing attack scheduling', default=0)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--total_epochs', type=int, default=100, help='total epochs used for computing lr with multistep scheduling (pretrain + finetune)')
    parser.add_argument('--lr_max', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, choices=['multistep', 'cosine', 'none'], default='multistep')
    parser.add_argument('--resume', help='epoch to resume training from if corresponding ckpt file exists')
    parser.add_argument('--num_iters', type=int, default=10)
    parser.add_argument('--chkpt_iters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--use_syn', action='store_true')
    parser.add_argument('--batch_size_syn', type=int, default=350)
    parser.add_argument('--syn_data_path', type=str, default='data/cifar10_ddpm.npz')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument("--curriculum", choices=["step", "linear", "finetune"], help="Lambda schedule for curriculum learning", default=None)
    args = parser.parse_args()

    if not args.model_dir_ft:
        basename = os.path.basename(args.model_dir)
    else:
        basename = os.path.basename(args.model_dir_ft)
        
    tbdir = f'logs/{basename}/tensorboard'
    npdir = f'logs/{basename}/acc_files'

    if not os.path.exists(tbdir):
        os.makedirs(tbdir)
    if not os.path.exists(npdir):
        os.makedirs(npdir)

    if args.ft_fname:
        writer = SummaryWriter(log_dir=f'{tbdir}/{args.ft_fname}')
    else:
        writer = SummaryWriter(log_dir=f'{tbdir}/{args.fname}')

    # adding model name to model dir
    args.model_dir = args.model_dir + '/' + args.fname
    if args.model_dir_ft:
        args.model_dir_ft = args.model_dir_ft + '/' + args.ft_fname
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if args.model_dir_ft and not os.path.exists(args.model_dir_ft):
        os.makedirs(args.model_dir_ft)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # load data
    train_loader, test_loader, mean, std, num_classes = data.__dict__[args.dataset](args.data_dir, args.batch_size, args.use_syn, args.syn_data_path, args.batch_size_syn, contrastive=args.contrastive_aug)

    # set up model
    if args.arch in models.__dict__ or 'convnext' in args.arch:
        if 'convnext' in args.arch:
            args.lr_max = 0.001
            model = create_model(args.arch, pretrained=False, 
                                 num_classes=num_classes
                                 #drop_path_rate=args.drop_path,
        #layer_scale_init_value=args.layer_scale_init_value,
        #head_init_scale=args.head_init_scale,
        )
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)
        if args.normalize:
            model = models.apply_normalization(model, mean, std)
        model = model.cuda()
       #model = nn.DataParallel(model).cuda()
    else:
        raise ValueError('unsupported architecture')

    # initialize adversary
    atks = []
    test_atks = []
    regularizers = []
    assert(len(args.attacks) > 0)
    assert(len(args.attacks) == len(args.eps))
    #for e in args.atk_scheduling:
    #    assert(e < args.epochs)
    num_iters = args.num_iters
    for i, attack in enumerate(args.attacks):
        if attack == 'NoAttack':
            atk = attacks.__dict__[attack]()
            test_atk = attacks.__dict__[attack]()
        else:
            eps = args.eps[i]
            if "cifar" in args.dataset:
                dataset_name = "cifar"
            else:
                dataset_name = "imagenet"
            
            # for training non-LP attacks, we will take the step size to be eps / (num_iters - 2)
            step_size = eps / (num_iters - 2)

            if args.latent_reg != 'none':
                reg_params = (args.hn_tau_plus, args.hn_beta, args.contrastive_adv_weight, args.use_self_features)
                reg_step_size = args.reg_step_size
                if args.reg_step_size == 0 and args.reg_num_iters == 1:
                    reg_step_size = eps
                elif args.reg_step_size == 0 and args.reg_num_iters > 3:
                    reg_step_size = eps / (args.reg_num_iters - 2)
                else:
                    reg_step_size = eps / args.reg_num_iters
                if args.reg_noise != 'adv':
                    regularizer = Regularizer(model, args.latent_reg, attack, args.latent_reg_eps, args.reg_noise, uniform_freq=args.reg_uniform_freq, include_orig=args.include_orig, temp=args.nce_t, loss_params=reg_params, dataset_name=dataset_name)
                elif args.reg_noise == 'adv' and args.reg_uniform_freq == 0:
                    regularizer = Regularizer(model, args.latent_reg, attack, eps, args.reg_noise, uniform_freq=args.reg_uniform_freq, include_orig=args.include_orig, temp=args.nce_t, loss_params=reg_params, num_steps=args.reg_num_iters, step_size=reg_step_size, dataset_name=dataset_name)
                else:
                    regularizer = Regularizer(model, args.latent_reg, attack, eps, args.reg_noise, uniform_freq=args.reg_uniform_freq, epsilon_noise_only=args.latent_reg_eps ,include_orig=args.include_orig, temp=args.nce_t, loss_params=reg_params, num_steps=args.reg_num_iters, step_size=reg_step_size, dataset_name=dataset_name)
                regularizers.append(regularizer)
            if attack == 'FastLagrangePerceptualAttack':
                if dataset_name == "cifar":
                    atk = attacks.__dict__[attack](model, bound=eps, lpips_model='alexnet_cifar', num_iterations=num_iters)
                    test_atk = attacks.__dict__['LPIPSAttack'](model, bound=eps, lpips_model='alexnet_cifar', num_iterations=num_iters)
                else:
                    atk = attacks.__dict__[attack](model, bound=eps, lpips_model='alexnet', num_iterations=num_iters)
                    test_atk = attacks.__dict__['LPIPSAttack'](model, bound=eps, lpips_model='alexnet', num_iterations=num_iters)
            else:
                try:
                    atk = attacks.__dict__[attack](model, dataset_name=dataset_name,bound=eps, num_iterations=num_iters, step_size=step_size)
                except:
                    atk = attacks.__dict__[attack](model, bound=eps, num_iterations=num_iters)
                test_atk = atk
        atks.append(atk)
        test_atks.append(test_atk)

    # run training
    logger = logging.getLogger(__name__)
    logging_f = args.fname
    logging_md = args.model_dir
    if args.model_dir_ft:
        logging_f = args.ft_fname
        logging_md = args.model_dir_ft
    logging_name = 'at_{}_output.log'.format(logging_f)
    if os.path.exists(os.path.join(logging_md, logging_name)):
        log_fcount = 2
        logging_name = 'at_{}_output_{}.log'.format(logging_f, log_fcount)
        while os.path.exists(os.path.join(logging_md,logging_name)):
            log_fcount += 1
            logging_name = 'at_{}_output_{}.log'.format(logging_f, log_fcount)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(logging_md, logging_name)),
            logging.StreamHandler()
        ])

    logger.info(args)

    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_max, betas=(0.9, 0.95), weight_decay=5e-4)

    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)

    best_test_robust_acc = 0
    clean_accs = np.zeros(args.total_epochs)
    avg_rob_accs = np.zeros(args.total_epochs)
    union_rob_accs = np.zeros(args.total_epochs)
    indiv_rob_accs = np.zeros((len(atks), args.total_epochs))

    if args.resume:
        if args.resume != 'best':
            start_epoch = int(args.resume)
            load_data = torch.load(os.path.join(args.model_dir, f'model_{start_epoch-1}.pth'))
            model.load_state_dict(load_data['state_dict'])
            opt.load_state_dict(torch.load(os.path.join(args.model_dir, f'opt_{start_epoch-1}.pth')))
            if args.lr_scheduler == 'cosine':
                scheduler.load_state_dict(torch.load(os.path.join(args.model_dir, f'sch_{start_epoch-1}.pth')))
            logger.info(f'Resuming at epoch {start_epoch}')
        
            if os.path.exists(os.path.join(args.model_dir, f'model_best.pth')):
                best_test_robust_acc = torch.load(os.path.join(args.model_dir, f'model_best.pth'))['test_robust_acc']
        else:
            start_epoch = 0 # starting from 0 for finetuning
            # we are performing finetuning and loading from the best checkpoint
            load_data = torch.load(os.path.join(args.model_dir, f'model_best.pth'))
            model.load_state_dict(load_data['state_dict'])
            # we'll use the new optimizer object with learning rate and scheduler set

        # load numpy data
        #clean_accs = np.load(f'{npdir}/{args.fname}_clean_accs.npy')
        #avg_rob_accs = np.load(f'{npdir}/{args.fname}_avg_rob_accs.npy')
        #union_rob_accs = np.load(f'{npdir}/{args.fname}_union_rob_accs.npy')
        #indiv_rob_accs = np.load(f'{npdir}/{args.fname}_indiv_rob_accs.npy')

    else:
        start_epoch = 0

    criterion = nn.CrossEntropyLoss(reduction='none')

    if args.atk_scheduling:
        all_atks = False
        epoch_atks = [atks[0]]
        if args.latent_reg != 'none':
            epoch_regs = [regularizers[0]]
    else:
        epoch_atks = atks
        all_atks = True
        if args.latent_reg != 'none':
            epoch_regs = regularizers

    atk_sched_idx = 0
    fixed_layers = False

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    train_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0        
        train_n = 0
        train_n_rob = 0
        acc_norms = [[0., 0.] for _ in range(len(atks))]

        if not all_atks:
            if epoch == args.atk_scheduling[atk_sched_idx]:
                epoch_atks.append(atks[atk_sched_idx + 1])
                atk_sched_idx += 1
                if args.latent_reg != 'none':
                    epoch_regs.append(regularizers[atk_sched_idx + 1])
                if args.ewc and args.ewc_str != 0:
                    ewc = EWC(model, train_loader, epoch_atks[:len(epoch_atks)-1], obj=args.ewc_obj, use_fisher=args.ewc_use_fisher)

            if atk_sched_idx == len(args.atk_scheduling):
                all_atks = True
                epoch_atks = atks
                epoch_regs = regularizers
                if args.ewc and args.ewc_str != 0:
                    ewc = EWC(model, train_loader, epoch_atks[:len(epoch_atks)-1], obj=args.ewc_obj, use_fisher=args.ewc_use_fisher)

        if args.curriculum == "finetune":
            if len(args.atk_scheduling) != 1:
                raise NotImplementedError
            ft_epoch = args.atk_scheduling[0]
            curr_lambda = [None, curriculum_schedule(epoch - ft_epoch, args.epochs - ft_epoch, 'linear')]
        else:
            curr_lambda = [curriculum_schedule(epoch, args.epochs, args.curriculum)]

        if atk_sched_idx > 0 and not fixed_layers and args.fix_last > 0:
            total_num_layers = count_layers(model)
            fix_end_layers(model, total_num_layers - args.fix_last)
            fixed_layers = True

        for i, batch in enumerate(train_loader):
            if args.contrastive_aug:
                (Xt1, Xt2, X), y = batch
                Xt1 = Xt1.cuda()
                Xt2 = Xt2.cuda()
            else:
                X, y = batch
            model.eval()
            X, y = X.cuda(), y.cuda()
            if args.lr_scheduler == 'multistep':
                lr = lr_schedule(epoch + (i + 1) / len(train_loader), args.total_epochs, max_lr=args.lr_max)
                opt.param_groups[0].update(lr=lr)
            norm_curr = 0
            step_regs = []
            #print('num attacks using', len(epoch_atks))
            if len(epoch_atks) == 1:
                step_attacks = epoch_atks
                epoch_lambda = curr_lambda[0]
                if args.latent_reg != "none":
                    step_regs = epoch_regs
            elif args.procedure == 'single':
                step_attacks = [epoch_atks[-1]]
                epoch_lambda = curr_lambda[0]
                if args.latent_reg != 'none':
                    step_regs = [epoch_regs[-1]]
            elif args.procedure == 'finetune':
                # sample which norm to use for the current batch
                if all([val[1] > 0 for val in acc_norms]):
                    ps = [val[0] / val[1] for val in acc_norms]
                else:
                    ps = [.5 for i in range(len(acc_norms))]
                ps = [1. - val for val in ps]
                if len(epoch_atks) < len(acc_norms):
                    # set probability of picking attacks 
                    # that haven't been scheduled yet to 0
                    for i in range(len(acc_norms)-1, len(epoch_atks)-1, -1):
                        ps[i] = 0
                norm_curr = random.choices(range(len(ps)), weights=ps)[0]
                step_attacks = [epoch_atks[norm_curr]]
                if args.latent_reg != 'none':
                    step_regs = [epoch_regs[norm_curr]]
                if args.curriculum == "finetune":
                    epoch_lambda = curr_lambda[norm_curr]
                else:
                    epoch_lambda = curr_lambda[0]
            elif args.procedure == 'random':
                r = random.choice(range(len(epoch_atks)))
                step_attacks = [epoch_atks[r]]
                if args.latent_reg != 'none':
                    step_regs = [epoch_regs[r]]
                epoch_lambda = curr_lambda[0]
            else:
                step_attacks = epoch_atks
                if args.latent_reg != 'none' and args.reg_noise == "adv":
                    step_regs = epoch_regs
                elif args.latent_reg != 'none':
                    # for random noise the same regularization is applied across threat model
                    # we will incorporate it only once
                    step_regs = [epoch_regs[0]] 
                epoch_lambda = curr_lambda[0]

            all_adv = []
            for atk in step_attacks:
                if args.curriculum is None or epoch_lambda is None:
                    X_adv = atk(X, y)
                else:
                    X_adv = curriculum_attack(atk, X, y, model, epoch_lambda, args.num_iters)
                all_adv.append(X_adv)
            all_adv =  torch.cat(all_adv)
            all_labels = torch.cat([y for atk in step_attacks])
            all_adv = all_adv.detach()

            model.train()
            robust_output = model(all_adv)
            robust_loss = criterion(robust_output, all_labels)
            if args.procedure == 'max':
                robust_loss, _ = robust_loss.resize(len(step_attacks), X.size()[0]).max(0)
            robust_loss = robust_loss.mean()

            if atk_sched_idx > 0 and args.ewc and args.ewc_str != 0:
                robust_loss +=  args.ewc_str * ewc.penalty(model)
           
            if args.latent_reg_str != 0:
                for regularizer in step_regs:
                    if args.contrastive_aug:
                        robust_loss += args.latent_reg_str * regularizer.compute_loss(X, images_t1=Xt1, images_t2=Xt2)
                    else:
                        computed_loss = regularizer.compute_loss(X)
                        #print("Computed loss for batch:", computed_loss.item())
                        #print("Adv loss no reg:", robust_loss.item())
                        robust_loss += args.latent_reg_str * regularizer.compute_loss(X)

            opt.zero_grad()
            robust_loss.backward()
            opt.step()
            if args.lr_scheduler == 'cosine':
                scheduler.step()

            output = model(X)
            loss = criterion(output, y)
            loss = loss.mean()

            train_robust_loss += robust_loss.item() * all_labels.size(0)
            correct = (robust_output.max(1)[1] == all_labels).sum().item()
            train_robust_acc += correct
            acc_norms[norm_curr][0] += correct
            acc_norms[norm_curr][1] += y.size(0)
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            train_n_rob += all_labels.size(0)

        train_time = time.time()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        test_n_rob = 0
        test_accs_per_atk_type = np.zeros(len(test_atks))
        test_union_acc = 0
        for i, (X, y) in enumerate(test_loader):
            with torch.no_grad():
                X, y = X.cuda(), y.cuda()

                if args.procedure == "random" or args.procedure == "finetune" or args.procedure == "single":
                    # more efficient for model selection
                    r = random.choice(range(len(test_atks)))
                    with torch.enable_grad():
                        X_adv = test_atks[r](X, y)
                    robust_output = model(X_adv)
                    all_labels = y
                    robust_loss = criterion(robust_output, y).mean()
                    test_union_acc += (robust_output.max(1)[1] == y).sum().item()
                    
                else:
                    all_adv = []
                    for test_atk in test_atks:
                        with torch.enable_grad():
                            X_adv = test_atk(X, y)
                        all_adv.append(X_adv)
                    all_adv =  torch.cat(all_adv)
                    all_labels = torch.cat([y for atk in test_atks])

                    robust_output = model(all_adv)
                    test_union_corr = 1
                    for i in range(len(test_atks)):
                        out_atk_i = robust_output[i*len(y):(i+1)*len(y)]
                        correct_atk_i = (out_atk_i.max(1)[1] == y)
                        test_accs_per_atk_type[i] += correct_atk_i.sum().item()
                        test_union_corr *= correct_atk_i
                    test_union_acc += test_union_corr.sum().item()

                    robust_loss = criterion(robust_output, all_labels)

                    if args.procedure == 'max':
                        robust_loss, _ = robust_loss.resize(len(step_attacks), X.size()[0]).max(0)
                    robust_loss = robust_loss.mean()

                output = model(X)
                loss = criterion(output, y)
                loss = loss.mean()

                test_robust_loss += robust_loss.item() * all_labels.size(0)
                test_robust_acc += (robust_output.max(1)[1] == all_labels).sum().item()
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)
                test_n_rob += all_labels.size(0)

        test_accs_per_atk_type /= test_n

        test_time = time.time()
        logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
            epoch, train_time - start_time, test_time - train_time,
            train_loss/train_n, train_acc/train_n, train_robust_loss/train_n_rob, train_robust_acc/train_n_rob,
            test_loss/test_n, test_acc/test_n, test_robust_loss/test_n_rob, test_robust_acc/test_n_rob)
        
        writer.add_scalar('train/clean_loss', train_loss/train_n, epoch)
        writer.add_scalar('train/clean_acc', train_acc/train_n, epoch)
        writer.add_scalar('train/robust_loss', train_robust_loss/train_n_rob, epoch)
        writer.add_scalar('train/robust_acc', train_robust_acc/train_n_rob, epoch)

        writer.add_scalar('test/clean_loss', test_loss/test_n, epoch)
        writer.add_scalar('test/clean_acc', test_acc/test_n, epoch)
        writer.add_scalar('test/avg_robust_loss', test_robust_loss/test_n_rob, epoch)
        writer.add_scalar('test/avg_robust_acc', test_robust_acc/test_n_rob, epoch)
        writer.add_scalar('test/union_acc', test_union_acc/test_n, epoch)


        clean_accs[epoch] = test_acc/test_n
        avg_rob_accs[epoch] = test_robust_acc/test_n_rob
        union_rob_accs[epoch] = test_union_acc/test_n_rob
        for i in range(len(test_accs_per_atk_type)):
            writer.add_scalar(f'acc_per_atk/{args.attacks[i]}_{args.eps[i]}', test_accs_per_atk_type[i], epoch)
            indiv_rob_accs[i, epoch] = test_accs_per_atk_type[i]

        # save checkpoint
        if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == args.epochs:
            # save numpy data
            if not args.ft_fname:
                np.save(f'{npdir}/{args.fname}_clean_accs.npy', clean_accs)
                np.save(f'{npdir}/{args.fname}_avg_rob_accs.npy', avg_rob_accs)
                np.save(f'{npdir}/{args.fname}_union_rob_accs.npy', union_rob_accs)
                np.save(f'{npdir}/{args.fname}_indiv_rob_accs.npy', indiv_rob_accs)
            else:
                np.save(f'{npdir}/{args.ft_fname}_clean_accs.npy', clean_accs)
                np.save(f'{npdir}/{args.ft_fname}_avg_rob_accs.npy', avg_rob_accs)
                np.save(f'{npdir}/{args.ft_fname}_union_rob_accs.npy', union_rob_accs)
                np.save(f'{npdir}/{args.ft_fname}_indiv_rob_accs.npy', indiv_rob_accs)

            # save model checkpoint
            save_dict = {
                    'state_dict':model.state_dict(),
                    'test_robust_acc':test_robust_acc/test_n,
                    'test_robust_loss':test_robust_loss/test_n,
                    'test_loss':test_loss/test_n,
                    'test_acc':test_acc/test_n,
                    'epoch': epoch,
                }
            model_dir = args.model_dir
            if args.model_dir_ft:
                model_dir = args.model_dir_ft

            if not args.ft_fname:
                torch.save(save_dict, os.path.join(model_dir, f'model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(model_dir, f'opt_{epoch}.pth'))
                if args.lr_scheduler == 'cosine':
                    torch.save(scheduler.state_dict(), os.path.join(model_dir, f'sch_{epoch}.pth'))
            else:
                torch.save(save_dict, os.path.join(model_dir, f'model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(model_dir, f'opt_{epoch}.pth'))
                if args.lr_scheduler == 'cosine':
                    torch.save(scheduler.state_dict(), os.path.join(model_dir, f'sch_{epoch}.pth'))
        model_dir = args.model_dir
        if args.model_dir_ft:
            model_dir = args.model_dir_ft
        # save best
        if test_robust_acc/test_n > best_test_robust_acc:
            if not args.ft_fname:
                torch.save({
                    'state_dict':model.state_dict(),
                    'test_robust_acc':test_robust_acc/test_n,
                    'test_robust_loss':test_robust_loss/test_n,
                    'test_loss':test_loss/test_n,
                    'test_acc':test_acc/test_n,
                    'epoch': epoch,
                    }, os.path.join(model_dir, f'model_best.pth'))
            else:
                torch.save({
                    'state_dict':model.state_dict(),
                    'test_robust_acc':test_robust_acc/test_n,
                    'test_robust_loss':test_robust_loss/test_n,
                    'test_loss':test_loss/test_n,
                    'test_acc':test_acc/test_n,
                    'epoch': epoch,
                    }, os.path.join(model_dir, f'model_best.pth'))
            best_test_robust_acc = test_robust_acc/test_n
    train_total_time = time.time() - train_start_time
    logger.info(f'total training time: {train_total_time}')
    

if __name__== '__main__':
    main()
