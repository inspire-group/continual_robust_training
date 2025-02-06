import argparse
import torch
import numpy as np
import models
import attacks
import os
import data
from timm.models import create_model
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'imagenette', "imagenet100"])
    parser.add_argument('--metrics', nargs='+')
    parser.add_argument('--arch', type=str, help='model architecture', default='resnet18')
    parser.add_argument('--normalize', action='store_true', help='whether data is normalized before passing into model or not')
    parser.add_argument('--num_iters', type=int, default=10)
    parser.add_argument('--attacks_in', type=str, nargs='+', default=['NoAttack'])
    parser.add_argument('--eps_in', type=float, nargs='+', default=[0])
    parser.add_argument('--attacks_out', type=str, nargs='+', default=[])
    parser.add_argument('--eps_out', type=float, nargs='+', default=[])
    parser.add_argument('--model_dir', type=str, help='path to directory of ckpts', default='adv_train_resnet')
    parser.add_argument('--model_subdir', type=str, help='ckpt subdir to evaluate')
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument('--log_dir', type=str, default="eval_logs_best")
    parser.add_argument('--run_full', action='store_true', help='run full attack for comparison to baselines')
    args = parser.parse_args()

    if args.run_full:
        args.num_iters = 20 # do nonLp attack evals with 20 iterations instead of 10

    args.log_dir = os.path.join(args.log_dir, os.path.basename(args.model_dir))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # load data
    _, test_loader, mean, std, num_classes = data.__dict__[args.dataset](args.data_dir, args.batch_size, False, '', 0, contrastive=False)

    if args.arch in models.__dict__ or 'convnext' in args.arch:
        if 'convnext' in args.arch:
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
    model.eval() 
    test_atks_in = []
    for i, attack in enumerate(args.attacks_in):
        print('adding attacks_in')
        if attack == 'NoAttack':
            test_atk = attacks.__dict__[attack]()
        else:
            eps = args.eps_in[i]
            try:
                if 'Auto' in attack:
                    if 'cifar' in args.dataset:
                        test_atk = attacks.__dict__[attack](model, dataset_name='cifar',bound=eps, num_iterations=args.num_iters, full=args.run_full)
                    else: # it is an imagenet subset
                        test_atk = attacks.__dict__[attack](model, dataset_name='imagenet',bound=eps, num_iterations=args.num_iters, full=args.run_full)
                else:
                    if 'cifar' in args.dataset:
                        test_atk = attacks.__dict__[attack](model, dataset_name='cifar',bound=eps, num_iterations=args.num_iters)
                    else: # it is an imagenet subset
                        test_atk = attacks.__dict__[attack](model, dataset_name='imagenet',bound=eps, num_iterations=args.num_iters)
            except:
                if 'Auto' in attack:
                    test_atk = attacks.__dict__[attack](model, bound=eps, num_iterations=args.num_iters, full=args.run_full)
                else:
                    test_atk = attacks.__dict__[attack](model, bound=eps, num_iterations=args.num_iters)
        test_atks_in.append(test_atk)

    test_atks_out = []
    for i, attack in enumerate(args.attacks_out):
        print('adding attacks_out')
        if attack == 'NoAttack':
            test_atk = attacks.__dict__[attack]()
        else:
            eps = args.eps_out[i]
        try:
            if 'Auto' in attack:
                if 'cifar' in args.dataset:
                    test_atk = attacks.__dict__[attack](model, dataset_name='cifar',bound=eps, num_iterations=args.num_iters, full=args.run_full)
                else: # it is an imagenet subset
                    test_atk = attacks.__dict__[attack](model, dataset_name='imagenet',bound=eps, num_iterations=args.num_iters, full=args.run_full)
            else:
                if 'cifar' in args.dataset:
                    test_atk = attacks.__dict__[attack](model, dataset_name='cifar',bound=eps, num_iterations=args.num_iters)
                else: # it is an imagenet subset
                    test_atk = attacks.__dict__[attack](model, dataset_name='imagenet',bound=eps, num_iterations=args.num_iters)            
        except:
            if 'Auto' in attack:
                test_atk = attacks.__dict__[attack](model, bound=eps, num_iterations=args.num_iters, full=args.run_full)
            else:
                test_atk = attacks.__dict__[attack](model, bound=eps, num_iterations=args.num_iters)
        test_atks_out.append(test_atk)
    print('beginning eval')
    data_size = len(test_loader.dataset)
    path1 = os.path.join(args.model_dir, args.model_subdir)
    fname = 'model_best.pth'
    path2 = os.path.join(path1, fname)
    load_data = torch.load(path2)
    model.load_state_dict(load_data['state_dict'])
    clean_acc = 0
    indiv_acc = np.zeros(len(test_atks_in) + len(test_atks_out))
    union_in_acc = 0
    union_acc = 0
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        pred = model(x).argmax(1)
        clean_corr = (y == pred).float()
        clean_acc += clean_corr.sum().item()
        union_correct = torch.ones_like(clean_corr) * clean_corr
        if args.attacks_out:
            union_correct2 = torch.ones_like(clean_corr) * clean_corr
        for i, atk in enumerate(test_atks_in):
            pred_atk = model(atk(x, y)).argmax(1)
            atk_corr = (y == pred_atk).float()
            union_correct *= atk_corr
            if args.attacks_out:
                union_correct2 *= atk_corr
            indiv_acc[i] += atk_corr.sum().item()
        for i, atk in enumerate(test_atks_out):
            pred_atk = model(atk(x, y)).argmax(1)
            atk_corr = (y == pred_atk).float()
            union_correct2 *= atk_corr
            indiv_acc[len(test_atks_in) + i] += atk_corr.sum().item()
        union_in_acc += union_correct.sum().item()
        if args.attacks_out:
            union_acc += union_correct2.sum().item()
    clean_acc /= data_size
    indiv_acc /= data_size
    union_in_acc /= data_size
    if args.attacks_out:
        union_acc /= data_size
    
    entry_dict = {}
    entry_dict['attacks_in'] = ", ".join(args.attacks_in)
    entry_dict['eps_in'] = ", ".join([str(eps) for eps in args.eps_in])
    entry_dict['attacks_out'] = ", ".join(args.attacks_out)
    entry_dict['eps_out']= ", ".join([str(eps) for eps in args.eps_out])
    entry_dict['clean'] = clean_acc
    entry_dict['indiv'] = indiv_acc.tolist() 
    if args.attacks_out:
        entry_dict['avg_in'] = indiv_acc[:len(test_atks_in)].mean()
        entry_dict['union_in'] = union_in_acc
    entry_dict['avg'] = indiv_acc.mean()
    if args.attacks_out:
        entry_dict['union'] = union_acc
    else:
        entry_dict['union'] = union_in_acc
    
    with open(f'{args.log_dir}/{args.model_subdir}_last.json', "w") as outfile:
        json.dump(entry_dict, outfile)


if __name__== '__main__':
    main()




