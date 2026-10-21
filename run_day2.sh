#!/bin/sh
run() { name=$1; shift; python train.py "$@" --save-model checkpoints/$name.pth --plot-file plots/loss_$name.png 2>&1 | tee logs/$name.txt; }

# Tiny ImageNet (required Step 4 run first), real val split
run tiny_alexnet_bn_aug --model alexnet --norm bn --dataset tiny_imagenet --epochs 10 --seed 42
run tiny_alexnet_bn --model alexnet --norm bn --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment

# Multi-seed norm study (seed 42 already done)
for s in 123 2026; do
  run alexnet_none_s$s --model alexnet --norm none --dataset cifar10 --epochs 10 --seed $s --no-augment
  run alexnet_bn_s$s --model alexnet --norm bn --dataset cifar10 --epochs 10 --seed $s --no-augment
done

# Extended ablations on AlexNet+BN (baseline lr=1e-3, bs=128 comes from the multi-seed runs)
for s in 42 123 2026; do
  run abl_lr1e-4_s$s --model alexnet --norm bn --dataset cifar10 --epochs 10 --seed $s --no-augment --lr 1e-4
  run abl_lr3e-3_s$s --model alexnet --norm bn --dataset cifar10 --epochs 10 --seed $s --no-augment --lr 3e-3
  run abl_bs64_s$s --model alexnet --norm bn --dataset cifar10 --epochs 10 --seed $s --no-augment --batch-size 64
  run abl_bs256_s$s --model alexnet --norm bn --dataset cifar10 --epochs 10 --seed $s --no-augment --batch-size 256
done

# Rest of the Tiny ImageNet column
run tiny_lenet --model lenet --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
run tiny_lenet_modern --model lenet_modern --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
run tiny_alexnet_none --model alexnet --norm none --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
run tiny_zoo_alexnet --model zoo_alexnet --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
# Collapse check: does no-norm AlexNet train on Tiny at a lower learning rate?
run tiny_alexnet_none_lr1e-4 --model alexnet --norm none --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment --lr 1e-4

# Slow LRN runs last
for s in 123 2026; do
  run alexnet_lrn_s$s --model alexnet --norm lrn --dataset cifar10 --epochs 10 --seed $s --no-augment
done
run tiny_alexnet_lrn --model alexnet --norm lrn --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
