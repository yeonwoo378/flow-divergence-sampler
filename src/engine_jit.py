import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2
import json
import time

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity

from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id
from torch_fidelity.metric_prc import prc_features_to_metric
from util.prc_eval import calculate_precision_recall_with_virtual_imagenet_npz

import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    if args.gen_path:
        print('eval only. no gen.')
        save_folder = args.gen_path
    else:

        # Construct the folder name for saving generated images.
        save_folder = os.path.join(
            args.output_dir,
            "{}-steps{}-{}-cfg{}".format(
                model_without_ddp.method, model_without_ddp.steps, args.model.replace('/', '-'), args.cfg
            )
        )
        print("Save to:", save_folder)
        if misc.get_rank() == 0 and not os.path.exists(save_folder):
            os.makedirs(save_folder)
            image_dir = os.path.join(save_folder, 'images')
            os.makedirs(image_dir)
        
        # save configuration here
        if misc.get_rank() == 0:
            config_path = os.path.join(save_folder, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(vars(args), f, indent=4)

        # switch to ema params, hard-coded to be the first one
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = model_without_ddp.ema_params1[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

        # ensure that the number of images per class is equal.
        class_num = args.class_num
        if args.class_idx == -1:
            assert args.num_images % class_num == 0, "Number of images per class must be the same"
            class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
            class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
        else:
            class_label_gen_world = np.full(args.num_images, args.class_idx, dtype=np.int64)

        assert args.num_images % class_num == 0, "Number of images per class must be the same"
        # class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
        # class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
        times = []
        for i in range(num_steps):
            print("Generation step {}/{}".format(i, num_steps))

            start_idx = world_size * batch_size * i + local_rank * batch_size
            end_idx = start_idx + batch_size
            labels_gen = class_label_gen_world[start_idx:end_idx]
            labels_gen = torch.Tensor(labels_gen).long().cuda()


            start_time = time.time()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sampled_images = model_without_ddp.generate(labels_gen, args)
            end_time = time.time()
            elpased_time = end_time - start_time
            
            if i!=0 and args.check_time:
                print('elpased time: ', elpased_time)
                times.append(elpased_time)
            if args.check_time and i==64:
                print('avg gen time', sum(times)/len(times))
                break

            torch.distributed.barrier()

            # denormalize images
            sampled_images = (sampled_images + 1) / 2
            sampled_images = sampled_images.detach().cpu()

            # distributed save images
            for b_id in range(sampled_images.size(0)):
                img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
                if img_id >= args.num_images:
                    break
                gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
                gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_folder, 'images/{}.png'.format(str(img_id).zfill(5))), gen_img)

        torch.distributed.barrier()

        # back to no ema
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            # input1=f'{save_folder}/images',
            input1=f'{save_folder}',
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            # prc=True,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        prc_statistics_file = 'fid_stats/imagenet_val-inception-v3-compat-features-2048.pt'
        # os.symlink(prc_statistics_file, 
        #            os.path.join(temp_cache_dir, f'imagenet_val-inception-v3-compat-features-2048.pt'))
        prc_feature_path = "fid_stats/imagenet_val-inception-v3-compat-features-2048.pt"


        # metrics_dict = compute_prc_from_feature_file(
        #     gen_images_dir=f'{save_folder}/images',
        #     real_feature_pt=prc_feature_path,
        #     batch_size=64,
        #     prc_neighborhood=3,
        #     prc_batch_size=10000,
        #     save_cpu_ram=True,
        #     use_cuda=True,
        # )
        prc_dict = calculate_precision_recall_with_virtual_imagenet_npz(
            # gen_images_dir=f"{save_folder}/images",
            gen_images_dir=f"{save_folder}",
            virtual_npz_path= "VIRTUAL_imagenet256_labeled.npz",

            prc_num_gen=None,
            prc_num_ref=None,
            # seed=getattr(args, "seed", 0),

            batch_size=getattr(args, "prc_feat_batch", 64),
            prc_batch_size=10000,
            prc_neighborhood=getattr(args, "prc_neighborhood", 3),

            save_cpu_ram=True,
            cuda=True,
            verbose=False,

            cache=True,
            # cache_root=getattr(args, "torch_fidelity_cache_root", None),
            # npz_cache_dir=getattr(args, "npz_cache_dir", None),  # e.g. f"{args.output_dir}/npz_cache"
        )
        metrics_dict.update(prc_dict)
    
        pre = metrics_dict['precision']
        rec = metrics_dict['recall']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}, pre: {:.4f}, recall: {:.4f}".format(fid, inception_score, pre, rec))
        
        # save the result 
        json_path = os.path.join(save_folder, 'stats.json')
        stats = {
            "FID": fid,
            "IS": inception_score,
            "Precision": pre, 
            "Recall": rec
        }

        with open(json_path, 'w') as f:
            json.dump(stats, f)

        # shutil.rmtree(save_folder) # remove all generated samples

    torch.distributed.barrier()

    return save_folder


def compute_prc_from_feature_file(
    gen_images_dir: str,
    real_feature_pt: str,
    *,
    batch_size: int = 64,
    prc_neighborhood: int = 3,
    prc_batch_size: int = 10000,
    save_cpu_ram: bool = True,
    use_cuda: bool = True,
) -> dict:
    """
    - gen_images_dir: 생성 이미지 폴더 (input1, generated)
    - real_feature_pt: real(ImageNet val) 특징이 저장된 pt (input2에 해당)
    """
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # 1) real features 로드 (input2=real)
    real_feats = _load_cached_feature_pt(real_feature_pt).to(device)

    # 2) generated images -> inception-v3-compat 2048 features 추출 (input1=generated)
    #    (PRC features 파일명이 inception-v3-compat / 2048 이므로 동일 extractor/layer로 맞춤)
    feat_extractor = create_feature_extractor(
        "inception-v3-compat", ["2048"],
        cuda=(device.type == "cuda"),
        verbose=False,
    )

    # input1 슬롯에서 feature 추출 (캐시 없이 바로 추출)
    fake_featuresdict = extract_featuresdict_from_input_id(
        1,
        feat_extractor,
        input1=gen_images_dir,
        cuda=(device.type == "cuda"),
        batch_size=batch_size,
        verbose=False,
    )
    fake_feats = fake_featuresdict["2048"].to(device)

    # 3) PRC 계산
    # torch-fidelity 문서 기준: PRC는 input1=generated, input2=real 로 해석 :contentReference[oaicite:3]{index=3}
    prc_dict = prc_features_to_metric(
        fake_feats,           # features_1 = generated (input1)
        real_feats,           # features_2 = real      (input2)
        prc_neighborhood=prc_neighborhood,
        prc_batch_size=prc_batch_size,
        save_cpu_ram=save_cpu_ram,
        verbose=False,
    )
    return prc_dict

def _load_cached_feature_pt(path: str, expected_dim: int = 2048) -> torch.Tensor:
    """
    torch-fidelity 캐시로 생성된 *.pt feature 파일을 robust하게 로드.
    보통 torch.Tensor 하나가 저장되어 있습니다.
    """
    obj = torch.load(path, map_location="cpu")

    # 가장 흔한 케이스: 바로 Tensor
    if isinstance(obj, torch.Tensor):
        feats = obj
    # 혹시 dict 형태로 저장된 경우를 대비
    elif isinstance(obj, dict):
        # 키가 '2048' 같은 형태거나, features 같은 키일 수 있음
        if "2048" in obj and isinstance(obj["2048"], torch.Tensor):
            feats = obj["2048"]
        elif "features" in obj and isinstance(obj["features"], torch.Tensor):
            feats = obj["features"]
        else:
            # dict 안에서 첫 Tensor를 찾아 사용
            feats = None
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    feats = v
                    break
            if feats is None:
                raise ValueError(f"Cannot find Tensor in dict loaded from: {path}")
    # (드물지만) list/tuple로 저장된 경우
    elif isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
        feats = obj[0]
    else:
        raise TypeError(f"Unsupported content in feature file {path}: {type(obj)}")

    feats = feats.detach().to(torch.float32)

    # 혹시 [N, 2048, 1, 1] 같은 형태면 [N, 2048]로 펴기
    if feats.dim() != 2:
        feats = feats.reshape(feats.shape[0], -1)

    if feats.shape[1] != expected_dim:
        raise ValueError(f"Feature dim mismatch: got {feats.shape}, expected (*, {expected_dim}) from {path}")

    return feats
