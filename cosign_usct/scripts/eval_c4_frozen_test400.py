#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math
from pathlib import Path
import numpy as np
import torch as th
from cc.script_util import create_model_and_diffusion

CENTER=1502.5; SCALE=102.5
SPEED_MIN=1400.0; SPEED_MAX=1605.0; DATA_RANGE=SPEED_MAX-SPEED_MIN
SEED=20260927; SIGMA=11.71851402
HP_ALPHA=0.20
AGR_MSE_TAU=3.0; AGR_MSE_ALPHA=0.175
AGR_STABLE_TAU=3.0; AGR_STABLE_ALPHA=0.125

def norm_to_speed(x): return x*SCALE+CENTER
def speed_to_norm(x): return (x-CENTER)/SCALE

def metric_dict(pred, gt):
    from skimage.metrics import structural_similarity
    err=pred-gt; mse=float(np.mean(err**2)); mae=float(np.mean(np.abs(err))); rmse=float(np.sqrt(mse))
    psnr=float('inf') if mse<=0 else float(10.0*math.log10((DATA_RANGE**2)/mse))
    ssim=float(structural_similarity(pred, gt, data_range=DATA_RANGE))
    return dict(mse=mse,mae=mae,rmse=rmse,psnr=psnr,ssim=ssim)

def create_models(backbone_path, control_path, device):
    control_net, controlled_unet, diffusion = create_model_and_diffusion(
        image_size=256,class_cond=False,learn_sigma=False,num_channels=256,num_res_blocks=2,
        channel_mult='',num_heads=4,num_head_channels=64,num_heads_upsample=-1,
        attention_resolutions='32,16,8',dropout=0.0,use_checkpoint=False,
        use_scale_shift_norm=False,resblock_updown=True,use_fp16=True,use_new_attention_order=False,
        weight_schedule='uniform',sigma_min=0.002,sigma_max=80.0,loss_norm='l2',loss_type='recon',
        distillation=True,control=True,in_channels=1)
    controlled_unet.load_state_dict(th.load(backbone_path,map_location='cpu'),strict=True)
    control_net.load_state_dict(th.load(control_path,map_location='cpu'),strict=True)
    controlled_unet.to(device); control_net.to(device)
    controlled_unet.convert_to_fp16(); control_net.convert_to_fp16()
    controlled_unet.eval(); control_net.eval()
    return control_net, controlled_unet, diffusion

@th.no_grad()
def reconstruct_batch(hint_norm, eps, control_net, controlled_unet, diffusion, device):
    h=th.from_numpy(hint_norm[:,None].astype(np.float32)).to(device)
    e=th.from_numpy(eps[:,None].astype(np.float32)).to(device)
    x_t=h+SIGMA*e
    sigma=th.full((h.shape[0],),SIGMA,dtype=th.float32,device=device)
    _,pred=diffusion.recon(controlled_unet,control_net,x_t,h,sigma)
    return pred.clamp(-1,1).float().cpu().numpy()[:,0].astype(np.float32)

def summarize(rows, method):
    sub=[r for r in rows if r['method']==method]; out={'method':method,'n':len(sub)}
    for k in ['mse','mae','rmse','psnr','ssim']:
        v=np.asarray([r[k] for r in sub],dtype=np.float64)
        out[k+'_mean']=float(v.mean()); out[k+'_std']=float(v.std()); out[k+'_median']=float(np.median(v))
    return out

def write_csv(path, rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backbone',required=True); ap.add_argument('--control',required=True)
    ap.add_argument('--gt',required=True); ap.add_argument('--hint',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--batch_size',type=int,default=1); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    gt_norm=np.load(args.gt).astype(np.float32); hint_norm=np.load(args.hint).astype(np.float32)
    if gt_norm.shape!=hint_norm.shape: raise RuntimeError(f'GT/hint mismatch: {gt_norm.shape} vs {hint_norm.shape}')
    if gt_norm.shape!=(400,256,256): raise RuntimeError(f'Expected TEST400 (400,256,256), got {gt_norm.shape}')
    if not np.isfinite(gt_norm).all() or not np.isfinite(hint_norm).all(): raise RuntimeError('NaN/Inf in input arrays')
    device=th.device(args.device)
    print('='*100); print('FROZEN C4 / AGR FINAL TEST400'); print('='*100)
    print('seed=',SEED,'sigma=',SIGMA,'HP alpha=',HP_ALPHA)
    print('AGR-MSE hard tau/alpha=',AGR_MSE_TAU,AGR_MSE_ALPHA)
    print('AGR-Stable soft tau/alpha=',AGR_STABLE_TAU,AGR_STABLE_ALPHA)
    print('TEST TUNING = FORBIDDEN')
    rng=np.random.default_rng(SEED); eps=rng.standard_normal(gt_norm.shape).astype(np.float32)
    np.save(out/'fixed_eps_seed20260927.npy',eps)
    control_net, controlled_unet, diffusion=create_models(args.backbone,args.control,device)
    preds=[]; bs=args.batch_size
    for start in range(0,400,bs):
        stop=min(start+bs,400)
        p=reconstruct_batch(hint_norm[start:stop],eps[start:stop],control_net,controlled_unet,diffusion,device)
        preds.append(p); print(f'C4 {start:03d}:{stop:03d} range=({p.min():.6f},{p.max():.6f})')
    raw_pred_norm=np.concatenate(preds,axis=0).astype(np.float32)
    np.save(out/'c4_raw_pred_norm.npy',raw_pred_norm)
    gt_speed=norm_to_speed(gt_norm).astype(np.float32); hint_speed=norm_to_speed(hint_norm).astype(np.float32)
    raw_speed=norm_to_speed(raw_pred_norm).astype(np.float32); delta=raw_speed-hint_speed
    hp_speed=(hint_speed+HP_ALPHA*delta).astype(np.float32)
    agr_mse_speed=(hint_speed+AGR_MSE_ALPHA*delta*(np.abs(delta)>=AGR_MSE_TAU)).astype(np.float32)
    soft=np.sign(delta)*np.maximum(np.abs(delta)-AGR_STABLE_TAU,0.0)
    agr_stable_speed=(hint_speed+AGR_STABLE_ALPHA*soft).astype(np.float32)
    arrays={'inversionnet':hint_speed,'c4_raw':raw_speed,'hp_c4':hp_speed,'agr_mse':agr_mse_speed,'agr_stable':agr_stable_speed}
    for name,arr in arrays.items():
        np.save(out/f'{name}_pred_speed.npy',arr.astype(np.float32)); np.save(out/f'{name}_pred_norm.npy',speed_to_norm(arr).astype(np.float32))
    methods=list(arrays.keys()); rows=[]
    for i in range(400):
        inv=metric_dict(hint_speed[i],gt_speed[i])
        for method in methods:
            m=metric_dict(arrays[method][i],gt_speed[i])
            rows.append({'sample':i+1,'method':method,**m,
                'mse_win_vs_inv':int(m['mse']<inv['mse']),'mae_win_vs_inv':int(m['mae']<inv['mae']),
                'psnr_win_vs_inv':int(m['psnr']>inv['psnr']),'ssim_win_vs_inv':int(m['ssim']>inv['ssim'])})
    write_csv(out/'test400_per_sample_metrics.csv',rows)
    summaries=[]
    for method in methods:
        s=summarize(rows,method); sub=[r for r in rows if r['method']==method]
        for k in ['mse','mae','psnr','ssim']:
            s[k+'_wins_vs_inv']=int(sum(r[k+'_win_vs_inv'] for r in sub))
        summaries.append(s)
    write_csv(out/'test400_summary.csv',summaries)
    payload={'protocol':{'split':'TEST400','hyperparameter_search_on_test':False,'seed':SEED,'sigma':SIGMA,
        'c4_control_checkpoint':str(args.control),'c3_backbone_checkpoint':str(args.backbone),
        'hp_c4':{'alpha':HP_ALPHA},'agr_mse':{'gate':'hard','tau_mps':AGR_MSE_TAU,'alpha':AGR_MSE_ALPHA},
        'agr_stable':{'gate':'soft_shrinkage','tau_mps':AGR_STABLE_TAU,'alpha':AGR_STABLE_ALPHA}},
        'summary':{s['method']:s for s in summaries}}
    (out/'test400_summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n'+'='*125); print('FINAL TEST400 SUMMARY'); print('='*125)
    print(f"{'method':>16s} {'MSE':>12s} {'MAE':>10s} {'RMSE':>10s} {'PSNR':>10s} {'SSIM':>10s} {'MSEwin':>10s} {'MAEwin':>10s}")
    for s in summaries:
        print(f"{s['method']:>16s} {s['mse_mean']:12.6f} {s['mae_mean']:10.6f} {s['rmse_mean']:10.6f} {s['psnr_mean']:10.4f} {s['ssim_mean']:10.6f} {s['mse_wins_vs_inv']:4d}/400 {s['mae_wins_vs_inv']:4d}/400")
    print('\n[PASS] Frozen TEST400 image-domain evaluation completed.'); print('output =',out)

if __name__=='__main__': main()
