#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast Attention-ASPP U-Net
✓ AMP  |  ✓ EarlyStopping  |  ✓ DataLoader pin_memory/num_workers
"""

import argparse, os
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
import torch.nn.functional as F


# ---------------- Model Blocks ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x): return self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_g, in_x, inter):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(in_g, inter, 1, bias=False),
                                 nn.BatchNorm2d(inter))
        self.W_x = nn.Sequential(nn.Conv2d(in_x, inter, 1, bias=False),
                                 nn.BatchNorm2d(inter))
        self.psi = nn.Sequential(nn.Conv2d(inter, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # ➕ 对 g1 尺寸进行插值，确保和 x1 匹配
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # 同样处理 attention mask 与 x 尺寸不一致的问题
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=False)

        return x * psi


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1,6,12,18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(True)) for r in rates])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch*len(rates), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self,x):
        return self.project(torch.cat([c(x) for c in self.convs],1))

class AttentionASPPUNet(nn.Module):
    def __init__(self,in_ch=1,num_classes=1,base=16):
        super().__init__()
        b=base
        self.inc=DoubleConv(in_ch,b)
        self.d1=nn.Sequential(nn.MaxPool2d(2),DoubleConv(b,b*2))
        self.d2=nn.Sequential(nn.MaxPool2d(2),DoubleConv(b*2,b*4))
        self.d3=nn.Sequential(nn.MaxPool2d(2),DoubleConv(b*4,b*8))
        self.aspp=ASPP(b*8,b*16)
        self.u3=nn.ConvTranspose2d(b*16,b*8,2,2)
        self.a3=AttentionBlock(b*8,b*4,b*4); self.c3=DoubleConv(b*8+b*4,b*8)
        self.u2=nn.ConvTranspose2d(b*8,b*4,2,2)
        self.a2=AttentionBlock(b*4,b*2,b*2); self.c2=DoubleConv(b*4+b*2,b*4)
        self.u1=nn.ConvTranspose2d(b*4,b*2,2,2)
        self.a1=AttentionBlock(b*2,b,b);     self.c1=DoubleConv(b*2+b,b*2)
        self.outc=nn.Conv2d(b*2,num_classes,1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)

        center = self.aspp(x4)

        # ------- Decoder stage 3 -------
        d3 = self.u3(center)
        if d3.shape[2:] != x3.shape[2:]:                      # ★ 对齐
            d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, self.a3(d3, x3)], dim=1)
        d3 = self.c3(d3)

        # ------- Decoder stage 2 -------
        d2 = self.u2(d3)
        if d2.shape[2:] != x2.shape[2:]:                      # ★ 对齐
            d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, self.a2(d2, x2)], dim=1)
        d2 = self.c2(d2)

        # ------- Decoder stage 1 -------
        d1 = self.u1(d2)
        if d1.shape[2:] != x1.shape[2:]:                      # ★ 对齐
            d1 = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, self.a1(d1, x1)], dim=1)
        d1 = self.c1(d1)

        return self.outc(d1)

# ---------------- Dataset ----------------
class FetalAbdomenDataset(Dataset):
    def __init__(self, root, size=224, keep_empty=False):
        self.root = Path(root);
        self.size = size
        imgs = sorted((self.root / 'images').glob('*.png'))
        if keep_empty:
            self.imgs = imgs
        else:  # 过滤掉掩码全 0 的 slice
            self.imgs = [p for p in imgs
                         if (cv2.imread(str(self.root / 'masks' / p.name), 0) > 0).any()]
        self.masks = [self.root / 'masks' / p.name for p in self.imgs]
    def __len__(self): return len(self.imgs)
    def __getitem__(self,idx):
        img  = cv2.imread(str(self.imgs[idx]),0)
        mask = cv2.imread(str(self.masks[idx]),0)
        img  = cv2.resize(img,(self.size,self.size),cv2.INTER_AREA)
        mask = cv2.resize(mask,(self.size,self.size),cv2.INTER_NEAREST)
        img  = torch.from_numpy(img.astype(np.float32)/255.).unsqueeze(0)
        mask = torch.from_numpy((mask>0).astype(np.float32)).unsqueeze(0)
        return img,mask

# ---------------- Metrics ----------------
@torch.no_grad()
def dice(pred,tgt,eps=1e-6):
    pred=(pred>0.5).float(); inter=(pred*tgt).sum()
    return (2*inter+eps)/(pred.sum()+tgt.sum()+eps)
@torch.no_grad()
def iou(pred,tgt,eps=1e-6):
    pred=(pred>0.5).float(); inter=(pred*tgt).sum(); union=pred.sum()+tgt.sum()-inter
    return (inter+eps)/(union+eps)
@torch.no_grad()
def hd(pred,tgt):
    p=(pred.squeeze().cpu().numpy()>0.5).astype(np.uint8)
    t=tgt.squeeze().cpu().numpy().astype(np.uint8)
    if p.sum()==0 or t.sum()==0: return np.nan
    pp=np.column_stack(np.where(p)); tt=np.column_stack(np.where(t))
    return max(directed_hausdorff(pp,tt)[0], directed_hausdorff(tt,pp)[0])

# ---------------- Epoch runner ----------------
def run_epoch(model,loader,opt,crit,dev,train=True,amp=False,scaler=None):
    model.train() if train else model.eval()
    loop=tqdm(loader,leave=False,desc='Train' if train else 'Val')
    losses=[]; dices=[]; ious=[]; hds=[]
    for img,mask in loop:
        img,mask=img.to(dev),mask.to(dev)
        with torch.set_grad_enabled(train):
            with autocast(enabled=amp):
                out=model(img); loss=crit(torch.sigmoid(out),mask)
        if train:
            scaler.scale(loss).backward() if amp else loss.backward()
            scaler.step(opt) if amp else opt.step()
            scaler.update() if amp else None; opt.zero_grad()
        else:
            out=torch.sigmoid(out)
            if mask.sum() == 0 or out.sum() == 0:
                print(
                    f"[DEBUG] Empty Pred or GT - Pred sum: {out.sum().item():.2f}, Target sum: {mask.sum().item():.2f}")
            dices.append(dice(out,mask).item()); ious.append(iou(out,mask).item()); hds.append(hd(out,mask))
        losses.append(loss.item())
    if train: return np.mean(losses)
    return np.mean(dices),np.mean(ious),np.nanmean(hds)
# ---------------- Main ----------------
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train_dir',required=True); p.add_argument('--val_dir',required=True)
    p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--size',type=int,default=224)
    p.add_argument('--base_ch',type=int,default=16); p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--amp',action='store_true'); p.add_argument('--patience',type=int,default=15)
    p.add_argument('--out_dir',default='./checkpoints')
    args=p.parse_args()

    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds=FetalAbdomenDataset(args.train_dir,args.size)
    val_ds  =FetalAbdomenDataset(args.val_dir ,args.size)
    tl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                  num_workers=args.num_workers,pin_memory=True)
    vl=DataLoader(val_ds,batch_size=1,shuffle=False,
                  num_workers=2,pin_memory=True)

    model=AttentionASPPUNet(base=args.base_ch).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    crit=nn.BCEWithLogitsLoss()
    scaler=GradScaler(enabled=args.amp)

    best=0; patience=args.patience; wait=0
    os.makedirs(args.out_dir,exist_ok=True)

    for ep in range(1,args.epochs+1):
        tr_loss=run_epoch(model,tl,opt,crit,dev,True,args.amp,scaler)
        dice,IOU,HD=run_epoch(model,vl,opt,crit,dev,False,args.amp)
        print(f'Epoch {ep:03d} | Loss {tr_loss:.4f} | Dice {dice:.4f} | IoU {IOU:.4f} | HD {HD:.2f}')
        if dice>best:
            best=dice; wait=0
            torch.save(model.state_dict(),Path(args.out_dir)/'best_model.pth')
            print('✔️  Saved new best model')
        else:
            wait+=1
            if wait>=patience:
                print('⏹️  Early stopping'); break

if __name__=='__main__':
    main()
