import torch
from torch.utils.data import DataLoader

from .dataloader import VehicleDataset, PKSampler, train_transform, train_transform_alb
from .modelloader import DinoV2Embedder, SubCenterArcFace, triplet_loss_with_stats

def train(
    data_root="dataset/",
    backbone_name="facebook/dinov2-with-registers-base",
    embed_dim=256,
    epochs=30,
    P=8, K=4,
    lr=1e-5, head_lr=1e-3,
    freeze_backbone_layers=6,
    device="cuda" if torch.cuda.is_available() else "cpu",
    twin_groups=None,
):
    ds = VehicleDataset(data_root, transform=train_transform)
    sampler = PKSampler(ds.class_to_indices, P=P, K=K, twin_groups=twin_groups)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
 
    model = DinoV2Embedder(backbone_name, embed_dim, freeze_backbone_layers).to(device)
    criterion = SubCenterArcFace(embed_dim, num_classes=len(ds.classes)).to(device)
 
    # separate LR: backbone fine-tunes slowly, new head learns fast
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr},
        {"params": model.proj.parameters(), "lr": head_lr},
        {"params": criterion.parameters(), "lr": head_lr},
    ], weight_decay=0.05)
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    scaler = torch.cuda.amp.GradScaler()
 
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
 
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                emb = model(imgs)
                loss = criterion(emb, labels)
 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
 
            running_loss += loss.item()
 
        print(f"epoch {epoch+1}/{epochs}  loss={running_loss/len(loader):.4f}")
 
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"dinov2_embedder_epoch{epoch+1}.pt")
 
    return model, ds.classes
 
def train_triplet(
    data_root,
    backbone_name="facebook/dinov2-with-registers-base",
    embed_dim=256,
    epochs=20,
    P=16, K=2,
    lr=1e-5, head_lr=1e-3,
    freeze_backbone_layers=8,
    margin=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    ds = VehicleDataset(data_root, transform=train_transform)
    sampler = PKSampler(ds.class_to_indices, P=P, K=K)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
 
    model = DinoV2Embedder(backbone_name, embed_dim, freeze_backbone_layers).to(device)
 
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr},
        {"params": model.proj.parameters(), "lr": head_lr},
    ], weight_decay=0.05)
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    scaler = torch.cuda.amp.GradScaler()
 
    model.train()
    for epoch in range(epochs):
        running_loss, running_active = 0.0, 0.0
 
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
 
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                emb = model(imgs)
                loss, frac_active = triplet_loss_with_stats(emb, labels, margin)
 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
 
            running_loss += loss.item()
            running_active += frac_active
 
        avg_loss = running_loss / len(loader)
        avg_active = running_active / len(loader)
        print(f"epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  active_triplets={avg_active:.1%}")
 
        # Diagnostic worth actually watching:
        # - active_triplets collapsing toward 0 over many epochs = loss has gone
        #   slack (margin already satisfied everywhere) -- either raise the
        #   margin, or this is genuine convergence.
        # - active_triplets staying near 100% for many epochs with loss not
        #   dropping = model isn't learning; check LR, check that positive
        #   pairs are actually being formed (group_to_indices not degenerate).
 
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"dinov2_triplet_epoch{epoch+1}.pt")
 
    return model

if __name__ == "__main__":
    # example: declare known badge-twin groups by class index after ds is built,
    # or pass class NAMES and resolve to indices before calling train().
    # twin_groups = [[class_to_idx["toyota_avanza_g3"], class_to_idx["daihatsu_xenia_g3"]], ...]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = r"C:\Users\AQI\JOB\pakuwon\image_raw\car_brands"

    dataset = VehicleDataset(dataset_path, transform=train_transform_alb)
    sampler = PKSampler(dataset.class_to_indices, P=4, K=4)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)

    model = DinoV2Embedder(backbone_name="facebook/dinov2-with-registers-base", 
                           embed_dim, 
                           freeze_backbone_layers=6).to(device)

    lr = 1e-5
    head_lr = 1e-3
    optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": lr},
            {"params": model.proj.parameters(), "lr": head_lr},
        ], weight_decay=0.05)



    model, classes = train(
        data_root="dataset/",
        backbone_name="facebook/dinov2-with-registers-base",
        freeze_backbone_layers=6,   # unfreeze more (e.g. 0-2) once loss plateaus
    )