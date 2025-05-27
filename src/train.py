import torch
import tqdm

def train(model,
          optimizer,
          data_handler,
          epochs,
          eval_interval,
          weights_save_path,
          checkpoint_path,
          checkpoint_interval,
          from_checkpoint=False
          ):

    train_losses = []
    val_losses = []

    start_epoch = 0
    if from_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint["epoch"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]

        print(f"training from the checkpoint {start_epoch} epoch")

    model.to(model.device)

    for epoch in range(start_epoch+1, epochs+1):
        model.train()
        pbar = tqdm.tqdm(data_handler.train_loader)

        train_loss = 0

        for i, (x, y) in enumerate(pbar):
            x = x.to(model.device)
            y = y.to(model.device)

            optimizer.zero_grad()

            loss = model(x, y)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}")

            train_loss += loss.item()

        if epoch == 1 or epoch % eval_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for (x, y) in data_handler.val_loader:
                    x = x.to(model.device)
                    y = y.to(model.device)
                    loss = model(x, y)
                    val_loss += loss.item()

            print(f"Epoch {epoch}: train loss = {train_loss / len(data_handler.train_loader):.4f}, val loss = {val_loss / len(data_handler.val_loader):.4f}")

            val_losses.append(val_loss / len(data_handler.val_loader))
            train_losses.append(train_loss / len(data_handler.train_loader))

        if epoch % checkpoint_interval == 0:

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)

            torch.save(model.state_dict(), weights_save_path)

    return train_losses, val_losses

