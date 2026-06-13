import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
from cartoongan_models import Generator, Discriminator, VGG19Features
from cartoongan_losses import ContentLoss, EdgePromotingAdversarialLoss

class CartoonDataset(Dataset):
    def __init__(self, photo_dir, cartoon_dir, transform=None, smooth_transform=None):
        self.photos = [os.path.join(photo_dir, f) for f in os.listdir(photo_dir) if f.endswith(('.jpg', '.png'))]
        self.cartoons = [os.path.join(cartoon_dir, f) for f in os.listdir(cartoon_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform
        self.smooth_transform = smooth_transform
        
    def __len__(self):
        return max(len(self.photos), len(self.cartoons))
        
    def __getitem__(self, idx):
        # Unpaired training: randomly sample from both domains
        photo_path = self.photos[torch.randint(0, len(self.photos), (1,)).item()]
        cartoon_path = self.cartoons[torch.randint(0, len(self.cartoons), (1,)).item()]
        
        photo = Image.open(photo_path).convert('RGB')
        cartoon = Image.open(cartoon_path).convert('RGB')
        
        if self.transform:
            photo = self.transform(photo)
            cartoon = self.transform(cartoon)
            
        smoothed = self.smooth_transform(cartoon) if self.smooth_transform else cartoon
            
        return photo, cartoon, smoothed

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)
    D = Discriminator().to(device)
    VGG = VGG19Features().to(device).eval()
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # [-1, 1]
    ])
    smooth_transform = T.GaussianBlur(kernel_size=5, sigma=1.0)
    vgg_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    dataset = CartoonDataset('data/photos', 'data/cartoons', transform, smooth_transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    criterion_content = ContentLoss(VGG)
    criterion_adv = EdgePromotingAdversarialLoss()
    
    INIT_EPOCHS = 10
    TOTAL_EPOCHS = 100
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(TOTAL_EPOCHS):
        is_init_phase = epoch < INIT_EPOCHS
        
        for photo, cartoon, smoothed in tqdm(loader, desc=f"Epoch {epoch+1}"):
            photo, cartoon, smoothed = photo.to(device), cartoon.to(device), smoothed.to(device)
            
            generated = G(photo)
            
            # VGG requires ImageNet normalization
            gen_vgg = vgg_normalize((generated + 1) / 2)
            cart_vgg = vgg_normalize((cartoon + 1) / 2)
            
            # 1. Generator Step
            loss_content = criterion_content(gen_vgg, cart_vgg)
            
            if is_init_phase:
                loss_G = loss_content
            else:
                d_generated = D(generated)
                loss_adv_G = criterion_adv.forward_G(d_generated)
                loss_G = loss_content + loss_adv_G
                
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            
            # 2. Discriminator Step (Only after init phase)
            if not is_init_phase:
                d_real = D(cartoon)
                d_smoothed = D(smoothed)
                d_generated = D(generated.detach())
                
                loss_D = criterion_adv.forward_D(d_real, d_smoothed, d_generated)
                
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
                
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), f'checkpoints/cartoongan_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train()