from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from train_loop_e import TrainLoop
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data
import model
from PIL import ImageFilter

# Training settings
parser = argparse.ArgumentParser(description='Inference model for compressed sensing - Train Generator on target domain')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--beta1', type=float, default=0.5, metavar='lambda', help='Adam beta param (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='lambda', help='Adam beta param (default: 0.999)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--gen-path', type=str, default=None, metavar='Path', help='Path for generative model')
parser.add_argument('--data-path', type=str, default='./celebA', metavar='Path', help='Path to data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=3, metavar='N', help='how many epochs to wait before logging training status. Default is 3')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

mnist = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.Resize((64, 64)), lambda x: x.filter(ImageFilter.GaussianBlur(radius=6)), transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=args.batch_size, shuffle=True)

generator = model.Generator(100, [1024, 512, 256, 128], 1).eval()
encoder = model.Encoder(1, [128, 256, 512, 1024], 100).train()

ckpt_gen = torch.load(args.gen_path, map_location = lambda storage, loc: storage)
generator.load_state_dict(ckpt_gen['model_state'])

if args.cuda:
	generator = generator.cuda()
	encoder = encoder.cuda()

optimizer = optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

trainer = TrainLoop(generator, encoder, optimizer, train_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
