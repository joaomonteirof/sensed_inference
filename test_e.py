from __future__ import print_function
import argparse
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
import torchvision
from PIL import ImageFilter
import matplotlib.pyplot as plt
import model as model_
import numpy as np
import os


def denorm(unorm):

	norm = (unorm + 1) / 2

	return norm.clamp(0, 1)

def test_model(generator_, encoder_, data, cuda_mode):

	generator_.eval()
	encoder_.eval()

	pool = torch.nn.MaxPool2d(2)

	to_pil = transforms.ToPILImage()
	to_tensor = transforms.ToTensor()

	if cuda_mode:
		data = data.cuda()

	pooled_data = pool(Variable(data))

	out = generator_.forward(encoder_.forward(pooled_data).unsqueeze(-1).unsqueeze(-1))

	for i in range(out.size(0)):
		high_sample, low_sample = denorm(out[i].data), denorm(pooled_data.data[i])
		high_sample, low_sample = to_pil(high_sample.cpu()), to_pil(low_sample.cpu())
		high_sample.save('highres_sample_{}.png'.format(i+1))
		low_sample.save('lowres_sample_{}.png'.format(i+1))

def save_samples(generator, cp_name, cuda_mode, save_dir='./', fig_size=(5, 5)):
	generator.eval()

	n_tests = fig_size[0]*fig_size[1]

	noise = torch.randn(n_tests, 100).view(-1, 100, 1, 1)

	if cuda_mode:
		noise = noise.cuda()

	noise = Variable(noise, volatile=True)
	gen_image = generator(noise)
	#gen_image = denorm(gen_image)
	gen_image = gen_image

	generator.train()

	n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
	n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
	for ax, img in zip(axes.flatten(), gen_image):
		ax.axis('off')
		ax.set_adjustable('box-forced')
		# Scale to 0-255
		img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).squeeze().astype(np.uint8)
		# ax.imshow(img.cpu().data.view(image_size, image_size, 3).numpy(), cmap=None, aspect='equal')
		ax.imshow(img, cmap=None, aspect='equal')
	plt.subplots_adjust(wspace=0, hspace=0)
	title = 'Samples'
	fig.text(0.5, 0.04, title, ha='center')

	# save figure

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_fn = save_dir + 'sensed_inference'+ cp_name + '.png'
	plt.savefig(save_fn)

	plt.close()

def plot_learningcurves(history, *keys):

	for key in keys:
		plt.plot(history[key])
	
	plt.show()


if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs')
	parser.add_argument('--gen-path', type=str, default=None, metavar='Path', help='path to generator')
	parser.add_argument('--enc-path', type=str, default=None, metavar='Path', help='path to encoder')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data .hdf')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to  (default: 64)')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.gen_path is None or args.enc_path is None:
		raise ValueError('There is no generator/encoder to load. Use arg --gen-path and --enc-path to indicate the paths!')

	generator = model_.Generator(100, [1024, 512, 256, 128], 3)
	encoder = model_.Encoder(3, [128, 256, 512, 1024], 100)

	ckpt_gen = torch.load(args.gen_path, map_location = lambda storage, loc: storage)
	generator.load_state_dict(ckpt_gen['model_state'])

	ckpt_enc = torch.load(args.enc_path, map_location = lambda storage, loc: storage)
	encoder.load_state_dict(ckpt_enc['model_state'])

	if args.cuda:
		generator = generator.cuda()
		encoder = encoder.cuda()

	print('Cuda Mode is: {}'.format(args.cuda))

	history = ckpt_enc['history']

	if not args.no_plots:
		plot_learningcurves(history, 'loss')
		plot_learningcurves(history, 'loss_minibatch')

	transform = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	celebA_data = datasets.ImageFolder(args.data_path, transform = transform)
	loader = torch.utils.data.DataLoader(celebA_data, batch_size=args.n_tests, shuffle=True, num_workers=args.workers)

	for data_, _ in loader:
		test_model(generator, encoder, data_, cuda_mode=args.cuda)
		break
	#save_samples(generator, encoder, cp_name=args.cp_path.split('/')[-1].split('.')[0], cuda_mode=args.cuda)
