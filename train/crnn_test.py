import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
from config import opt
from crnn import crnn
from PIL import Image
from torchvision import transforms

class resizeNormalize(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		self.size = size
		self.interpolation = interpolation
		self.toTensor = transforms.ToTensor()

	def __call__(self, img):
		img = img.resize(self.size, self.interpolation)
		img = self.toTensor(img)
		img.sub_(0.5).div_(0.5)
		return img

def decode(preds,char_set):
	pred_text = ''
	for i in range(len(preds)):
		if preds[i] != 0 and ((i == 0) or (i != 0 and preds[i] != preds[i-1])):
			pred_text += char_set[int(preds[i])-1]

	return pred_text

# test if crnn work

if __name__ == '__main__':

	imagepath = './test.jpg'

	img_h = opt.img_h
	use_gpu = opt.use_gpu
	modelpath = opt.modelpath
	char_set = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
	char_set = ''.join([ch.strip('\n') for ch in char_set[1:]] + ['卍'])
	n_class = len(char_set)

	model = crnn.CRNN(img_h, 1, n_class, 256)
	if torch.cuda.is_available and use_gpu:
		model.cuda()

	if os.path.exists(modelpath):
		print('Load model from "%s" ...' % modelpath)
		model.load_state_dict(torch.load(modelpath))
		print('Done!')

	image = Image.open(imagepath).convert('L')
	(w,h) = image.size
	size_h = 32
	ratio = size_h / float(h)
	size_w = int(w * ratio)
	# keep the ratio
	transform = resizeNormalize((size_w, size_h))
	image = transform(image)
	image = image.unsqueeze(0)
	if torch.cuda.is_available and use_gpu:
		image = image.cuda()
	model.eval()
	preds = model(image)
	preds = preds.max(2)[1]
	preds = preds.squeeze()
	pred_text = decode(preds,char_set)
	print('predict == >',pred_text)


