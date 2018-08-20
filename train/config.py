class DefaultConfig(object):
	train_data_root = '/home/DataSet/data_train.txt'
	validation_data_root = '/home/DataSet/data_test.txt'
	modelpath = '/home/chineseocr/train/models/pytorch-crnn.pth'
	image_path = '/home/DataSet/images'

	batch_size = 64
	img_h = 32
	num_workers = 4
	use_gpu = True
	max_epoch = 10
	learning_rate = 0.0005
	weight_decay = 1e-4
	printinterval = 200
	valinterval = 1000

def parse(self,**kwargs):
	for k,v in kwargs.items():
		setattr(self,k,v)

DefaultConfig.parse = parse
opt = DefaultConfig()