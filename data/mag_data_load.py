from __future__ import print_function
import os
import os.path
import errno
import torch.utils.data as data
import torch
import codecs

class mag_load(data.Dataset):
	raw_folder = 'raw'
	r_training_file = 'training.mat'
	r_test_file = 'test.mat'
	processed_folder = 'processed'
	p_training_file = 'training.pt'
	p_test_file = 'test.pt'

	def __init__(self, root,train=True,transform=None,target_transform=None,download=False):
		self.root = os.path.expanduser(root)
		self.transform = transform
		self.target_transform = target_transform
		self.train = train
		
		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found.' +
					' You can use download=True to download it')

		if self.train:
			self.train_data, self.train_labels = torch.load(
					os.path.join(self.root, self.processed_folder, self.p_training_file))
		else:
			self.test_data, self.test_labels = torch.load(
					os.path.join(self.root, self.processed_folder, self.p_test_file))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
		tuple: (signal, [T1,T2])
		"""
		if self.train:
			sig, T_val = self.train_data[index], self.train_labels[index]
		else:
			sig, T_val = self.test_data[index], self.test_labels[index]
		
		return sig, T_val
	
	def __len__(self):
		if self.train:
			return len(self.train_data)
		else:
			return len(self.test_data)
	
	def _check_exists(self):
		return os.path.exists(os.path.join(self.root, self.processed_folder, self.p_training_file)) and \
			os.path.exists(os.path.join(self.root, self.processed_folder, self.p_test_file))

	def download(self):
		"""Creates the data set from the .mat file if not done"""
		from scipy.io import loadmat

		if self._check_exists():
			return

		if not os.path.exists(os.path.join(self.root, self.raw_folder, self.r_training_file)):
			raise RuntimeError('Dataset not found.' +
					' You have to compute it through matlab')
		try:
			os.makedirs(os.path.join(self.root, self.processed_folder))
		except OSError as e:
			if e.errno == errno.EEXIST:
				pass
			else:
				raise

		# process and save as torch files
		print('Processing...')
		train_data = loadmat(os.path.join(self.root,self.raw_folder,self.r_training_file))
		test_data = loadmat(os.path.join(self.root,self.raw_folder,self.r_test_file))
		if 'T1' in train_data.keys():
			"""TODO : fix size single comp data"""
			training_set = (
					torch.from_numpy(train_data['sig']).type(torch.FloatTensor),
					torch.from_numpy(train_data['T1']).type(torch.FloatTensor).squeeze().unsqueeze(1)
					)
			test_set = (
					torch.from_numpy(train_data['sig']).type(torch.FloatTensor),
					torch.from_numpy(train_data['T1']).type(torch.FloatTensor).squeeze().unsqueeze(1)
					)
		else:
			"""Either T1 is a key or T, depending on the single or double parameter estimation"""
			training_set = (
					torch.from_numpy(train_data['sig']).type(torch.FloatTensor),
					torch.from_numpy(train_data['T']).type(torch.FloatTensor)
					)
			test_set = (
					torch.from_numpy(test_data['sig_t']).type(torch.FloatTensor),
					torch.from_numpy(test_data['T_t']).type(torch.FloatTensor)
					)


		with open(os.path.join(self.root, self.processed_folder, self.p_training_file), 'wb') as f:
			torch.save(training_set, f)
		with open(os.path.join(self.root, self.processed_folder, self.p_test_file), 'wb') as f:
			torch.save(test_set, f)

		print('Done!')
