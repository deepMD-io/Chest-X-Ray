import os
import time
import numpy as np
import torch
from scipy.special import softmax


def MultiLabelLoss(criterion, output, target):
	loss, length = 0, target.size()[1] # output dims: 0 for sample, 1 for label, 2 for class	
	for i in range(length):
		loss += criterion(output[:,i,:], target[:,i])

	return loss/length

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():
		batch_size = target.size(0)
		output_label = torch.sigmoid(output)
		pred = (torch.sign(output_label - 0.5)+1)/2
		correct = pred.eq(target).sum()
		return correct * 100.0 / batch_size


def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	#accuracy = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		
		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input) # num_batch x 14 x 3
		loss = MultiLabelLoss(criterion,output, target) # mean of CrossEntropyLoss() over 14 labels
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		#accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
		#For speed, we dont compute accuracy
		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses))

	return losses.avg#, accuracy.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	#accuracy = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input) # num_batch x 14 x 3
			loss = MultiLabelLoss(criterion, output, target) # mean of CrossEntropyLoss() over 14 labels

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			#accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
			
			y_true = target.detach().to('cpu').numpy().tolist()
			# .max(-1) return ( tensor([[maxes],...]), tensor([[idxes of max], ...])
			y_pred = output.detach().to('cpu').max(-1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred))) # use .extend() for batch

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses))

	return losses.avg, results

# def getprob(model, device, data_loader, print_freq=10):
# 	batch_time = AverageMeter()
# 	losses = AverageMeter()

# 	results = []

# 	model.eval()

# 	with torch.no_grad():
# 		end = time.time()
# 		for i, (input, target) in enumerate(data_loader):

# 			if isinstance(input, tuple):
# 				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
# 			else:
# 				input = input.to(device)
# 			target = target.to(device)

# 			output = model(input) # num_batch x 14 x 3

# 			# measure elapsed time
# 			batch_time.update(time.time() - end)
# 			end = time.time()

# 			#accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
			
# 			y_true = target.detach().to('cpu').numpy()#.tolist()
# 			# output: num_batch x 14 x 3
# 			y_pred = output.detach().to('cpu').numpy()
# 			y_pred = y_pred[:,:,:2] # drop uncertain
# 			y_pred = softmax(y_pred, axis = -1)
# 			y_pred = y_pred[:,:,1] # only positive
# 			results.extend(list(zip(y_true, y_pred))) # use .extend() for batch

# 			if i % print_freq == 0:
# 				print('Test: [{0}/{1}]\t'
# 					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
# 					i, len(data_loader), batch_time=batch_time))

# 	return results


def make_kaggle_submission(list_id, list_prob, path):
	if len(list_id) != len(list_prob):
		raise AttributeError("ID list and Probability list have different lengths")

	os.makedirs(path, exist_ok=True)
	output_file = open(os.path.join(path, 'my_predictions.csv'), 'w')
	output_file.write("SUBJECT_ID,MORTALITY\n")
	for pid, prob in zip(list_id, list_prob):
		output_file.write("{},{}\n".format(pid, prob))
	output_file.close()
