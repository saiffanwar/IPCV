import numpy as np
from pprint import pprint

# Calculates the precision, recall and F1 score for a detection task with targets and predictions made.
def eval(targets, predicts):

	# Counters for True Positives, False Positives and False Negatives
	TP, FP, FN = 0, 0, 0

	# If there are no targets or predicts: assign perfect precision, recall & f1
	if(len(targets) == 0 and len(predicts) == 0):
		return 1, 1, 1

	# If there are targets but no predicts are made: assign all targets are FNs
	elif(len(targets) != 0 and len(predicts) == 0):
		FN = len(targets)

	# If there are no targets but predicts are made: assign all predicts as FPs
	elif(len(targets) == 0 and len(predicts) != 0):
		FP = len(predicts)

	# If there are both targets and predicts
	else:

		# Calculate IOU between each target and predict
		IOUs = [[iou(t, p) for p in predicts] for t in targets]

		# Apply threshold on IOUs to determine detections
		detections = np.asarray([[iouThreshold(x) for x in iou] for iou in IOUs])

		# for each row that has no 1s in the thresholded iou matrix, there is a FN
		for row in detections:
			if np.sum(row) == 0:
				FN +=1
		detections, FP = conflictResolver(detections)
		TP = np.sum(detections)
		print(detections)


	# calculates precision, recall and F1 score
	p, r = precision(TP, FP), recall(TP, FN)
	return p, r, f1(p, r)

def conflictResolver(detections):
	FP = 0
	for j in range(detections.shape[1]):
		if np.sum(detections[:,j]) > 1:
			detections[:,j] = 0
	for i in range(detections.shape[0]):
		if np.sum(detections[i,:]) > 1:
			detections[i,:] = 0
	for col in detections.T:
		if np.sum(col) == 0:
			FP += 1


	return detections, FP



# Given two boundaries target and predict, determines prediction was correct based on IoU threshold
def iou(target, predict):
	Xt1, Yt1, Wt, Ht = target
	Xp1, Yp1, Wp, Hp = predict

	# Calculates bottom right corner of boundaries and their areas
	Xt2, Yt2 = Xt1 + Wt, Yt1 + Ht
	At = (Xt2 - Xt1) * (Yt2 - Yt1)

	Xp2, Yp2 = Xp1 + Wp, Yp1 + Hp
	Ap = (Xp2 - Xp1) * (Yp2 - Yp1)

	# Determines intersection rectangle corners
	Xi1 = max(Xt1, Xp1)
	Yi1 = max(Yt1, Yp1)
	Xi2 = min(Xt2, Xp2)
	Yi2 = min(Yt2, Yp2)


	# Intersection area
	intersectionWidth = Xi2 - Xi1
	intersectionHeight = Yi2 - Yi1

	# Checks for no overlap case
	if (intersectionWidth < 0) or (intersectionHeight < 0):
		return 0

	intersection = intersectionWidth * intersectionHeight
	# Union area
	union = At + Ap - intersection

	# IntersectionOverUnion
	iou = intersection/union
	return iou

def iouThreshold(iou, threshold = 0.5) :
	# Applies threshold to determine if positive or negative prediction
	if iou <= threshold:
		return 0
	else:
		return 1

# Calculates Recall measurement based on True Positive and False Negative counts
def recall(TP, FN):

	# If both 0, this is equal to 100% recall
	if(TP == 0 and FN == 0):
		return 1

	return TP / (TP+FN)

# Calculates Precision measurement based on True Positive and False Positive counts
def precision(TP, FP):

	# If both 0, this is equal to 100% precision
	if(TP == 0 and FP == 0):
		return 1
	print(TP, FP)
	return TP / (TP+FP)

# Calculates F1 Score based on precision and recall
def f1(precision, recall):
	return 2 * ((precision*recall) / (precision+recall))
