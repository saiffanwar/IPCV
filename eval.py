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
		FN += len(targets)

	# If there are no targets but predicts are made: assign all predicts as FPs
	elif(len(targets) == 0 and len(predicts) != 0):
		FP += len(predicts)

	# If there are both targets and predicts
	else:

		# Calculate IOU between each target and predict
		overlaps = []
		for t in targets:
			#ious in each iteration will be an array for each groundtruth image containing the iou with every prediction
			ious = []
			for p in predicts:
				ious.append(iou(t, p))
			overlaps.append(ious)
			print(ious)
		#overlaps is the array containing the subarrays of all the ious
		overlaps = np.asarray(overlaps)

#		for i in range(overlaps.shape[1])
#			iousi
#			for j in range(overlaps.shape[0])
#			iousi = np.append()




			# Pair target with closest predict and remove predict
#		i = np.argmin(dists)
		pairs.append((t, unpaired_predicts[i]))
		np.delete(unpaired_predicts, i)
		print(pairs)
		# Evaluate matched pairs
		for t, p in pairs:
			correct = iou(t, p)

			if(correct==1):
				TP += 1
			else:
				FP += 1

		# Handle left over targets/predicts that weren't paired
		FP += len(unpaired_predicts)
		FN += max(0, len(targets)-len(predicts))

	# calculates precision, recall and F1 score
	p, r = precision(TP, FP), recall(TP, FN)
	return p, r, f1(p, r)

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

	return TP / (TP+FP)

# Calculates F1 Score based on precision and recall
def f1(precision, recall):
	return 2 * ((precision*recall) / (precision+recall))

# Calculates the coords of the centre of a given boundary
def boundary_centre(boundary):
	x, y, w, h = boundary
	return round((x+w)/2), round((y+h)/2)

# Calculates the distance between the centres of two boundaries, b1 and b2
def boundary_distance(b1, b2):
	x1, y1 = boundary_centre(b1)
	x2, y2 = boundary_centre(b2)
	return np.sqrt((x2-x1)**2 + (y2-y1)**2)
