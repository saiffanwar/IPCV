import argparser
import numpy as np
import matplotlib.pyplot as plt 

# Calculates the precision, recall and F1 score for a detection task with targets and predictions made.
def eval(tagets, predicts):

	# Counters for True Positives, False Positives and False Negatives
	TP, FP, FN = 0

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

		# Pair up targets with closest predict
		pairs = []
		unpaired_predicts = predicts

		# Calculate dist between each target and remaining predicts
		for t in targets:
			dists = []
			for p in unpaired_predicts:
				dists.append(boundary_distance(t, p))

			# Pair target with closest predict and remove predict 
			i = np.argmin(dists)
			pairs.append(t, unpaired_predicts[i])
			del unpaired_predicts[i]

		# Evaluate matched pairs
		for t, p in pairs:
			correct = iou(t, p)

			if(correct==1):
				TP += 1
			else:
				FP += 1

		# Handle left over targets/predicts that weren't paired
		FP += len(unpaired_predicts)
		FN += max(0, targets-predicts)

	# calculates precision, recall and F1 score
	precision, recall = precision(TP, FP), recall(TP, FN)
	return precision, recall, f1(precision, recall)

# Given two boundaries target and predict, determines prediction was correct based on IoU threshold
def iou(target, predict, threshold=0.5):
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
	intersection = (Xi2 - Xi1) * (Yi2 - Yi1)

	# Union area
	union = At + Ap - intersection

	# IntersectionOverUnion
	iou = intersection/union

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
	retrun round((x+w)/2), round((y+h)/2)

# Calculates the distance between the centres of two boundaries, b1 and b2
def boundary_distance(b1, b2):
	x1, y1 = boundary_centre(b1)
	x2, y2 = boundary_centre(b2)
	return sqrt((x2-x1)**2 + (y2-y1)**2)



