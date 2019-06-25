from sklearn.metrics import classification_report
import autokeras as ak
import os

from autokeras import ImageClassifier


RAINING_TIMES = [60 * 60, # 1 hour
		60 * 60 * 2,	# 2 hours
		60 * 60 * 4,	# 4 hours
		60 * 60 * 8,	# 8 hours
		60 * 60 * 12,	# 12 hours
		60 * 60 * 24,	# 24 hours
	]
