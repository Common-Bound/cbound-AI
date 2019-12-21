import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from sklearn.preprocessing import RobustScaler
from keras.layers import InputLayer, Dense, Activation, Dropout

class Model():
	def __init__(self, weight='weight.h5', optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
		self.model = Sequential([
			Dense(256, input_shape=(6,)), Activation('relu'),
			Dense(256), Activation('tanh'),
			Dense(16), Activation('tanh'),
			Dense(128), Activation('sigmoid'),
			Dense(8), Activation('sigmoid'),
			Dense(8), Activation('sigmoid'),
			Dense(8), Activation('relu'),
			Dense(8), Activation('tanh'),
			Dense(1), Activation('sigmoid'),
			])

		self.load_weight(weight)
		self.load_scaler()

	def load_weight(self, weight="weight.h5"):
		self.model.load_weights("weight.h5")

	def complie_model(self):
		self.model.compile(optimizer='adam',
			loss='binary_crossentropy',
			metrics=['accuracy'])

	def load_scaler(self):
		self.read_features = pd.read_csv('features.csv', index_col=0)
		self.robust = RobustScaler()
		self.robust.fit(self.read_features)

	def predict(self, features):
		scaled_features = self.robust.transform(features)
		preds = self.model.predict(scaled_features)

		return preds

def extract_feature(data):
	features = pd.DataFrame(columns=['crop_time', 'image_time', 'human_len', 'ai_len', 'prob', 'similarity'])

	index = 0
	sum_crop_time = 0
	for image_data in data['meta']['crop_image']:
		sum_crop_time += image_data['region_attributes']['crop_time']

	for image_data in data['meta']['crop_image']:
		row = dict()

		row['crop_time'] = image_data['region_attributes']['crop_time'] / 1000
		row['image_time'] = image_data['region_attributes']['image_time'] *(image_data['region_attributes']['crop_time'] / sum_crop_time) / 1000
		row['human_len'] = int(len(image_data['region_attributes']['label']))
		row['ai_len'] = int(len(image_data['region_attributes']['ai_label']))
		row['prob'] = image_data['region_attributes']['prob']
		row['similarity'] = image_data['region_attributes']['similarity']

		features.loc[index] = row
		index+=1

	return features