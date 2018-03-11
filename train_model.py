from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import pickle
import glob
import time
from common import extract_features

# YCrCb gave the highest test accuracy
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)
hist_bins = 16
spatial_feat = True
hist_feat = True
hog_feat = True

cars = glob.glob('vehicles/**/*.png')
noncars = glob.glob('non-vehicles/**/*.png')

print('Car images:', len(cars))
print('Non-Car images:', len(noncars))

t1 = time.time()
car_features = extract_features(cars, cspace=color_space,
                                hist_feat=hist_feat, hist_bins=hist_bins,
                                spatial_feat=spatial_feat, spatial_size=spatial_size,
                                hog_feat=hog_feat, orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block, hog_channel=hog_channel)
notcar_features = extract_features(noncars, cspace=color_space,
                                   hist_feat=hist_feat, hist_bins=hist_bins,
                                   spatial_feat=spatial_feat, spatial_size=spatial_size,
                                   hog_feat=hog_feat, orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block, hog_channel=hog_channel)
t2 = time.time()
diff = t2 - t1
avg = diff / (len(cars) + len(noncars))
print('Time to extract features:', round(diff, 2), 'sec')
print('Avg extraction time per image:', round(avg, 4), 'sec')

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(car_features[0]))

X = np.vstack((car_features, notcar_features)).astype(np.float64)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# shuffle
X, y = shuffle(X, y)

rand_state = np.random.randint(0, 100)
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand_state)

X_scaler = None
if hist_feat or spatial_feat:
    # fit scalar on training data
    X_scaler = StandardScaler().fit(X)
    # Scale both training and test data
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

svc = LinearSVC()

# time the classifier
t1 = time.time()
svc.fit(X_train, y_train)
t2 = time.time()

print('Time to train:', round(t2 - t1, 2), 'sec')

print('Test accuracy:', round(svc.score(X_test, y_test), 4))

model_info = {'svc': svc,
              'X_scaler': X_scaler,
              'color_space': color_space,
              'orient': orient,
              'pix_per_cell': pix_per_cell,
              'cell_per_block': cell_per_block,
              'hog_channel': hog_channel,
              'spatial_size': spatial_size,
              'hist_bins': hist_bins,
              'spatial_feat': spatial_feat,
              'hist_feat': hist_feat,
              'hog_feat': hog_feat,
              }
pickle.dump(model_info, open("model_info.p", "wb"))
