import pickle
from scipy.ndimage.measurements import label
from common import *
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

model_info = pickle.load(open("model_info.p", "rb"))

svc = model_info['svc']
X_scaler = model_info['X_scaler']
color_space = model_info['color_space']
orient = model_info['orient']
pix_per_cell = model_info['pix_per_cell']
cell_per_block = model_info['cell_per_block']
hog_channel = model_info['hog_channel']
spatial_size = model_info['spatial_size']
hist_bins = model_info['hist_bins']
spatial_feat = model_info['spatial_feat']
hist_feat = model_info['hist_feat']
hog_feat = model_info['hog_feat']


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins):
    # rectangles where cars found
    ractangles = []
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            if X_scaler is not None:
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            else:
                test_features = hog_features.reshape(1, -1)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                ractangles.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return ractangles


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    rects = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image and final rectangles
    return img, rects


class Heatmap:
    def __init__(self):
        self.heatmaps = []
        self.num_heats = 5
        self.avg_heat = None

    def add(self, heatmap):
        self.heatmaps.append(heatmap)
        if len(self.heatmaps) > self.num_heats:
            self.heatmaps = self.heatmaps[len(self.heatmaps) - self.num_heats:]
        self.avg_heat = np.mean(self.heatmaps, axis=0)


ystart = 400
ystop = None
scales = [1.1, 1.5, 2, 2.5]
heatmap_threshold = 2

heat = Heatmap()


def get_heatmap(img):
    boxes = []
    for scale in scales:
        rects = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                          orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        boxes.append(rects)
    boxes = [box for recs in boxes for box in recs]
    heatmap_img = np.zeros_like(img[:, :, 0]).astype(np.float)
    heatmap_img = add_heat(heatmap_img, boxes)
    heat.add(heatmap_img)
    heatmap_img = apply_threshold(heat.avg_heat, heatmap_threshold)

    return heatmap_img


def process_frame(img):
    draw_img = np.copy(img)
    heatmap_img = get_heatmap(img)
    labels = label(heatmap_img)
    draw_img, rects = draw_labeled_bboxes(draw_img, labels)
    return draw_img

# test_images = glob.glob('./test_images/test*.jpg')

# fig, axs = plt.subplots(3, 2, figsize=(16,14))
# fig.subplots_adjust(hspace = .004, wspace=.002)
# axs = axs.ravel()

# for i, im in enumerate(test_images):
#     heat = Heatmap()
#     axs[i].imshow(get_heatmap(mpimg.imread(im)), cmap='hot')
#     axs[i].axis('off')
# plt.show()

# fig, axs = plt.subplots(3, 2, figsize=(16,14))
# fig.subplots_adjust(hspace = .004, wspace=.002)
# axs = axs.ravel()
#
# for i, im in enumerate(test_images):
#     heat = Heatmap()
#     axs[i].imshow(process_frame(mpimg.imread(im)))
#     axs[i].axis('off')
# plt.show()


heat = Heatmap()
test_out_file = 'test_video_out.mp4'
print('started', test_out_file)
clip_test = VideoFileClip('test_video.mp4')
clip_test_out = clip_test.fl_image(process_frame)
clip_test_out.write_videofile(test_out_file, audio=False)

heat = Heatmap()
project_out_file = 'project_video_out.mp4'
print('started', project_out_file)
clip_project = VideoFileClip('project_video.mp4')
clip_project_out = clip_project.fl_image(process_frame)
clip_project_out.write_videofile(project_out_file, audio=False)
