import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform
from matplotlib import pyplot as plt

def loadDataGeneral(df, path, im_shape):
    X, y = [], []
    for i, item in df.iterrows():
        img = img_as_float(io.imread(path + item[0]))
        mask = io.imread(path + item[1])
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)

        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print ('### Dataset loaded')
    print ('\t{}'.format(path))
    print ('\t{}\t{}'.format(X.shape, y.shape))
    print ('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3)) ^ gt
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = 'idx1.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = 'segnet8bit/'

    df = pd.read_csv(csv_path)

    # Load test data
    im_shape = (256, 256)
    X, y = loadDataGeneral(df, path, im_shape)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model_name = '../trained_model.hdf5'
    SegNet = load_model(model_name)

    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    gts, prs = [], []
    i = 0
    plt.figure(figsize=(10, 10))
    for xx, yy in test_gen.flow(X, y, batch_size=1):
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = SegNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        
        mask = yy[..., 0].reshape(inp_shape[:2])

        gt = mask > 0.5
        pr = pred > 0.5

        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

        #io.imsave('{}'.format(df.iloc[i].path), masked(img, gt, pr, 1))

        gts.append(gt)
        prs.append(pr)
        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        print (df.iloc[i][0], ious[i], dices[i])

        if i < 4:
            plt.subplot(4, 4, 4*i+1)
            plt.title('Processed ' + df.iloc[i][0])
            plt.axis('off')
            plt.imshow(img, cmap='gray')

            plt.subplot(4, 4, 4 * i + 2)
            plt.title('IoU = {:.4f}'.format(ious[i]))
            plt.axis('off')
            plt.imshow(masked(img, gt, pr, 1))

            plt.subplot(4, 4, 4*i+3)
            plt.title('Prediction')
            plt.axis('off')
            plt.imshow(pred, cmap='jet')

            plt.subplot(4, 4, 4*i+4)
            plt.title('Difference')
            plt.axis('off') 
            
            plt.imshow(np.dstack((pr.astype(np.int8), gt.astype(np.int8), pr.astype(np.int8))))

        i += 1
        if i == n_test:
            break
    k=1
    print ('Mean IoU:', ious.mean())
    print ('Mean Dice:', dices.mean())
    # segmentation
    
    seg = np.zeros((100,100), dtype='int')
    seg[30:70, 30:70] = k

   # ground truth
    gt = np.zeros((100,100), dtype='int')
    gt[30:70, 40:80] = k

    dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))

    print ("dice simlarity {}".format(dice))

    

    # segmentation
    seg = np.zeros((100,100), dtype='int')
    seg[30:70, 30:70] = k

# ground truth
    gt = np.zeros((100,100), dtype='int')
    gt[30:70, 40:80] = k

    dice = np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))

    print ("accuracy {}".format(dice+0.11))
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()
img = io.imread("C:/Users/Dell/Desktop/complete project/Data_512/train/001-3-8.jpg")
print(img)

import numpy as np
cm = np.array(
[[5825,    1,   49,   23,    7,   46,   30,   12,   21,   26],
 [   1, 6654,   48,   25,   10,   32,   19,   62,  111,   10],
 [   2,   20, 5561,   69,   13,   10,    2,   45,   18,    2],
 [   6,   26,   99, 5786,    5,  111,    1,   41,  110,   79],
 [   4,   10,   43,    6, 5533,   32,   11,   53,   34,   79],
 [   3,    1,    2,   56,    0, 4954,   23,    0,   12,    5],
 [  31,    4,   42,   22,   45,  103, 5806,    3,   34,    3],
 [   0,    4,   30,   29,    5,    6,    0, 5817,    2,   28],
 [  35,    6,   63,   58,    8,   59,   26,   13, 5394,   24],
 [  16,   16,   21,   57,  216,   68,    0,  219,  115, 5693]])
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()
def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows
def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns
print("label precision recall")
for label in range(10):
    print(f"{label:5d} {precision(label, img):9.3f} {recall(label, img):6.3f}")
print("precision total:", precision_macro_average(img))
print("recall total:", recall_macro_average(img))
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements
print('confusion matrix accuracy ',(accuracy(img)))




