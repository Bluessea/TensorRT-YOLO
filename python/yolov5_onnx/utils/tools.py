import cv2
import numpy as np
import random, colorsys
from config import voc, anchors, strides
import torchvision
def get_grid(nx, ny):

    nx_vec = np.arange(nx)
    ny_vec = np.arange(ny)
    yv,xv = np.meshgrid(ny_vec, nx_vec)
    grid = np.stack((yv, xv), axis=2)
    grid = grid.reshape(1, 1, ny, nx, 2)
    return  grid

def sigmoid(array):
    return np.reciprocal(np.exp(-array) + 1.0)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(boxes, confs, classes, iou_thres=0.6):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = confs.flatten().argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    boxes = boxes[keep]
    confs = confs[keep]
    classes = classes[keep]
    return boxes, confs, classes

def NMS(pred, iou_thres=0.6):

    pred[:,5:] *= pred[:, 4:5]
    # c = pred[:, 5:6] * 7680
    boxes = xywh2xyxy(pred[..., :4])
    confs = np.amax(pred[:, 5:], 1, keepdims=True)
    classes = np.argmax(pred[:, 5:], axis=-1)
    # i = torchvision.ops.nms(boxes,pred[:,4],0.5)
    # print("i_num:",i)
    return non_max_suppression(boxes,confs,classes, iou_thres)

# handle boxes
def processor_boxes(output):

    nl = len(anchors)
    a = anchors.copy().astype(np.float32)
    a = a.reshape(nl, -1, 2)
    anchor_grid = a.copy().reshape(nl, 1, -1, 1, 1, 2)
    
    conf_thres = 0.5
    """
    i = 0
    for out in output:
        print("outshape:",out.shape)
       
        print(out[0][0][0][i])
        i = i +1
        if i == 3:
            return 0 
    """
    # get object and get class
    scaled = []
    grids = []
    for out in output:
        # out = sigmoid(out)
        _,_,width, height,_= out.shape
        grid = get_grid(width, height)
        grids.append(grid)
        scaled.append(out)
    z = []
    for out, grid, stride, anchor in zip(scaled, grids, strides, anchor_grid):

        _, _, width, height, _ = out.shape
        out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
        out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor

        out[..., 5:] = out[..., 4:5] * out[..., 5:]
        out = out.reshape((1, 3 * width * height, 25))
        z.append(out)

    pred = np.concatenate(z, 1)
    xc = pred[..., 4] > conf_thres
    print("xc_num:",np.sum(xc==True))
    pred = pred[xc]
    # boxes = xywh2xyxy(pred[:, :4])
    return NMS(pred)

def processor_boxes_one(output):

    conf_thres = 0.5
    xc = output[..., 4] > conf_thres
    output = output[xc]
    return NMS(output)


def draw_boxes(img, boxes):
    window_name = 'boxes'
    cv2.namedWindow(window_name)
    copy = img.copy()
    overlay = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # cv2.addWeighted(overlay, 0.05, copy, 1 - 0.5, 0, copy)
    cv2.imshow(window_name, overlay)
    cv2.waitKey(1)

def gen_colors(classes):
    """
        generate unique hues for each class and convert to bgr
        classes -- list -- class names (80 for coco dataset)
        -> list
    """
    hsvs = []
    for x in range(len(classes)):
        hsvs.append([float(x) / len(classes), 1., 0.7])
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = []
    for hsv in hsvs:
        h, s, v = hsv
        rgb = colorsys.hsv_to_rgb(h, s, v)
        rgbs.append(rgb)

    bgrs = []
    for rgb in rgbs:
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        bgrs.append(bgr)
    return bgrs


def draw_results(img, boxes, confs, classes):
    window_name = 'final results'
    cv2.namedWindow(window_name)
    overlay = img.copy()
    final = img.copy()
    color_list = gen_colors(voc)
    for box, conf, cls in zip(boxes, confs, classes):
        # draw rectangle
        x1, y1, x2, y2 = box
        conf = conf[0]
        cls_name = voc[cls]
        color = color_list[cls]
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        # draw text
        cv2.putText(final, '%s %f' % (cls_name, conf), org=(int(x1), int(y1 + 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255))
    cv2.addWeighted(overlay, 0.5, final, 1 - 0.5, 0, final)
    cv2.imwrite("result.jpg",final)
    cv2.imshow(window_name, final)
    cv2.waitKey(10000)
    return final
