import numpy as np
import numpy.random as npr
import cv2
from torchvision import transforms
from PIL import Image

from core.config import cfg
import utils.blob as blob_utils


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data', 'rois', 'labels']
    return blob_names


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales = _get_image_blob(roidb)

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    blobs['data'] = im_blob
    # print(im_blob.shape)
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)

    num_images = len(roidb)
    for im_i in range(num_images):
        labels, im_rois = _sample_rois(roidb[im_i], num_classes)

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        # print('minibatch', len(rois_blob_this_image))

        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            rois_blob_this_image = rois_blob_this_image[index, :]

        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels blob
        labels_blob = np.vstack((labels_blob, labels))

    blobs['rois'] = rois_blob
    blobs['labels'] = labels_blob

    return blobs, True, index, im_scales, inv_index


def get_minibatch_tuple(roidb, extra_roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_blob_extra, im_scales = _get_image_blob_tuple(roidb, extra_roidb)

    assert len(im_scales) == 2, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    blobs['data'] = im_blob
    blobs['data_extra'] = im_blob_extra
    # print(im_blob.shape)
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)

    rois_blob_extra = np.zeros((0, 5), dtype=np.float32)
    labels_blob_extra = np.zeros((0, num_classes), dtype=np.float32)

    num_images = len(roidb)
    for im_i in range(num_images):
        labels, im_rois = _sample_rois(roidb[im_i], num_classes)
        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            rois_blob_this_image = rois_blob_this_image[index, :]
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        # Add to labels blob
        labels_blob = np.vstack((labels_blob, labels))


        labels_extra, im_rois_extra = _sample_rois(extra_roidb[im_i], num_classes)
        # Add to RoIs blob
        rois_extra = _project_im_rois(im_rois_extra, im_scales[im_i+1])
        batch_ind_extra = im_i * np.ones((rois_extra.shape[0], 1))
        rois_blob_this_image_extra = np.hstack((batch_ind_extra, rois_extra))
        if cfg.DEDUP_BOXES > 0:
            if not cfg.OICR.MSCALES:
                v = np.array([1, 1e3, 1e6, 1e9, 1e12])
                hashes_extra = np.round(rois_blob_this_image_extra * cfg.DEDUP_BOXES).dot(v)
                _, index_extra, inv_index_extra = np.unique(hashes_extra, return_index=True,
                                                return_inverse=True)
            else:
                index_extra, inv_index_extra = index, inv_index
            rois_blob_this_image_extra = rois_blob_this_image_extra[index_extra, :]
        
        rois_blob_extra = np.vstack((rois_blob_extra, rois_blob_this_image_extra))
        # Add to labels blob
        labels_blob_extra = np.vstack((labels_blob_extra, labels_extra))

    blobs['rois'] = rois_blob
    blobs['labels'] = labels_blob
    blobs['rois_extra'] = rois_blob_extra
    blobs['labels_extra'] = labels_blob_extra

    return blobs, True, index, im_scales

def get_minibatch_sametuple(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_blob_t, im_scales = _get_image_blob_sametuple(roidb)

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    blobs['data'] = im_blob
    blobs['data_t'] = im_blob_t
    # print(im_blob.shape)
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)

    num_images = len(roidb)
    for im_i in range(num_images):
        labels, im_rois = _sample_rois(roidb[im_i], num_classes)

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        # print('minibatch', len(rois_blob_this_image))

        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            rois_blob_this_image = rois_blob_this_image[index, :]

        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels blob
        labels_blob = np.vstack((labels_blob, labels))

    blobs['rois'] = rois_blob
    blobs['labels'] = labels_blob

    return blobs, True, index, im_scales, inv_index


def _sample_rois(roidb, num_classes):
    """Generate a random sample of RoIs"""
    labels = roidb['gt_classes']
    rois = roidb['boxes']

    if cfg.TRAIN.BATCH_SIZE_PER_IM > 0:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_IM
    else:
        batch_size = np.inf
    if batch_size < rois.shape[0]:
        rois_inds = npr.permutation(rois.shape[0])[:batch_size]
        rois = rois[rois_inds, :]

    return labels.reshape(1, -1), rois


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales


def _get_image_blob_tuple(roidb, extra_roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = 1
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images*2)
    processed_ims = []
    processed_ims_extra = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[0]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

        im_extra = cv2.imread(extra_roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(extra_roidb[i]['image'])
        if extra_roidb[i]['flipped']:
            im_extra = im_extra[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[1]]
        im_extra, im_scale_extra = blob_utils.prep_im_for_blob(
            im_extra, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale_extra[0])
        processed_ims_extra.append(im_extra[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)
    blob_extra = blob_utils.im_list_to_blob(processed_ims_extra)
    print(blob.shape, blob_extra.shape)

    return blob, blob_extra, im_scales

def _get_image_blob_sametuple(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    processed_ims_extra = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])

        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        # transform
        transforms_img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        im_t = Image.fromarray(im)
        im_t = transforms_img(im_t)
        im_t = np.array(im_t)

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])
        
        im_t, _ = blob_utils.prep_im_for_blob(
            im_t, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        processed_ims_extra.append(im_t[0])

        # print(im[0].shape, im_t[0].shape)

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)
    blob_extra = blob_utils.im_list_to_blob(processed_ims_extra)

    return blob, blob_extra, im_scales


def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois
