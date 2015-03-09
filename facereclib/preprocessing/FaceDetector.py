import math
import numpy

import bob.ip.facedetect
import bob.ip.flandmark

import bob.io.base

from .Preprocessor import Preprocessor
from .NullPreprocessor import NullPreprocessor
from .. import utils

class FaceDetector (NullPreprocessor):

  def __init__(
        self,
        post_processor = "face-crop",
        use_flandmark = False,
        cascade = None,
        detection_overlap = 0.2,
        distance = 2,
        scale_base = math.pow(2., -1./16.),
        lowest_scale = 0.125,
        **kwargs
  ):
    """Performs a face detection in the given image (ignoring any annotations)."""
    # call base class constructor
    NullPreprocessor.__init__(
      self,
      cascade = cascade,
      post_processor = post_processor,
      detection_overlap = detection_overlap,
      distance = distance,
      scale_base = scale_base,
      lowest_scale = lowest_scale,
      **kwargs
    )

    self.m_sampler = bob.ip.facedetect.Sampler(scale_factor=scale_base, lowest_scale=lowest_scale, distance=distance)
    if cascade is None:
      self.m_cascade = bob.ip.facedetect.default_cascade()
    else:
      self.m_cascade = bob.ip.facedetect.Cascade(bob.io.base.HDF5File(cascade))
    self.m_detection_overlap = detection_overlap
    # load post-processor
    if isinstance(post_processor, Preprocessor):
      self.m_post_processor = post_processor
    else:
      self.m_post_processor = utils.resources.load_resource(post_processor, "preprocessor")
    self.m_flandmark = bob.ip.flandmark.Flandmark() if use_flandmark else None

    self.m_quality = None


  def _landmarks(self, image, bounding_box):
    # get the landmarks in the face
    if self.m_flandmark is not None:
      # use the flandmark detector
      uint8_image = image.astype(numpy.uint8)
      # make the bounding box square shape by extending the horizontal position by 2 pixels times width/20
      bb = bob.ip.facedetect.BoundingBox(topleft = (bounding_box.top_f, bounding_box.left_f - bounding_box.size[1] / 10.), size = bounding_box.size)

      top = max(bb.topleft[0], 0)
      left = int(max(bb.topleft[1], 0))
      bottom = min(bb.bottomright[0], image.shape[0])
      right = int(min(bb.bottomright[1], image.shape[1]))
      landmarks = self.m_flandmark.locate(uint8_image, bb.top, bb.left, bb.size[0], bb.size[1])

      if landmarks is not None and len(landmarks):
        return {
          'reye' : ((landmarks[1][0] + landmarks[5][0])/2., (landmarks[1][1] + landmarks[5][1])/2.),
          'leye' : ((landmarks[2][0] + landmarks[6][0])/2., (landmarks[2][1] + landmarks[6][1])/2.)
        }
      else:
        utils.warn("Could not detect landmarks -- using estimated landmarks")

    # estimate from default locations
    return bob.ip.facedetect.expected_eye_positions(bounding_box)


  def __call__(self, image, annotations=None):
    # convert to the desired color channel
    gray_image = NullPreprocessor.__call__(self, image)

    # detect the face
    bounding_box, self.m_quality = bob.ip.facedetect.detect_single_face(gray_image, self.m_cascade, self.m_sampler, self.m_detection_overlap)

    # get the eye landmarks
    annotations = self._landmarks(gray_image, bounding_box)

    # perform post-processing
    return self.m_post_processor(image, annotations)


  def read_data(self, filename):
    return self.m_post_processor.read_data(filename)

  def save_data(self, data, filename):
    return self.m_post_processor.save_data(data, filename)


  def quality(self):
    return self.m_quality
