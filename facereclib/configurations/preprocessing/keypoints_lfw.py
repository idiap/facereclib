#!/usr/bin/env python

import facereclib

# Cropping
CROPPED_IMAGE_HEIGHT = 200
CROPPED_IMAGE_WIDTH  = 200

# eye positions for frontal images
RIGHT_EYE_POS = (49, 74)
LEFT_EYE_POS  = (49,124)

# eye and mouth position for profile images
# (only appropriate for left profile images; change them for right profiles)
EYE_POS = (16, 20)
MOUTH_POS = (52, 20)

MOUTH_L = 0.891206278045033
EYE_NOSE_Y = 0.668364812848271
OUTER_EYE_D = 1.479967422
INNER_EYE_D = 0.51218784
EYE_MOUTH_Y = 0.9956389228141739

# define the preprocessor
preprocessor = facereclib.preprocessing.Keypoints(
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cropped_positions = {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS, 'eye' : EYE_POS, 'mouth' : MOUTH_POS},
    crop_image = True,
    fixed_annotations = {'reyeo': [0.,-OUTER_EYE_D/2.], 'reyei': [0.,-INNER_EYE_D/2.], 'leyei': [0.,INNER_EYE_D/2.], 'leyeo': [0.,OUTER_EYE_D/2.], 'nose': [EYE_NOSE_Y,0.], 'mouthr': [EYE_MOUTH_Y,-MOUTH_L/2.], 'mouthl': [EYE_MOUTH_Y,MOUTH_L/2.]},
    cropped_domain_annotations = True,
    relative_annotations = True,
    use_eye_corners = False,
)


## STATS from LFW view 1 
## {'reyei': [115.48380670536831, 113.11136914304652], 'noser': [140.4323506468296, 113.13770860302006], 'reyeo': [115.63667387613764, 91.82381848032283], 'mouthr': [159.08627651327848, 106.45847958499651], 'leyeo': [113.86483469742254, 157.2321633797239], 'leyei': [114.74230842973407, 135.74792073056972], 'mouth_l': 0.891206278045033, 'nose': [144.31331172804994, 125.1101254873035], 'eye_nose_y': 0.668364812848271, 'eye_l': 0.4865302894481733, 'eye_mouth_y': 0.9956389228141739, 'mouthl': [158.30553167261158, 145.45601446753412], 'nosel': [140.28373842721356, 138.31275313820768], 'eye_d': 44.195800478591934}
