#!/bin/python


preprocessor = "facereclib.preprocessing.FaceCrop(cropped_image_size = (#1, #1*4/5), cropped_positions = {'reye':(#1/5, #1/5-1), 'leye':(#1/5, #1*3/5)}, offset=#4)"

feature_extractor = "lgbphs"

tool = "facereclib.tools.LGBPHS(distance_function = #2, is_distance_function = #3)"


replace = {
    'preprocessing' : {
        "#1" : {
            'S080' : '80',
            'S040' : '40',
            'S160': '160'
        },
        "#4" : {
            'O0' : '0',
            'O2' : '2'
        }
    },

    'scoring' : {
        "(#2, #3)" : {
            'D1' : ('bob.math.histogram_intersection', 'False'),
            'D2' : ('bob.math.chi_square', 'True')
        }
    }
}

requirements = ["#1 > #4"]

imports = ['bob', 'facereclib']
