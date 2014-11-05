#!/bin/python

# This file describes an exemplary configuration file that can be used in combination with the bin/parameter_test.py script.


# The preprocessor uses various image resolutions, which are specified by the height #1, and several offsets, specified by #4
preprocessor = "facereclib.preprocessing.FaceCrop(cropped_image_size = (#1, #1*4/5), cropped_positions = {'reye':(#1/5, #1/5-1), 'leye':(#1/5, #1*3/5)}, offset=#4)"

# The feature extractor uses the **default** 'lgbphs' option, which is registered as a resource
feature_extractor = "lgbphs"

# The face recognition algorithm applies different distance functions (#2), which might be distance or similarity functions (#3)
tool = "facereclib.tools.LGBPHS(distance_function = #2, is_distance_function = #3)"


# Here, we define, which placeholder keys (#.) should be replaces by which values in which stage of the processing toolchain
replace = {
    # For preprocessing, select several image resolutions and overlaps
    'preprocessing' : {
        # image resolution (height)
        "#1" : {
            # place height 80 in sub-directory 'S080'
            'S080' : '80',
            # place height 40 in sub-directory 'S040'
            'S040' : '40',
            # place height 160 in sub-directory 'S160'
            'S160' : '160'
        },
        # offsets
        "#4" : {
            # place offset 0 to sub-directory 'O0'
            'O0' : '0',
            # place offset 2 to sub-directory 'O2'
            'O2' : '2'
        }
    },

    # For scoring, select several distance functions
    'scoring' : {
        # Replace #2 and #3 **at the same time**
        "(#2, #3)" : {
            # For distance_function = 'bob.math.histogram_intersection' and is_distance_function = False, place result in sub-directory 'D1'
            'D1' : ('bob.math.histogram_intersection', 'False'),
            # For distance_function = 'bob.math.chi_square' and is_distance_function = True, place result in sub-directory 'D2'
            'D2' : ('bob.math.chi_square', 'True')
        }
    }
}

# An optional list of requirements
# If these requirements are not fulfilled for the current values of #1 and #4, these experiments will not be executed.
requirements = ["#1 > 10*#4"]

# A list of imports that are required to use the defined preprocessor, feature_extractor and tool from above
imports = ['bob.math', 'facereclib']
