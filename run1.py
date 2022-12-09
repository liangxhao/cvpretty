axis = [4, 2, 2, 1, 1]
rank = 3

validator.check_value_type("axis", axis, [int, tuple, list], 'nansum')
validator.check_axis_valid(axis, rank)


