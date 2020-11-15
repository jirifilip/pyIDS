def test_submodularity(s_set, t_set, function):
    left_side = function(s_set.union(t_set)) + function(s_set.intersection(t_set))
    right_side = function(s_set) + function(t_set)

    result = left_side <= right_side

    return result 