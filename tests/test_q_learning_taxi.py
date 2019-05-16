import pytest
import main
import numpy as np


@pytest.mark.parametrize("list,line,cols_list,expected", [
    ([[0, 0], [0, 0]], 0, None, 0),
    ([[0, 1], [0, 0]], 0, None, 1),
    ([[0, 1], [0, 0]], 0, [0], 0),
    ([[1, 1], [0, 0]], 0, None, 1),
])
def test_max_expected_future_reward(list, line, cols_list, expected):
    q_table = np.array(list)
    result = main.max_expected_future_reward(q_table, line,
                                             cols_list)
    assert result == expected