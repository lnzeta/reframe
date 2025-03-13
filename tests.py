import numpy as np
import pandas as pd

from reframe.reframe import run_analysis


class TestReframe:
    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10000],
            "B": [-1000, 1, 2, 3, 4, 5, 5, 6, 7, 8],
            "C": [-1, -2, -3, -4, -5, 0, 1, 2, 3, 4],
            "D": "a,b,c,d,e,f,g,h,i,i".split(","),
            "E": [1e-10] * 10,
            "F": [np.nan] * 10,
            "G": [np.inf] * 10,
            "H": [1] * 5 + ["a"] * 5,
            "I": [1] * 5 + ["a"] * 5,
            "L": [1] * 5 + ["a"] * 5,
        }
    )

    df2 = pd.concat(
        [df1, pd.DataFrame({"L": [1] * 5 + ["b"] * 5, "I": [1] * 5 + ["a"] * 5})],
        axis=1,
    )

    df3 = pd.DataFrame(
        {
            "A": [1, 1, 2],
            "B": [1, 1, 3],
        }
    )

    def test_has_issues(self):
        result = run_analysis(self.df1)
        assert result.has_issues == True
