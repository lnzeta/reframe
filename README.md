# Re(view) (data)Frame
Quickly spot common issues in a Pandas dataframe

# Install
```
pip install git+https://github.com/lnzeta/reframe.git@main
```

# Examples
## Usage
```
from reframe.reframe import run_analysis

df = pd.DataFrame(
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
df = pd.concat(
    [df, pd.DataFrame({"L": [1] * 5 + ["b"] * 5, "I": [1] * 5 + ["a"] * 5})], axis=1
)

result = run_analysis(df)
```

## Output
```
###################################
### RE(VIEW) (DATA)FRAME REPORT ###
###################################

### Possible issues in dataframe: Yes

Issues per column        A    B    C    D    E    F    G    H    I    L    L   I
----------------------  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---
Missing Values           X                        X
Duplicate Values              X         X    X    X    X    X    X    X    X   X
Has Multiple Datatypes                                      X    X    X    X   X
Is Object Datatype                      X                   X    X    X    X   X
Positive Values          X    X    X         X         X
Negative Values               X    X
Zero Values                        X
Small Values                                 X
Large Values                                           X
Infinite Values                                        X
Outliers Boxplot         X    X
Outliers Zscore          X    X
High Correlation         X         X
Zero Variance                                X    X    X
Empty                                             X

Issues per column (count)     A    B    C    D    E    F    G    H    I    L    L   I
---------------------------  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---
Missing Values                1    0    0    0    0   10    0    0    0    0    0   0
Unique Values                10    9   10    9    1    1    1    2    2    2    2   2
Duplicate Values              0    1    0    1    9    9    9    8    8    8    8   8
Has Multiple Datatypes        0    0    0    0    0    0    0    1    1    1    1   1
Is Object Datatype            0    0    0    1    0    0    0    1    1    1    1   1
Positive Values               9    9    4        10    0   10
Negative Values               0    1    5         0    0    0
Zero Values                   0    0    1         0    0    0
Small Values                  0    0    0        10    0    0
Large Values                  0    0    0         0    0   10
Infinite Values               0    0    0         0    0   10
Outliers Boxplot              1    1    0         0    0    0
Outliers Zscore               1    1    0         0    0    0
High Correlation              1    0    1         0    0    0

Issues per row
---------------------------  -----
Has Duplicated Rows          False
Has Duplicated Column Names  True
Duplicated Column Names      I, L
Duplicated Columns           I, L
```
