import os
import sys

import numpy as np
import pandas as pd

# TODO add option to print long or short report

_UGLY_TO_PRETTY_STRING = {}


def _disable_print():
    sys.stdout = open(os.devnull, "w")


def _enable_print():
    sys.stdout = sys.__stdout__


def _get_pretty_string(ugly):
    p = ugly.replace("count_", "").replace("_", " ").title()
    _UGLY_TO_PRETTY_STRING[ugly] = p
    return p


def _get_ugly_string(pretty):
    if pretty in _UGLY_TO_PRETTY_STRING:
        return _UGLY_TO_PRETTY_STRING[pretty]
    return ""


def _prettify_index(df):
    res = df.copy()
    res.index = [_get_pretty_string(c) for c in df.index]
    res.index.name = df.index.name
    return res


class ReframeResult:
    def __init__(self):
        self.has_issues = False
        self.issues_per_row = pd.Series()
        self.issues_per_column_count = pd.DataFrame()
        self.issues_per_column = pd.DataFrame()
        self.report = ReframeResultReport()


class ReframeResultReport:
    def __init__(self):
        self.has_issues = "No"
        self.issues_per_row = ""
        self.issues_per_column_count = ""
        self.issues_per_column = ""


# Count of outliers according to the box plot method
def _count_outliers_boxplot(
        series, iqr_multiplier=1.5, lower_quantile=0.25, upper_quantile=0.75
):
    q1 = series.quantile(lower_quantile)
    q3 = series.quantile(upper_quantile)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    return ((series < lower_bound) | (series > upper_bound)).sum()


# Count of outliers according to abs(zscore) > 2
def _count_outliers_zscore(series, max_z_score=2):
    from scipy import stats

    z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
    return (z_scores > max_z_score).sum()


def _analyze_columns(
        df,
        small_threshold=1e-9,
        large_threshold=1e9,
        max_z_score=2,
        iqr_multiplier=1.5,
        lower_quantile=0.25,
        upper_quantile=0.75,
        max_corr=0.5,
        axis=0,
        **kwargs
):
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(index=df.columns)

    # Count missing values (for all columns)
    result_df["count_missing_values"] = df.isnull().sum(axis=axis)

    # Count unique values (for all columns)
    result_df["count_unique_values"] = df.nunique(dropna=False, axis=axis)

    # Count duplicates (for all columns)
    result_df["count_duplicate_values"] = df.apply(
        lambda x: x.duplicated(), axis=axis
    ).astype(int).sum(axis=axis)

    result_df["has_multiple_datatypes"] = df.map(type).nunique() - 1
    result_df["is_object_datatype"] = df.dtypes.apply(lambda x: 1 if x == object else 0)

    # Initialize columns for numeric analysis
    result_df["count_positive_values"] = pd.NA
    result_df["count_negative_values"] = pd.NA
    result_df["count_zero_values"] = pd.NA
    result_df["count_small_values"] = pd.NA
    result_df["count_large_values"] = pd.NA
    result_df["count_infinite_values"] = pd.NA
    result_df["count_outliers_boxplot"] = pd.NA
    result_df["count_outliers_zscore"] = pd.NA
    result_df["count_high_correlation"] = pd.NA

    # Analyze numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return result_df.T

    # Count positive values
    result_df.loc[numeric_df.columns, "count_positive_values"] = (numeric_df > 0).sum(
        axis=axis
    )

    # Count negative values
    result_df.loc[numeric_df.columns, "count_negative_values"] = (numeric_df < 0).sum(
        axis=axis
    )

    # Count zero values
    result_df.loc[numeric_df.columns, "count_zero_values"] = (numeric_df == 0).sum(
        axis=axis
    )

    # Count small values
    result_df.loc[numeric_df.columns, "count_small_values"] = (
            (abs(numeric_df) < small_threshold) & (numeric_df != 0)
    ).sum(axis=axis)

    result_df.loc[numeric_df.columns, "count_large_values"] = (
            abs(numeric_df) > large_threshold
    ).sum(axis=axis)

    result_df.loc[numeric_df.columns, "count_infinite_values"] = (
            abs(numeric_df) == np.inf
    ).sum(axis=axis)

    result_df.loc[numeric_df.columns, "count_outliers_boxplot"] = numeric_df.apply(
        _count_outliers_boxplot,
        args=(iqr_multiplier, lower_quantile, upper_quantile),
        axis=axis,
    )

    result_df.loc[numeric_df.columns, "count_outliers_zscore"] = numeric_df.apply(
        _count_outliers_zscore, args=(max_z_score,), axis=axis
    )

    if axis != 0:
        return result_df.T

    # Number of columns with correlation > max_corr
    correlation_matrix = numeric_df.corr()
    for column in numeric_df.columns:
        result_df.loc[column, "count_high_correlation"] = max(
            (correlation_matrix[column].abs() > max_corr).sum()
            - 1,  # Exclude self-correlation
            0,
        )

    return result_df.T


def _get_issues_per_column(orig_df, df):
    df_tmp = df.copy()
    df_tmp.loc["zero_variance", :] = df_tmp.loc["count_unique_values", :] == 1
    df_tmp.loc["empty", :] = df_tmp.loc["count_missing_values", :] == len(orig_df)

    df_tmp = (
        df_tmp.drop("count_unique_values", axis=0)
        .apply(lambda x: x > 0, axis=0)
        .replace({True: "X", False: ""})
    )

    return df_tmp


def _get_duplicated_columns(df):
    tdf = df.transpose()
    duplicate_columns = tdf.duplicated(keep="first")
    return sorted(list(set(tdf[duplicate_columns].index)))


def _analyze_rows(df: pd.DataFrame) -> pd.Series:
    from collections import Counter

    result_dict = pd.Series(name="Issues per row")
    result_dict["has_duplicated_rows"] = [df.duplicated().any()]
    duplicated_column_names = sorted(
        [k for k, v in Counter(df.columns).items() if v > 1]
    )
    result_dict["has_duplicated_column_names"] = [len(duplicated_column_names) != 0]
    result_dict["duplicated_column_names"] = [", ".join(duplicated_column_names)]
    result_dict["duplicated_columns"] = [", ".join(_get_duplicated_columns(df))]
    return result_dict


def _run_analysis(df: pd.DataFrame, **kwargs) -> ReframeResult:
    result = ReframeResult()
    result.issues_per_row = _analyze_rows(df)
    result.issues_per_row.index.name = "Issues per row"

    result.issues_per_column_count = _analyze_columns(df, **kwargs)
    result.issues_per_column_count.index.name = "Issues per column (count)"

    exclude_issues = kwargs.get("exclude_issues", ())
    if exclude_issues:
        include_rows = [issue for issue in result.issues_per_column_count.index if
                        all(exc.lower() not in issue.lower() for exc in exclude_issues)]
        result.issues_per_column_count = result.issues_per_column_count.loc[include_rows,:]

    result.issues_per_column = _get_issues_per_column(
        df, result.issues_per_column_count
    )
    result.issues_per_column.index.name = "Issues per column"
    return result


def _has_issues(result: ReframeResult) -> bool:
    if len(result.issues_per_column) > 0:
        return True

    for c, val in result.issues_per_row:
        if val:
            return True

    return False


def _cap_data(df):
    for col in df.select_dtypes(np.number).columns:
        percentiles = df[col].quantile([0.01, 0.75]).values
        df.loc[df[col] <= percentiles[0], col] = percentiles[0]
        df.loc[df[col] >= percentiles[1], col] = percentiles[1]
    return df


def prepare_report(result):
    from tabulate import tabulate
    hi = result.has_issues
    result.report.has_issues = "Yes" if hi else "No"

    if not hi:
        return result

    tablefmt = "simple"

    _df = _prettify_index(result.issues_per_column)
    result.report.issues_per_column = (
        tabulate(
            _df,
            headers=[result.issues_per_column.index.name] + list(_df.columns),
            tablefmt=tablefmt,
            colalign=["left"]
                     + ["center"] * (len(result.issues_per_column_count.columns) - 1),
            rowalign="center",
        )
    )

    result.report.issues_per_column_count = (
        tabulate(
            _prettify_index(result.issues_per_column_count).fillna(""),
            headers="keys",
            tablefmt=tablefmt,
            colalign=["left"]
                     + ["center"] * (len(result.issues_per_column_count.columns) - 1),
            rowalign="center",
        )
    )

    result.report.issues_per_row = (tabulate(_prettify_index(result.issues_per_row), tablefmt=tablefmt,
                                             headers=[result.issues_per_row.name, ""]))
    return result


def print_report(result):
    hi = result.report.has_issues

    print("###################################")
    print("### RE(VIEW) (DATA)FRAME REPORT ###")
    print("###################################")
    print()

    print("### Possible issues in dataframe: " + hi)
    print()

    if not hi:
        return

    print(result.report.issues_per_column)
    print()

    print(result.report.issues_per_column_count)
    print()

    print(result.report.issues_per_row)
    print()

    return


def run_analysis(
        df: pd.DataFrame,
        small_threshold=1e-9,
        large_threshold=1e9,
        max_z_score=2,
        iqr_multiplier=1.5,
        lower_quantile=0.25,
        upper_quantile=0.75,
        max_corr=0.5,
        exclude_issues=(),
        print_to_console=True,
) -> ReframeResult:
    import warnings

    if not print_to_console:
        _disable_print()

    with (
        warnings.catch_warnings(),
        pd.option_context("display.max_rows", None, "display.max_columns", None),
    ):
        warnings.simplefilter("ignore")

        result = _run_analysis(**locals())
        hi = _has_issues(result)
        result.has_issues = hi

        result = prepare_report(result)

        if print_to_console:
            print_report(result)

        return result
