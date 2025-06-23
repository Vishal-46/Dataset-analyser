def run_quality_checks(df):
    report = {}
    report['num_rows'] = df.shape[0]
    report['num_columns'] = df.shape[1]
    report['missing_values'] = df.isnull().sum().to_dict()
    report['duplicates'] = df.duplicated().sum()
    return report
