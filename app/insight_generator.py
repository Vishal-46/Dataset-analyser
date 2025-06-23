def generate_insights(df, report):
    insights = []
    insights.append(f"The dataset has {report['num_rows']} rows and {report['num_columns']} columns.")
    
    num_missing = sum(v for v in report["missing_values"].values() if v > 0)
    if num_missing:
        insights.append("Some columns have missing values. Consider cleaning before analysis.")
    
    if report["duplicates"] > 0:
        insights.append(f"There are {report['duplicates']} duplicate rows.")
    else:
        insights.append("No duplicate rows found. Good!")

    return insights
