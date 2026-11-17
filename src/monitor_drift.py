from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset


def check_data_drift():

    print("Starting data drift monitoring...")

    # 1. Load reference Iris dataset
    iris = load_iris(as_frame=True)

    reference = iris.frame.sample(
        n=100,
        random_state=42
    )

    # 2. Simulate current production data
    current = iris.frame.sample(
        n=100,
        random_state=99
    ).copy()

    # Introduce artificial drift
    current["sepal length (cm)"] *= 1.5

    # 3. Generate HTML drift report
    try:
        report = Report(
            metrics=[DataDriftPreset()]
        )

        snapshot = report.run(
            reference_data=reference,
            current_data=current
        )

        snapshot.save_html("drift_report.html")

        print("HTML drift report generated successfully.")

    except Exception as e:
        print(f"HTML report generation failed: {e}")

    # 4. Detect drift using KS statistical test
    drifted_columns = 0

    feature_cols = iris.feature_names

    for col in feature_cols:

        p_value = ks_2samp(
            reference[col],
            current[col]
        ).pvalue

        if p_value < 0.05:
            drifted_columns += 1

    # 5. Declare drift if at least one feature has drifted
    dataset_drift = drifted_columns >= 1

    print(f"Drifted columns: {drifted_columns}")
    print(f"Dataset Drift Detected: {dataset_drift}")

    return dataset_drift


if __name__ == "__main__":
    check_data_drift()