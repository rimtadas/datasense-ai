
import gradio as gr
import pandas as pd
from pathlib import Path

from profiler import profile_dataset


def analyze_file(file_path):
    """Load a dataset and generate its profiling report."""

    if file_path is None:
        raise gr.Error("Please upload a CSV or Parquet file.")

    file_path = Path(file_path)

    try:
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)

        elif file_path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(file_path)

        else:
            raise gr.Error("Unsupported file format. Upload CSV or Parquet.")

        if df.empty:
            raise gr.Error("The uploaded dataset is empty.")

        profile = profile_dataset(df)

        # Prepare the report for display
        display_profile = profile.copy()

        if "top_values" in display_profile.columns:
            display_profile["top_values"] = (
                display_profile["top_values"]
                .apply(lambda x: str(x) if isinstance(x, dict) else x)
            )

        summary = f"""
### Dataset Overview

- **Rows:** {len(df):,}
- **Columns:** {len(df.columns):,}
- **Total missing values:** {int(df.isna().sum().sum()):,}
- **Duplicate rows:** {int(df.duplicated().sum()):,}
- **Memory usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""

        return summary, display_profile

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(f"Unable to profile dataset: {str(e)}")


with gr.Blocks(title="DataSense AI") as app:

    gr.Markdown(
        """
        # DataSense AI
        ### Intelligent Data Discovery & Profiling Assistant

        Upload a dataset to automatically discover its structure,
        data quality issues and statistical characteristics.
        """
    )

    file_input = gr.File(
        label="Upload your dataset",
        file_types=[".csv", ".parquet", ".pq"],
        type="filepath"
    )

    analyze_button = gr.Button(
        "Analyze Dataset",
        variant="primary"
    )

    summary_output = gr.Markdown(
        label="Dataset Overview"
    )

    profile_output = gr.Dataframe(
        label="Column Profiling Report",
        interactive=False,
        wrap=True
    )

    analyze_button.click(
        fn=analyze_file,
        inputs=file_input,
        outputs=[summary_output, profile_output]
    )


if __name__ == "__main__":
    app.launch()
