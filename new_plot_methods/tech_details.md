# Interactive Chart Builder (Detailed Description)

This repository contains anonymized, evidence-based documentation of my work on an in-app interactive chart builder feature. The implementation lives in the `plot_methods` package of a core ML library that is used as a dependency by three other services. The chart builder exposes configurable plotting methods that accept `DictDataFrame` inputs (a wrapper around `pandas` data structures) and, in some cases, an ML model instance.

---

## Built from scratch (new chart methods)

The following chart methods were built from scratch and are represented in `new_plot_methods/`:

### BubbleChart (`new_plot_methods/bubble_chart.py`)
- Extends scatter plotting with a third variable controlling **size**, **color**, and optionally **opacity**.
- User-facing parameters include `column_x`, `column_y`, `size_condition`, and `conditioned_opacity`.
- Defaults include a frequency-based size condition and a configurable marker size.
- Validates column selection and parameter types before plotting.

### DecisionBoundaryDisplay (`new_plot_methods/decision_boundary_display.py`)
- Visualizes a classifier’s decision regions using a 2D feature grid and model predictions.
- Parameters include `x_1`, `x_2`, `grid_resolution`, plus required `target_column` and optional `predict_column`.
- Fits the provided model, generates a prediction heatmap, and overlays class-labeled points.
- Enforces non-text feature types and validates class count.

### GeoDataVisualizer (`new_plot_methods/geo_visualizer.py`)
- Plots geometry data on a map using Plotly’s mapbox layer.
- Parameters include geometry `columns` and `map_style`.
- Requires a `boto_handler` input to be supplied at runtime.
- Detects geometry columns using `geopandas` types.

### TreeGraphVisualizer (`new_plot_methods/tree_graph_visualizer.py`)
- Visualizes decision trees (or a chosen tree in a forest) using Plotly traces.
- Parameters include `tree_index` (for ensembles) and `max_depth`.
- Traverses the model’s tree structure to build nodes, edges, and class-colored labels.

### WindowCorrelationPlot (`new_plot_methods/window_correlation_plot.py`)
- Shows how correlation between two series changes across sliding windows.
- Parameters include `index_column`, `x_1`, `x_2`, `window_size`, and `corr_threshold`.
- Uses Pearson correlation for windowed analysis and highlights low-correlation periods.

### SHAP plot suite (`new_plot_methods/shap_methods/*`)
- Adds a SHAP-based visualization suite with multiple plot types:
  - **Bar**: top feature contributions.
  - **Beeswarm**: distribution of SHAP values per feature.
  - **Heatmap**: SHAP values across samples/features.
  - **Waterfall**: per-sample decomposition of predictions.
- Includes a shared base class for generating SHAP values and reusing layout/style logic.

---

## Transformed legacy (improvements to existing charts)

- **Sorting behavior and warnings**
  - Added duplicate-index handling and a warning when sorting by columns with repeated values (see `plot_methods/_sorting_mixin.py`).
  - Improved X-axis labeling when falling back to index-based ordering.

- **Index-to-values plot UX tuning**
  - Defaulted X-axis name to the selected sort column.
  - Allowed sorting by `datetime` columns.
  - Adjusted line width behavior tied to marker size (`plot_methods/index_to_values_plot.py`).

- **Bar chart data compatibility and hover text**
  - Enabled datetime columns as valid categorical/sorting inputs.
  - Standardized hover templates and y-axis labeling (`plot_methods/bar_chart.py`).

- **SHAP Waterfall behavior fixes**
  - Corrected base value handling per sample.
  - Ensured slider interactions preserve per-sample layout data.
  - Sorted displayed features by descending contribution magnitude (`plot_methods/shap_methods/waterfall.py`).

- **Display logic refinements**
  - Added a guard so regression plots are hidden in feature-only contexts (`plot_methods/regression.py`).
