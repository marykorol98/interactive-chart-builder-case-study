# Building an Interactive Chart Builder (Anonymized Case Study)

## Problem

The platform required an in-app, interactive chart builder that could be reused across multiple services. The charting logic lived in a shared ML-core package, and its plot methods had to accept standardized data inputs and, when needed, ML models. The existing plots covered common chart types, but users needed:

- More visualization options for exploratory analysis (e.g., decision boundaries, geospatial points, tree visualizations).
- Clearer, more consistent parameterization and validation for charts that already existed.
- Improvements to UX-related behavior like sorting and hover labels.

The challenge was to expand the chart catalog while keeping compatibility with the core application.

## Constraints

From the codebase and context, the key constraints were:

- **Shared-core packaging:** The plot methods live in a shared ML-core library used by three other services. This required careful, isolated changes (the context indicates the core library is pulled as a dependency in other services via `pyproject.toml`).
- **Uniform data interface:** Plots consume `DictDataFrame`, a wrapper around `pandas` structures, and results must be returned in a consistent format.
- **Model-aware plots:** Some charts need access to a trained ML model instance (e.g., decision boundary visualization, tree visualization, SHAP-based explanations).
- **Anonymization and evidence limits:** I cannot include product names, proprietary data details, or performance metrics that are not present in the provided evidence sources.
- **Uniform output structure** the output plot should be always a json file wich can bee easily interpereted as a plotly object.

## Solution

My role and responsibilities covered both **transformating the existing plot methods** and **building new plot methods from scratch**. This helped expand the chart catalog while stabilizing and refining the behavior of the existing charts.

### New plot methods (built from scratch)

Based on the `new_plot_methods/` folder (which contains only new charts authored from scratch), I implemented the following chart types:

1. **BubbleChart**
   - Extends scatter plotting with a third variable for size and color.
   - Allows optional opacity conditioning to reduce visual clutter.
   - Validates that `column_x`, `column_y`, and `size_condition` are distinct and that parameter types are correct.

2. **DecisionBoundaryDisplay**
   - Visualizes classifier decision regions using a grid-based heatmap plus observed points.
   - Fits the provided model to two selected features, then plots prediction boundaries.
   - Supports optional prediction annotations and grid resolution tuning.

3. **GeoDataVisualizer**
   - Plots geometry columns on a map using Plotly’s mapbox backend.
   - Exposes map-style choices and requires a runtime `boto_handler`.
   - Detects geometry columns via `geopandas` types.

4. **TreeGraphVisualizer**
   - Produces a node-and-edge view of a decision tree or a selected estimator from an ensemble.
   - Controls tree depth and handles class name labeling.
   - Generates dynamic node sizing and color mapping for readability.

5. **WindowCorrelationPlot**
   - Calculates windowed Pearson correlations between two series.
   - Highlights low-correlation intervals based on a configurable threshold.
   - Supports time-based indexing and window sizing controls.

6. **SHAP plot suite (bar, beeswarm, heatmap, waterfall)**
   - Implements multiple SHAP visualization modes with a shared base class.
   - Provides per-plot style options and parameter templates.
   - Adds support for sample-level navigation in the waterfall view.

These new plots are implemented as modular classes that align with the core library’s `BaseMethod` and data interface conventions.

### Transformations of legacy plots

To improve stability and user experience of existing plots, I targeted specific behavioral issues and parameter inconsistencies. The changes were focused on compatibility and clarity rather than wholesale rewrites.

Key improvements included:

- **Sorting mixin robustness:** Added a warning and fallback behavior when sorting by a column that contains duplicates, preserving consistent X-axis labeling.
- **Index-to-values plots:** Defaulted X-axis labeling to the active sort column and enabled sorting by `datetime` inputs.
- **Bar charts:** Added datetime column support and standardized hover labels to reflect aggregation semantics.
- **SHAP Waterfall:** Fixed per-sample base value handling and improved slider layout updates to keep each sample’s formatting intact.
- **Plot visibility logic:** Ensured certain plots (e.g., regression) are hidden in feature-only contexts.

## Representative Before → After examples

### 1) Sorting with duplicate keys (SortingMixin)

**Before:** Sorting by a selected column did not account for duplicate values in the index. The code assumed a clean index or replaced it without warning, which could cause confusing X-axis labeling.

**After:** The sorting logic now checks for duplicate values and warns when it must fall back to numeric indexing. It also preserves a consistent axis name when duplicates appear.

- **Before:** `sort_values(..., ignore_index=...)` without duplicate awareness.
- **After:**
  - Sets the index to the selected sort column.
  - Warns if duplicates are detected.
  - Resets the index and uses a numeric X-axis label when necessary.

Impact: Users get safer, more transparent sorting behavior and fewer surprises when data has repeated keys.

### 2) SHAP Waterfall: base value and slider behavior

**Before:** The waterfall plot derived `base_value` directly from the global `shap_values` object and used a single layout template for all samples. This could lead to mismatched base values and inconsistent per-sample layouts in the slider view.

**After:** The implementation now passes a per-sample `base_value` into `shap_plot_one`, builds the slider with per-sample layouts, and sorts features by descending contribution magnitude.

- **Before:**
  - `base_value = shap_values.base_values[index]`
  - Shared layout template reused for all slider steps.
- **After:**
  - `shap_plot_one(shap_values=shap_values[i], base_value=shap_values.base_values[i], ...)`
  - Slider steps include layout JSON for each sample.
  - Feature ordering uses descending absolute SHAP values.

Impact: Each slider step now reflects the correct base value and layout for the chosen sample, and feature ranking is more intuitive.

### 3) Bar chart hover labeling and datetime support

**Before:** Hover labels were inconsistent across aggregation modes, and datetime columns were not accepted as sortable categories.

**After:** The hover templates were standardized around aggregation semantics, and datetime columns were added to the allowed type list for chart grouping and sorting.

- **Before:** Limited dtype support and inconsistent hover labeling.
- **After:**
  - Datetime added to acceptable column types.
  - Hover template reflects `agg_func` (e.g., `count` vs. `sum`).

Impact: More predictable chart behavior for time-based data and clearer tooltip messaging for users.

## Trade-offs

- **Validation vs. flexibility:** Many plots enforce strict parameter and type checks to prevent invalid configurations. The trade-off is that users may need to adjust their data or parameters more carefully, but the resulting plots are more reliable and debuggable.
- **Performance vs. clarity:** Decision boundary and window-correlation plots accept adjustable grid/window sizes. Higher resolution yields more detail but increases computation time.
- **Shared layout consistency vs. per-sample fidelity:** The SHAP waterfall slider now stores per-sample layouts to avoid formatting drift; this is slightly more code and memory but preserves clarity for the viewer.

## Outcomes

Based on the code evidence and commit history, the outcome of this work was a broader, more resilient charting layer that:

- Expands the chart catalog with entirely new plot types while preserving the core `BaseMethod` interface.
- Improves the stability and UX of existing plots (sorting behavior, hover clarity, parameter validation).
- Adds richer interpretability plots (decision boundaries, tree visualizations, SHAP suite) suitable for model analysis workflows.

No quantitative metrics are included here because the provided evidence does not contain performance benchmarks or usage telemetry.

## Lessons learned

1. **Data validation is part of UX.** Making parameter checks explicit (e.g., missing columns, type mismatches) prevents silent failures and leads to more predictable plotting behavior.
2. **Sorting is deceptively complex.** Real-world data often contains duplicate keys; designing chart behavior around that reality (warnings, index fallback) reduces user confusion.
3. **Explainability plots require careful input handling.** SHAP visualizations depend on accurate base values and sample-level context; small mismatches can mislead users.
4. **Time-aware data support should be explicit.** Supporting `datetime` in sorting and grouping pipelines avoids awkward preprocessing in upstream services.
