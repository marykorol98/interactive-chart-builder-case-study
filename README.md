# Interactive Chart Builder — Portfolio Case Study (Anonymized)

## Overview
I worked on an in-app **interactive chart builder** that lets users configure charts through a GUI and render them in the application. The charting logic lives in a shared ML-core library used as a dependency by three other backend microservices.

---

## My Role (Scope of Work)
I focused on two tracks:
1. **Improving existing plots** (UX/UI tuning, validation, correctness of parameter labels, and sorting behavior).
2. **Building new plot methods from scratch** to expand the chart catalog.

---

## Project Context
- **Where the feature lives:** The chart builder is implemented in the `plot_methods` package of an ML-core library, which is then used as a dependency by three other services. 
- **Data interface:** Plot methods recieve around `pandas` data structures wrapped into a pre-defined `DictDataFrame` class which is used across the services, so I needed to consider its support.
- **Model-aware charts:** Some plots accept a trained ML model instance to visualize its behavior.

---

## What I Built From Scratch (New Plot Methods)
All items below are present in `new_plot_methods/` and summarized from code evidence.

1. **BubbleChart**
   - Scatter plot enhanced with size and color conditioning (optional opacity).
   - Validates columns and parameter types before plotting. 

2. **DecisionBoundaryDisplay**
   - 2D decision boundary visualization using a prediction grid plus labeled points.
   - Enforces non-text feature types and validates class counts

3. **GeoDataVisualizer**
   - Map-based visualization for geometry data via Plotly mapbox.
   - Requires a runtime `boto_handler` and detects geometry columns using `geopandas` types. 

4. **TreeGraphVisualizer**
   - Visualizes decision trees (or a selected estimator in a forest) using Plotly traces. 
   - Supports `tree_index` selection and `max_depth`.

5. **WindowCorrelationPlot**
   - Shows Pearson correlation drift between two series across sliding windows.
   - Highlights low-correlation intervals by threshold.

6. **SHAP Plot Suite (bar, beeswarm, heatmap, waterfall)**
   - Implements multiple SHAP visualization types with a shared base class.
   - Adds sample-level navigation in the waterfall view.
---

## Improvements to Existing Plots (Selected Examples)
The changes below target behavior and UX consistency in legacy plots.

- **SortingMixin:** Added duplicate-index handling and warnings; improved X-axis labeling when falling back to index-based ordering.
- **Index-to-Values Plot:** Defaulted X-axis naming to the selected sort column; enabled datetime sorting; adjusted line width based on marker size.
- **Bar Chart:** Expanded datetime support and standardized hover labels.
- **SHAP Waterfall:** Fixed per-sample base value handling and per-sample slider layout state; sorted features by descending contribution magnitude.
- **Regression Plot Visibility:** Added guard for feature-only contexts.

---

## Representative Before → After
These examples demonstrate the impact of the changes:

### 1) Sorting with Duplicate Keys
- **Before:** Sorting assumed unique values and could lead to confusing X-axis labeling when duplicates existed.
- **After:** Duplicate-aware sorting with warnings and consistent axis labeling fallback.

### 2) SHAP Waterfall: Base Value + Slider Behavior
- **Before:** Base values and layout were applied globally, risking mismatch per sample.
- **After:** Per-sample `base_value` and layout data preserved per slider step, with features sorted by contribution magnitude.

### 3) Bar Chart Hover Labels + Datetime Support
- **Before:** Hover labels varied across aggregation modes; datetime columns weren’t accepted as sortable categories.
- **After:** Standardized hover templates and datetime support for grouping/sorting.

---

## Design Trade-offs
- **Validation vs. flexibility:** Strong parameter/type checks increase reliability at the cost of stricter inputs.
- **Performance vs. detail:** Grid/window resolution choices balance computation cost and visual granularity.
- **Per-sample fidelity vs. code size:** Storing per-sample layout data in SHAP waterfall improves correctness but adds complexity.

---

## Outcomes
- Broadened the chart catalog with new, model-aware visualization types (decision boundaries, tree graphs, SHAP suite, etc.).
- Improved reliability and UX of existing plots (sorting, hover labels, parameter handling).
---