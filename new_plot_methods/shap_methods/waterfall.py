import numpy as np
import pandas as pd
from lazy_imports import try_import
import plotly.graph_objs as go

from smile_ml_core.plot_methods.shap_methods import PlotBaseShapMethod
from smile_ml_core.plot_methods.styles.style_modification import StyleModificator

with try_import() as plotly_import:
    import shap


class WaterFall(PlotBaseShapMethod):
    styles_fields_exclude = {'scheme_color', 'marker_size', 'marker_color'}
    xaxis_title_default = 'Contributions'
    yaxis_title_default = 'Features'
    description = """
    Waterfall-график визуализирует вклад каждого признака в предсказание модели для одного наблюдения. 
    Он основан на SHAP (SHapley Additive exPlanations) значениях и показывает, как каждый признак увеличивает 
    или уменьшает базовое значение (ожидаемое значение модели) до итогового предсказания.

    На графике:
    - Горизонтальные стрелки представляют признаки, отсортированные по вкладу в предсказание.
    - Красные прямоугольники (bars) означают увеличение предсказания (положительное влияние), синие — уменьшение (отрицательное влияние).
    - Серая пунктирная линия слева показывает базовое значение `E[f(X)]`, т.е. среднее значение модели без учёта конкретного примера.
    - Серая пунктирная линия справа показывает итоговое предсказание `f(x)` для текущего наблюдения.
    - Подписи к признакам включают их значение и имя (например: `<значение> = <название признака>`), менее важные признаки агрегируются в группу `"other features"`.

    Назначение графика — дать интерпретируемое объяснение предсказания модели для конкретного объекта. Это помогает понять, **почему модель приняла то или иное решение**, что особенно важно в задачах, требующих прозрачности (например, в медицине, финансах и юридических системах).

    **Как интерпретировать:**
    - Чем длиннее прямоугольник, тем сильнее влияние признака на результат.
    - Отсутствие вкладов (очень короткие прямоугольник) может говорить о низкой важности признака для конкретного предсказания.
    - Сумма всех SHAP значений + базовое значение = итоговое предсказание модели.

    """

    styles_modificators = getattr(PlotBaseShapMethod, 'styles_modificators', []) + [
        StyleModificator(id_='xaxis_title', value=xaxis_title_default),
        StyleModificator(id_='yaxis_title', value=yaxis_title_default),
    ]

    def shap_plot_one(
        self,
        shap_values: 'shap.Explanation',
        base_value: float,
        feature_names: list[str],
        y: float | int | None = None,
        **kwargs,
    ) -> go.Figure:
        """Waterfall plot для одного элемента датасета (точнее его shap_value)"""
        if len(shap_values.shape) != 1:
            raise ValueError('Only single sample SHAP values are supported.')

        if not isinstance(base_value, (int, float)):
            raise ValueError('One Explanation should have one base_value')

        values = shap_values.values
        features = shap_values.display_data if shap_values.display_data is not None else shap_values.data
        feature_names = shap_values.feature_names

        if isinstance(features, pd.Series):
            if feature_names is None:
                feature_names = features.index.tolist()
            features = features.values

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(values))]

        order = np.argsort(np.abs(values))  # сортировка по значениям
        ordered_values = values[order]
        ordered_names = [feature_names[i] for i in order]
        if features is not None:
            feature_values = features[order]
        else:
            feature_values = ['' for _ in ordered_names]

        num_features = min(self.max_display, len(values))
        individual_names = []
        individual_values = []

        # Шаблон для оборачивания серых подписей
        span_style_template = '<span style="color:gray">{value}</span>'

        EPSILON = 1e-6  # small value for visualizing zero SHAP values

        # Готовим данные для каждой фичи
        for i in range(len(ordered_values[: self.max_display - 1])):
            val = ordered_values[i]
            display_val = val if abs(val) > EPSILON**2 else EPSILON  # avoid zero height bar
            name = (
                span_style_template.format(value=feature_values[i]) + f' = {ordered_names[i]}'
                if feature_values[i] != ''
                else ordered_names[i]
            )
            individual_names.append(name)
            individual_values.append(display_val)

        if len(values) > self.max_display:
            remaining_value = np.sum(ordered_values[self.max_display - 1 :])
            display_val = remaining_value if abs(remaining_value) > EPSILON**2 else EPSILON
            individual_names.append(f'{len(values) - self.max_display + 1} other features')
            individual_values.append(display_val)

        # Итоговое предсказание f(x)
        fx = base_value + np.sum(values)

        texts = [f'{v:+.3f}' for v in individual_values[:num_features]]
        hover_texts = [f'SHAP: {v:+.3f}' for v in individual_values[:num_features]]
        font_size = self.styles.get('font_size', 12).value

        fig = go.Figure(
            go.Waterfall(
                name='SHAP',
                orientation='h',
                measure=['relative'] * len(individual_values),
                y=individual_names,
                x=individual_values,
                text=texts,
                base=base_value,
                connector={'line': {'color': 'rgba(0, 0, 0, 0.5)', 'dash': 'dot'}},  # серые вертикальные линии
                increasing={'marker': {'color': 'red'}},
                decreasing={'marker': {'color': 'blue', 'line': {'width': 0}}},
                totals={'marker': {'color': '#2ca02c'}},
                customdata=[[t] for t in hover_texts],
                hovertemplate='%{customdata[0]}<extra></extra>',
            )
        )

        # отметка E[f(X)]
        fig.add_vline(
            x=base_value,
            line_dash='dot',
            line_color='gray',
            annotation_text='E[f(X)]' + span_style_template.format(value=f' = {base_value:.3f}'),
            annotation_position='top left',
            annotation_font_size=font_size,
        )

        # отметка f(X)
        fig.add_vline(
            x=fx,
            line_dash='dot',
            line_color='gray',
            annotation_text=f'Target: {y}<br>f(x)' + span_style_template.format(value=f' = {fx:.3f}'),
            annotation_position='top right',
            annotation_font_size=font_size,
        )

        fig.update_layout(
            showlegend=False,
            waterfallgap=0.3,
            yaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)', gridwidth=1, tickfont=dict(size=font_size)),
            xaxis=dict(
                showline=True,
                showticklabels=True,
                ticks='outside',
                tickfont=dict(size=font_size),
                showgrid=False,
            ),
        )

        return fig

    def shap_plot_plotly(
        self,
        shap_values: 'shap.Explanation',
        feature_names: list[str],
        y: pd.Series | None = None,
        **kwargs,
    ) -> go.Figure:
        """Wrapper: Waterfall plot with Plotly slider to choose instance index."""

        if not isinstance(shap_values, shap.Explanation):
            raise TypeError('Expected a shap.Explanation object.')
        if y is None:
            raise ValueError('Target not found')

        n_samples = shap_values.values.shape[0]

        # Сохраним все traces от всех фигур
        all_traces = []
        all_layouts = []
        traces_per_sample = None

        for i in range(n_samples):
            fig = self.shap_plot_one(
                shap_values=shap_values[i],
                base_value=shap_values.base_values[i],
                feature_names=feature_names,
                y=y.iloc[i],
                **kwargs,
            )

            # сохраняем layout
            all_layouts.append(fig.layout)

            if traces_per_sample is None:
                traces_per_sample = len(fig.data)

            for trace in fig.data:
                trace.visible = i == 0
                all_traces.append(trace)

        sliders = [
            {
                'active': 0,
                'currentvalue': {'prefix': 'Row index: '},
                'pad': {'t': 50},
                'steps': [
                    {
                        'label': str(i),
                        'method': 'update',
                        'args': [
                            {
                                'visible': [
                                    i * traces_per_sample <= j < (i + 1) * traces_per_sample
                                    for j in range(len(all_traces))
                                ],
                            },
                            all_layouts[i].to_plotly_json(),
                        ],
                    }
                    for i in range(n_samples)
                ],
            }
        ]

        final_fig = go.Figure(data=all_traces)
        final_fig.update_layout(**all_layouts[0].to_plotly_json())  # по умолчанию показывается layout первого элемента
        final_fig.update_layout(sliders=sliders)

        return final_fig
