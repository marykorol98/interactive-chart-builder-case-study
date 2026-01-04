import pandas as pd
import plotly
from lazy_imports import try_import

from smile_ml_core.plot_methods.shap_methods import PlotBaseShapMethod

import numpy as np
import plotly.graph_objects as go

from smile_ml_core.plot_methods.styles.style_modification import StyleModificator

with try_import():
    from shap import Explanation


class ShapBar(PlotBaseShapMethod):
    xaxis_title_default = 'SHAP Value'
    yaxis_title_default = 'Feature'

    styles_modificators = getattr(PlotBaseShapMethod, 'styles_modificators', []) + [
        StyleModificator(id_='xaxis_title', value=xaxis_title_default),
        StyleModificator(id_='yaxis_title', value=yaxis_title_default),
        StyleModificator(id_='marker_color', value='red'),
    ]

    description = """
        Этот график визуализирует значения SHAP (SHapley Additive exPlanations) для каждого признака вашей модели, рассчитанные при помощи библиотеки shap. Значения SHAP показывают влияние каждого признака на предсказание модели, при этом более высокие положительные значения указывают на более значительное влияние на предсказанный результат, а отрицательные значения — на обратное влияние.

        Признаки расположены по убыванию их среднего абсолютного значения SHAP, то есть самые влиятельные признаки отображаются в верхней части графика. На графике также есть линия cut-off, которая представляет медиану значений SHAP. Признаки, значения SHAP которых выше этой медианы, считаются более влиятельными.

        Признаки группируются с помощью иерархической кластеризации, и на графике добавлены линии (скобки), указывающие на группы признаков, которые оказывают схожее влияние на предсказания. Эти группы могут помочь выявить корреляции между признаками, которые влияют на результат модели аналогичным образом.

        Интерпретация:
        - Чем длиннее полоса, тем более влиятельным является признак.
        - Положительные значения SHAP увеличивают предсказание модели, а отрицательные — уменьшают.
        - Линия cut-off помогает определить признаки, которые оказывают наибольшее влияние на модель.
        - Квадратные скобки (если есть) представляют группы признаков, которые оказывают схожее влияние на предсказание.
    """

    def shap_plot_plotly(
        self,
        shap_values: 'Explanation',
        feature_names: list[str],
        indices: list[int] | None = None,
        y: pd.Series | None = None,
        **kwargs,
    ) -> plotly.graph_objs.Figure:
        # Вычисляем среднее абсолютное значение SHAP по признакам
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        mean_abs_series = pd.Series(mean_abs_shap, index=feature_names)
        top_features = mean_abs_series.sort_values(ascending=False).head(self.max_display)

        # Ограничим shap_df только top_n признаками
        top_feature_names = top_features.index.tolist()
        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)[top_feature_names]
        if y is not None:
            shap_df['target'] = y

        # Построение bar-графика
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=top_features.values[::-1],
                y=top_features.index[::-1],
                orientation='h',
                marker=dict(color=self.marker_color),
                # генерируем подписи для значений SHAP, как на оригинальном графике:
                # добавляем плюсы к положительным, а отрицательные оставляем, как есть
                text=[f'+{val:.3f}' if val > 0 else f'{val:.3f}' for val in top_features.values[::-1]],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Impact score: %{x:.4f}<extra></extra>',
            )
        )

        # Cut-off линия
        cutoff = top_features.median()
        fig.add_shape(
            type='line',
            x0=cutoff,
            x1=cutoff,
            y0=0,
            y1=1,
            xref='x',
            yref='paper',
            line=dict(color='black', width=2, dash='dash'),
        )
        fig.add_annotation(
            x=cutoff,
            y=1.02,  # немного выше графика
            xref='x',
            yref='paper',
            text=f'clustering cutoff = {cutoff:.3f}',
            showarrow=False,
            font=dict(color='black', size=12),
            align='center',
        )

        fig.update_layout(template='plotly_white', showlegend=False, margin=dict(l=100, r=100))

        return fig
