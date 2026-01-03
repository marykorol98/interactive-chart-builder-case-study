import numpy as np
import pandas as pd
from lazy_imports import try_import
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

from smile_ml_core.plot_methods.shap_methods import PlotBaseShapMethod
from smile_ml_core.plot_methods.styles.style_modification import StyleModificator

with try_import() as shap_import:
    from shap import Explanation


class BeesWarm(PlotBaseShapMethod):
    styles_fields_exclude = ['marker_color']
    xaxis_title_default = 'SHAP Value (Inpact on Model Output)'
    yaxis_title_default = ''
    COLOR_SCHEME_DEFAULT = 'shap_original'
    COLOR_SCHEMES_CONTINUOUS = [COLOR_SCHEME_DEFAULT] + PlotBaseShapMethod.COLOR_SCHEMES_CONTINUOUS

    styles_modificators = PlotBaseShapMethod.styles_modificators + [
        StyleModificator(id_='xaxis_title', value=xaxis_title_default),
        StyleModificator(
            id_='scheme_color', value=COLOR_SCHEME_DEFAULT, options={color: color for color in COLOR_SCHEMES_CONTINUOUS}
        ),
    ]
    description = """
        Beeswarm-график для визуализации SHAP-значений по важности признаков.

        Этот график отображает распределение SHAP-значений по каждому из топ-признаков (по средней абсолютной важности),
        позволяя визуально оценить вклад каждого признака в предсказание модели. Каждый маркер на графике представляет
        собой один объект выборки, где:
        
        - X-ось: SHAP-значение (влияние признака на предсказание).
        - Y-ось: Названия признаков (сортировка по важности).
        - Цвет маркера: Значение признака, нормированное от минимума к максимуму.

        Цветовая шкала по умолчанию имитирует оригинальную палитру SHAP — от голубого (низкое значение признака) 
        к розовому (высокое значение). Также поддерживаются стандартные цветовые схемы Plotly.

        Подходит для моделей, где важна интерпретируемость и объяснение вклада признаков.
    """

    def shap_plot_plotly(
        self,
        shap_values: 'Explanation',
        feature_names: list[str],
        indices: list[int],
        y: pd.Series | None = None,
        **kwargs,
    ) -> go.Figure:
        values = np.copy(shap_values.values)  # shape: (n_samples, n_features)
        features = shap_values.data  # shape: (n_samples, n_features)
        feature_names = shap_values.feature_names

        if isinstance(features, pd.DataFrame):
            features = features.values

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(values.shape[1])]

        mean_abs_shap = np.abs(values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-self.max_display :][::-1]

        fig = go.Figure()
        row_height = 1
        scaler = MinMaxScaler(feature_range=(0, 1))

        # original SHAP colors: from deep pink/magenta to light blue
        shap_colorscale = [
            [0.0, '#1E88E5'],
            [1.0, '#FF0052'],
        ]

        if self.color_scheme != self.COLOR_SCHEME_DEFAULT:
            shap_colorscale = px.colors.get_colorscale(self.color_scheme)

        for pos, idx in enumerate(top_indices):
            shaps = values[:, idx]
            vals = features[:, idx] if features is not None else np.zeros_like(shaps)
            # Normalize color values
            color_vals = scaler.fit_transform(vals.reshape(-1, 1)).flatten()

            # джиттер по оси Y для формирования "роя" точек (beeswarm)
            quant = np.round(100 * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(len(quant)) * 1e-6)
            ys = np.zeros(len(shaps))
            layer = 0
            last_bin = -1
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.4 * (row_height / (np.max(np.abs(ys)) + 1e-8))
            marker_size = self.styles.get('marker_size').value

            colorbar_config = dict(
                title=dict(
                    text='Feature value (normalized)',
                    side='right',
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=shaps,
                    y=pos + ys,
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=color_vals,
                        colorscale=shap_colorscale,
                        cmin=0,
                        cmax=1,
                        colorbar=colorbar_config,
                        showscale=(pos == 0),
                        opacity=0.9,
                    ),
                    name=feature_names[idx],
                    showlegend=False,
                    hoverinfo='name+text',
                    text=[
                        f'Element index: {i}<br>Feature Value: {feat:.2f}<br>Target Value: {target}<br>SHAP Value: {shap:.2f}'
                        for (i, feat, shap, target) in zip(indices, vals, shaps, y)
                    ],
                )
            )

        fig.update_layout(
            template=None,
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(top_indices))),
                ticktext=[feature_names[i] for i in top_indices],
                autorange='reversed',
                zeroline=False,
            ),
            xaxis=dict(
                showline=True,
                showticklabels=True,
                ticks='outside',
                linecolor='black',
                linewidth=1,
            ),
        )
        fig.update_yaxes(showgrid=True, gridcolor='#eeeeee')
        fig.update_xaxes(showgrid=False)

        return fig
