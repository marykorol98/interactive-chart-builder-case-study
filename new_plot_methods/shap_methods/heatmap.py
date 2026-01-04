import pandas as pd
import plotly.graph_objects as go
from lazy_imports import try_import

from smile_ml_core.plot_methods.shap_methods import PlotBaseShapMethod
from smile_ml_core.plot_methods.styles.style_modification import StyleModificator
from plotly.subplots import make_subplots

with try_import() as plotly_import:
    import shap


class HeatMap(PlotBaseShapMethod):
    description = """
    Данный график представляет собой тепловую карту SHAP значений (SHapley Additive exPlanations) для различных признаков модели и объектов (экземпляров данных).

    Назначение графика:
    График предназначен для оценки влияния каждого признака на предсказания модели для каждого отдельного наблюдения. Это особенно полезно при анализе поведения модели на уровне отдельных объектов, а также для выявления закономерностей в том, как признаки воздействуют на выход модели.

    Как читать график:
    - По оси Y отображаются признаки (features).
    - По оси X — отдельные объекты (samples) из набора данных.
    - Цвет ячейки показывает значение SHAP, например при дефолтной палитре RdBu:  
      - Красные/теплые тона (положительные значения SHAP) означают, что признак увеличивает значение предсказания.  
      - Синие/холодные тона (отрицательные значения SHAP) означают, что признак уменьшает значение предсказания.  
      - Чем интенсивнее цвет, тем сильнее влияние признака.
    - Центр цветовой шкалы (`zmid=0`) соответствует нулевому влиянию, когда признак не оказывает эффекта на предсказание.
    - Кривая функции над тепловой картой (`f(x)`) — предсказания модели для каждого объекта.

    Преимущества такого представления:
    - Позволяет увидеть, какие признаки влияют на результат модели и в каких наблюдениях.
    - Удобно выявлять аномальные паттерны, например, признаки, оказывающие значительное влияние только в отдельных случаях.
    - Помогает проводить групповой анализ и находить кластеры объектов с похожими объяснениями.
    - Подходит для диагностики модели, в том числе для анализа смещений и проверки интерпретируемости.

    Пример использования:
    Тепловая карта особенно полезна при анализе сложных моделей (например, градиентного бустинга или нейросетей), когда необходимо понять, почему модель принимает те или иные решения на уровне отдельных строк данных.
    """
    MAX_DISPLAY_DEFAULT = 12

    columns_templates = [
        {
            'label': 'Max features displayed',
            'name': 'max_display',
            'source': 'max_display',
            'default': MAX_DISPLAY_DEFAULT,
        },
    ]

    params_description: dict[str, str] = {'max_display': 'Максимальное количество отображаемых признаков'}

    styles_fields_exclude = ['marker_color', 'marker_size']

    xaxis_title_default = 'Instances'
    yaxis_title_default = 'Features'

    COLOR_SCHEME_DEFAULT = 'shap_original'
    COLOR_SCHEMES_CONTINUOUS = [COLOR_SCHEME_DEFAULT] + PlotBaseShapMethod.COLOR_SCHEMES_CONTINUOUS

    styles_modificators = getattr(PlotBaseShapMethod, 'styles_modificators', []) + [
        StyleModificator(id_='xaxis_title', value=xaxis_title_default),
        StyleModificator(id_='yaxis_title', value=yaxis_title_default),
        StyleModificator(
            id_='scheme_color', value=COLOR_SCHEME_DEFAULT, options={color: color for color in COLOR_SCHEMES_CONTINUOUS}
        ),
    ]

    def shap_plot_plotly(
        self,
        shap_values,
        feature_names: list[str],
        indices: list[int],
        y: pd.Series | None = None,
        **kwargs,
    ) -> go.Figure:
        # Получаем массив SHAP значений
        if isinstance(shap_values, shap.Explanation):
            shap_array = shap_values.values
            feature_names = shap_values.feature_names or feature_names
        else:
            shap_array = shap_values

        # Преобразуем в DataFrame для удобства
        shap_df = pd.DataFrame(shap_array, columns=feature_names).iloc[:, : self.max_display]
        shap_matrix = shap_df.T.values  # транспонируем, чтобы фичи были по оси Y

        # Предсказания модели f(x)
        # Calculate model predictions f(x)
        if isinstance(shap_values, shap.Explanation):
            fx = shap_array.sum(axis=1) + shap_values.base_values  # сумма SHAP по фичам + базовое значение SHAP
        else:
            fx = shap_array.sum(axis=1)  # базовое значение недоступно для обычных массивов

        # Создаём подграфики: 2 строки, 1 колонка
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.3, 0.7],
            vertical_spacing=0.02,
        )

        # Преобразуем метки y в текст, если они переданы
        # Align labels to every point drawn
        if y is not None:
            if not indices:  # empty → use sequential order
                indices = list(range(len(fx)))
            y_labels = y.iloc[indices].astype(str).tolist()
            # Fallback for mismatch
            y_labels = (y_labels + [None] * len(fx))[: len(fx)]
        else:
            y_labels = [None] * len(fx)

        # График предсказаний f(x) — верхний
        fx_trace = go.Scatter(
            x=list(range(len(fx))),
            y=fx,
            mode='lines+markers',
            marker=dict(size=6, color='black'),
            line=dict(color='black', width=2),
            name='f(x)',
            text=[f'{v:.2f}' for v in fx],
            textposition='top center',
            customdata=[[label] for label in y_labels],
            hovertemplate='Instance index: %{x}<br>f(x) = %{text}<br>Label: %{customdata[0]}<extra></extra>',
        )

        fig.add_trace(fx_trace, row=1, col=1)

        # original SHAP colors: from deep pink/magenta to light blue
        shap_colorscale = [
            [0.0, '#FF0052'],
            [0.5, '#FFFFFF'],  # белый — нейтральное значение (0)
            [1.0, '#1E88E5'],
        ]

        # Тепловая карта SHAP значений — нижний
        heatmap = go.Heatmap(
            z=shap_matrix,
            x=list(range(shap_df.shape[0])),
            y=feature_names[: self.max_display],
            colorscale=self.color_scheme if self.color_scheme != self.COLOR_SCHEME_DEFAULT else shap_colorscale,
            reversescale=True,
            zmid=0,
            colorbar=dict(title='SHAP Value'),
            hoverongaps=False,
            hovertemplate='Feature: %{y}<br>Instance index: %{x}<br>SHAP: %{z}<extra></extra>',
        )

        fig.add_trace(heatmap, row=2, col=1)

        fig.update_yaxes(title_text='f(x)', showline=True, row=1, col=1)

        return fig

    def compile_layout(self, **kwargs) -> dict[str, str | dict]:
        layout = super().compile_layout()

        layout.update(
            dict(
                height=700,
                showlegend=False,
                margin=dict(l=80, r=40, t=50, b=40),
                xaxis2={'title': layout['xaxis'].pop('title')},
                yaxis2={'title': layout['yaxis'].pop('title')},
            )
        )

        return layout
