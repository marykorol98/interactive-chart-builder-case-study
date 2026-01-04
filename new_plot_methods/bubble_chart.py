from typing import Any

from lazy_imports import try_import
import pandas as pd
from smile_ml_core.data.exceptions import (
    PlottingAbsentColumnExceptions,
    PlottingParameterTypeException,
    PlottingParametersException,
)
from smile_ml_core.data.structures.dict_data_frame import DictDataFrame
from smile_ml_core.plot_methods.base_method import convert_ndarray
from smile_ml_core.plot_methods.scatter import Scatter
from smile_ml_core.plot_methods.styles.style_modification import StyleModificator

with try_import() as plotly_import:
    import plotly.express as px
    import plotly.graph_objects as go


class BubbleChart(Scatter):
    """
    График представляет собой Scatter, где размер точек, цвет и прозрачность зависят от выборного числового параметра.
    Хорошо подходит для кейсов, когда хочется отобразить сразу 3 признака.
    """

    styles_fields_exclude = {
        'line_type',
        'color',
        'marker_color',
    }

    description = """
        BubbleChart — это расширение диаграммы рассеяния, где каждая точка представляет собой объект с координатами (X, Y), а её размер и прозрачность отражают дополнительную информацию.

        📊 Что отображает график:
        - Координаты X и Y определяют позицию объекта.
        - Размер пузыря зависит от выбранного числового параметра (`Условие для размера точки`) или частоты появления точки (по умолчанию).
        - Цвет пузыря варьируется в зависимости от того же параметра.
        - Прозрачность может быть включена для более наглядного отображения плотности.

        🧠 Когда использовать:
        - Когда нужно визуализировать сразу 3 числовых признака: X, Y и дополнительный, влияющий на размер и цвет.
        - Когда важно показать плотность данных, различия между группами или выявить корреляции.
        - Особенно полезно для анализа распределения, кластера объектов и визуализации "перенасыщенных" точек.

        ⚙️ Параметры:
        - `Столбец X` и `Столбец Y` — определяют координаты объектов.
        - `Условие для размера точки` — числовой признак, который управляет размером (и цветом) пузырей. По умолчанию используется частота появления каждой точки.
        - `Включить условную прозрачность` — при включении делает плотные участки менее насыщенными, чтобы избежать визуального шума.

        🛠 Настройки визуализации:
        - `Размер маркера` позволяет задать максимальный размер пузыря.
        - Цветовая схема, заголовки осей и прочие параметры задаются отдельно.
        - Легенда отображается в виде трех пузырей с пояснением значений по размеру.

        ❗ Ограничения:
        - Все выбранные столбцы должны быть числовыми.
        - Столбцы `X`, `Y` и `Условие для размера` должны быть разными, иначе будет ошибка конфигурации.
        """

    params_description: dict[str, str] = {
        'column_x': 'Столбец, отражаемый по оси X',
        'column_y': 'Столбец, отражаемый по оси Y',
        'size_condition': 'Числовой признак, который управляет размером (и цветом) пузырей. По умолчанию используется частота появления каждой точки',
        'conditioned_opacity': 'При включении прозрачность меняется вместе с размером',
    }

    columns_templates = [
        {
            'label': 'Столбец X',
            'name': 'column_x',
            'source': 'column_x',
            'multi': False,
        },
        {
            'label': 'Столбец Y',
            'name': 'column_y',
            'source': 'column_y',
            'multi': False,
        },
        {
            'label': 'Условие для размера точки',
            'name': 'size_condition',
            'source': 'size_condition',
        },
        {
            'label': 'Включить условную прозрачность',
            'name': 'conditioned_opacity',
            'source': 'conditioned_opacity',
        },
    ]

    # Опция, когда размер точек зависит от частоты значений
    SIZE_CONDITION_DEFAULT = 'Frequency'
    CONDITIONED_OPACITY_DEFAULT = True
    MARKER_SIZE_DEFAULT = 15
    DATA_SLICE_DEFAULT = '100'

    styles_modificators = [
        StyleModificator(id_='marker_size', value=MARKER_SIZE_DEFAULT),
        StyleModificator(id_='data_slice', value=DATA_SLICE_DEFAULT),
        StyleModificator(
            id_='scheme_color', value='Viridis', options={color: color for color in Scatter.COLOR_SCHEMES_CONTINUOUS}
        ),
    ]

    def __init__(self):
        super().__init__()
        self.column_x: str = ''
        self.column_y: str = ''
        self.size_condition: str = self.SIZE_CONDITION_DEFAULT
        self.conditioned_opacity: bool = self.SIZE_CONDITION_DEFAULT

    def update_properties(self, properties: dict[str, Any]):
        self.column_x = properties.get('column_x', '')
        self.column_y = properties.get('column_y', '')
        if not self.column_x:
            raise PlottingAbsentColumnExceptions('column_x')
        if not self.column_y:
            raise PlottingAbsentColumnExceptions('column_y')

        self.size_condition = properties.get('size_condition', self.SIZE_CONDITION_DEFAULT)
        if not self.size_condition:
            self.size_condition = self.SIZE_CONDITION_DEFAULT
        if not isinstance(self.size_condition, str):
            raise PlottingParameterTypeException('size_condition', type(self.size_condition), ['str'])

        if len({self.column_x, self.column_y, self.size_condition}) < 3:
            raise PlottingParametersException('Column values must not be the same')

        self.conditioned_opacity = properties.get('conditioned_opacity', self.CONDITIONED_OPACITY_DEFAULT)
        if self.conditioned_opacity is None:
            self.conditioned_opacity = self.CONDITIONED_OPACITY_DEFAULT

        if not isinstance(self.conditioned_opacity, bool):
            raise PlottingParameterTypeException('conditioned_opacity', type(self.conditioned_opacity), ['bool'])

        self.xaxis_name_default = self.column_x
        self.yaxis_name_default = self.column_y

    def plot(self, ddf: DictDataFrame, properties: dict[str, Any], styles: dict[str, Any], **kwargs):
        self.apply_plotly_configs(properties, styles)

        df = self.validate_df(ddf)

        columns = [self.column_x, self.column_y]

        layout = self.compile_layout()

        cols_for_reducing = columns

        if self.size_condition != self.SIZE_CONDITION_DEFAULT:
            cols_for_reducing += [self.size_condition]

        df_reduced = self.get_series(df, kwargs, cols_for_reducing)
        df_reduced = df_reduced[0][1]

        freqs: pd.Series | None = None
        if df_reduced.isnull().values.any():
            df_reduced = df_reduced.dropna()

        # Since update 2.0.1 we can not group by float
        columns_float = df_reduced.select_dtypes('float').columns.tolist()
        df_transform = df_reduced.copy()
        if columns_float:
            df_transform[columns_float] = df_transform[columns_float].round(3).astype(str)

        if self.size_condition == self.SIZE_CONDITION_DEFAULT:
            freqs = df_transform.groupby(columns)[columns[0]].transform('count')
        else:
            freqs = df_transform[self.size_condition]

        size_max = styles.get('marker_size', self.MARKER_SIZE_DEFAULT)

        color_scheme = self.styles['scheme_color'].value

        fig = px.scatter(
            x=df_reduced[columns[0]].tolist(),
            y=df_reduced[columns[1]].tolist(),
            color=freqs,  # Цвет точек отражает их размер
            color_continuous_scale=color_scheme,
            size_max=size_max,
            size=freqs,
        )

        if not self.conditioned_opacity:
            # Отключаем автоматическое изменение прозрачности
            fig.update_traces(marker=dict(opacity=1.0), selector=dict(mode='markers'))

        # Координаты для размещения легенды в левом верхнем углу
        x_legend = min(df_reduced[columns[0]])  # Левый край
        y_legend = max(df_reduced[columns[1]])  # Верхний край

        # Определяем три характерных размера для легенды
        legend_sizes = set([min(freqs), (max(freqs) + min(freqs)) // 2, max(freqs)])  # Берем min, среднее и max

        # Масштабируем их в соответствии с размерностью точек на графике
        legend_marker_sizes = [s / max(freqs) * size_max for s in legend_sizes]  # max
        y_offset = (
            max(df_reduced[columns[1]]) - min(df_reduced[columns[1]])
        ) * 0.05  # Смещение между элементами легенды

        # Добавляем точки легенды
        for i, (size, marker_size) in enumerate(zip(legend_sizes, legend_marker_sizes)):
            colorscale = fig['layout']['coloraxis']['colorscale']
            fig.add_trace(
                go.Scatter(
                    x=[x_legend],
                    y=[y_legend - i * y_offset],  # Смещаем точки вниз
                    mode='markers',
                    marker=dict(size=marker_size, color=[freqs[i]], colorscale=colorscale),
                    showlegend=False,  # Чтобы не дублировать стандартную легенду
                )
            )

            # Добавляем подписи рядом с точками
            fig.add_annotation(
                x=x_legend + (max(df_reduced[columns[0]]) - min(df_reduced[columns[0]])) * 0.02,
                y=y_legend - i * y_offset,
                text=f'{self.size_condition}: {size:.1f}',
                showarrow=False,
                font=dict(size=styles.get('font_size', 12), color='black'),
                xanchor='left',
                yanchor='middle',
            )

        fig.update_coloraxes(dict(colorbar={'title': {'text': self.size_condition}}))
        fig.update_layout(layout)

        return convert_ndarray(fig.to_plotly_json())

    @classmethod
    def get_columns(cls, data_types=None, columns=None, instance=None, **kwargs):
        assert columns, 'no columns passed'
        assert data_types, 'no data_types passed'

        columns = cls.filter_dtypes_columns(data_types, methods=['number', 'bool'], columns=columns)

        selected_column_x = instance.column_x if instance else columns[0]
        selected_column_y = instance.column_y if instance else columns[1]
        selected_condition = instance.size_condition if instance else cls.SIZE_CONDITION_DEFAULT
        selected_opacity = instance.conditioned_opacity if instance else cls.CONDITIONED_OPACITY_DEFAULT

        column_x = dict(map(lambda col: (col, col == selected_column_x), columns))
        column_y = dict(map(lambda col: (col, col == selected_column_y), columns))

        condition_out = dict(map(lambda col: (col, col == selected_condition), [cls.SIZE_CONDITION_DEFAULT] + columns))

        return {
            'column_x': column_x,
            'column_y': column_y,
            'size_condition': condition_out,
            'conditioned_opacity': selected_opacity,
        }

    @classmethod
    def check_to_show(cls, data_types=None, columns=None, **kwargs):
        columns = cls.filter_dtypes_columns(data_types, methods=['number', 'bool'], columns=columns)
        return len(columns) >= 2
