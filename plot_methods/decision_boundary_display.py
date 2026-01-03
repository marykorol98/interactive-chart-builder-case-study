from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder

from smile_ml_core.data.exceptions import PlottingParametersException, PlottingAbsentColumnExceptions
from smile_ml_core.plot_methods.base_method import BaseMethod, convert_ndarray
from smile_ml_core.data.structures import DictDataFrame
from smile_ml_core.models._ml_model_protocol import MLModel
from smile_ml_core.plot_methods.styles.style_modification import StyleModificator

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


class DecisionBoundaryDisplay(BaseMethod):
    exclude_types = {'object', 'datetime64'}

    DEFAULT_GRID_RESOLUTION = 25
    columns_templates = [
        {
            'label': 'x1',
            'name': 'x_1',
            'source': 'x_1',
            'default': [],
            'multi': False,
        },
        {
            'label': 'x2',
            'name': 'x_2',
            'source': 'x_2',
            'default': [],
            'multi': False,
        },
        {
            'label': 'Heatmap grid resolution',
            'name': 'grid_resolution',
            'source': 'grid_resolution',
            'default': DEFAULT_GRID_RESOLUTION,
        },
    ]

    styles_fields_exclude = {'bg_color', 'line_type', 'marker_color', 'font_color'}

    styles_modificators = [
        StyleModificator(
            id_='scheme_color', value='Viridis', options={color: color for color in BaseMethod.COLOR_SCHEMES_CONTINUOUS}
        ),
    ]

    description = """
        График Decision Boundary Display визуализирует границы принятия решений классификационной модели. Он предназначен для демонстрации, как алгоритм машинного обучения разделяет пространство признаков, и помогает анализировать предсказания модели.

        Как интерпретировать график?

        - Фон окрашен в различные цвета, представляя области, которые модель классифицирует в разные классы.
        - Цветовая шкала подбирается автоматически в зависимости от количества классов. Если цветов в палитре недостаточно, отображается ошибка.
        - Каждая точка представляет объект из выборки.
        - Цвет точки соответствует её истинному классу.
        - Если есть предсказанные значения, они добавляются в виде аннотаций.
        - Переходы между цветами на фоне показывают границы, где модель меняет своё предсказание.
        - Чем плавнее границы, тем увереннее модель в своих решениях.
        - Если границы резкие, это может свидетельствовать о высокой сложности модели или недостатке данных.
        - Легенда располагается над графиком и группирует классы.
        - При наведении на точку отображаются её реальный класс и (если есть) предсказанное значение.

        Для чего используется этот график?
        - Оценка качества модели: помогает понять, насколько хорошо модель отделяет классы.
        - Диагностика ошибок: можно увидеть области, где модель делает ошибки, если предсказанные значения сильно отличаются от истинных.
        - Выбор модели: сравнивая границы решений у разных алгоритмов, можно выбрать наиболее подходящий.
        - Интерпретация признакового пространства: помогает понять, как модель использует признаки для классификации.

        Этот график особенно полезен в задачах бинарной и многоклассовой классификации.
    """

    params_description: dict[str, str] = {
        'x_1': 'Столбец, отображаемый по оси X',
        'x_2': 'Столбец, отображаемый по оси Y',
        'grid_resolution': 'Количество точек для построения фона (Heatmap). Чем выше значение, тем детализированнее фон, но увеличивается и время построения.',
    }

    def __init__(self):
        super().__init__()
        self.columns: list[str] = []
        self.x_1: str = ''
        self.x_2: str = ''
        self.target_column_name: str | None = None
        self.grid_resolution: int = self.DEFAULT_GRID_RESOLUTION

    def update_properties(self, properties: dict[str, Any]):
        self.x_1, self.x_2 = properties.get('x_1', None), properties.get('x_2', None)
        if not self.x_1:
            raise PlottingAbsentColumnExceptions('x_1')
        if not self.x_2:
            raise PlottingAbsentColumnExceptions('x_2')
        if self.x_1 == self.x_2:
            raise PlottingParametersException('Column values must not be the same')

        self.columns = [self.x_1, self.x_2]
        self.xaxis_name_default = self.x_1
        self.yaxis_name_default = self.x_2

        self.target_column_name = properties.get('target_column', None)
        if not self.target_column_name:
            raise PlottingAbsentColumnExceptions('target_column')

        # нам нужна только читаемая часть заголовка
        self.target_column_name = self.target_column_name.rsplit(':', 1)[-1]

        self.predict_column_name = properties.get('predict_column', None)

        self.grid_resolution = properties.get('grid_resolution', self.DEFAULT_GRID_RESOLUTION)

    def plot(self, ddf: DictDataFrame, properties: dict[str, Any], styles: dict[str, Any], **kwargs):
        self.apply_plotly_configs(properties, styles)

        df = ddf.view()

        cols_for_reducing = self.columns + [self.target_column_name]

        has_predict = self.predict_column_name in df.columns
        if has_predict:
            cols_for_reducing += [self.predict_column_name]

        self.check_columns_type(df, self.columns)

        df_reduced = self.get_series(df, kwargs, cols_for_reducing)
        df_reduced = df_reduced[0][1]

        X = df_reduced[self.columns]
        target = df_reduced[self.target_column_name]
        num_classes = len(target.unique())

        if num_classes <= 1:
            raise ValueError('Представлено слишком мало классов в данных. Увеличьте размер выборки (через data_slice)')

        color_scheme = self.styles['scheme_color'].value
        # получаем список цветов из выбранной палитры
        marker_colors = px.colors.get_colorscale(color_scheme)
        num_colors = len(marker_colors)

        if num_colors < num_classes:
            raise ValueError(
                f'В выбранной палитре не хватает цветов ({num_colors} шт) для всех уникальных классов ({num_classes} шт)'
            )

        x_min, x_max = (
            df_reduced[self.columns[0]].min() - 1,
            df_reduced[self.columns[0]].max() + 1,
        )
        y_min, y_max = (
            df_reduced[self.columns[1]].min() - 1,
            df_reduced[self.columns[1]].max() + 1,
        )

        x_, y_ = (
            np.linspace(x_min, x_max, num=self.grid_resolution),
            np.linspace(y_min, y_max, num=self.grid_resolution),
        )
        xx, yy = np.meshgrid(x_, y_)

        predictor: MLModel = self.validate_model(kwargs.get('model')).instance_model

        predictor.fit(X, target)

        z_predicted = predictor.predict(np.c_[xx.ravel(), yy.ravel()])
        colorbar = None
        enc = None
        if isinstance(target.values[0], str) or isinstance(target.values[0], bool):
            enc = LabelEncoder()
            enc.fit(target)
            z_encoded = enc.transform(z_predicted)
            colorbar = dict(tickvals=enc.transform(enc.classes_), ticktext=enc.classes_)
        else:
            z_encoded = z_predicted

        z_encoded = z_encoded.reshape(xx.shape)

        fig = make_subplots(rows=1, cols=1)

        display = go.Heatmap(
            x=x_,
            y=y_,
            z=z_encoded,
            showscale=True,
            showlegend=False,
            colorbar=colorbar,
            name='Predictions Heatmap',
            colorscale=color_scheme,
            xaxis='x',
            yaxis='y',
            customdata=np.array(z_predicted).reshape(z_encoded.shape),  # Убедимся, что форма совпадает
            hovertemplate='prediction: %{customdata}<extra></extra>',  # Добавляем customdata
        )

        fig.add_trace(display, row=1, col=1)

        marker_size = self.styles.get('marker_size', 10).value

        targets_enc = enc.transform(df_reduced[self.target_column_name]) if enc else df_reduced[self.target_column_name]

        annotations: list[str] = []
        for i, label in enumerate(df_reduced[self.target_column_name].values):
            class_name = label if isinstance(label, str) else str(label)

            # Индексы точек
            index_text = str(i)
            text = 'index: ' + index_text + '<br>target value: ' + class_name

            if has_predict:
                pred_text = str(df_reduced[self.predict_column_name].iloc[i])

                text += '<br>predicted value: ' + pred_text

            annotations.append(text)

        scatters = go.Scatter(
            x=df_reduced[self.columns[0]],
            y=df_reduced[self.columns[1]],
            mode='markers',
            showlegend=False,
            marker=dict(
                size=marker_size,
                color=targets_enc,
                colorscale=color_scheme,
                line=dict(color='black', width=1),
            ),
            text=annotations,  # Индивидуальные подписи
            hovertemplate='%{text}<extra></extra>',  # Полный текст в аннотации
        )
        fig.add_trace(scatters, row=1, col=1)

        layout = self.compile_layout()

        fig.update_layout(layout)

        return convert_ndarray(fig.to_plotly_json())

    @classmethod
    def get_columns(cls, data_types=None, columns=None, instance=None, **kwargs):
        columns = cls.filter_dtypes_columns(data_types, methods=['number'], columns=columns)
        columns = cls.filter_not_proba_columns(columns)

        if instance:
            selected_column1, selected_column2 = instance.x_1, instance.x_2
            selected_resolution = instance.grid_resolution
        else:
            selected_column1 = columns[0] if columns else None
            selected_column2 = columns[1] if len(columns) > 1 else None
            selected_resolution = cls.DEFAULT_GRID_RESOLUTION

        columns1 = dict(map(lambda col: (col, col == selected_column1), columns))
        columns2 = dict(map(lambda col: (col, col == selected_column2), columns))

        return {'x_1': columns1, 'x_2': columns2, 'grid_resolution': selected_resolution}

    @classmethod
    def check_to_show(cls, columns=None, data_types=None, **kwargs):
        columns = cls.filter_dtypes_columns(data_types, methods=['number', 'string'], columns=columns)
        return len(columns) >= 3  # must be at least 3 columns (2 regressors + target)
