from typing import Any

import pandas as pd
from lazy_imports import try_import
from pandas._libs.tslibs.timestamps import Timestamp

from smile_ml_core.plot_methods.base_method import BaseMethod
from smile_ml_core.data.exceptions import PlottingParametersException, PlottingAbsentColumnExceptions
from smile_ml_core.data.structures import DictDataFrame
from smile_ml_core.data.tools import try_to_number
from smile_ml_core.plot_methods.styles.style_modification import StyleModificator
import plotly.express as px

with try_import() as scipy_import:
    from scipy.stats import pearsonr


class WindowCorrelationPlot(BaseMethod):
    exclude_types = {'object'}

    styles_fields_exclude = {
        'marker_color',
    }

    # С этими дефолтами, как правило, получаются более наглядные графики
    DEFAULT_WINDOW_SIZE = 10
    DEFAULT_THRESHOLD = 0.2

    columns_templates = [
        {
            'label': 'TimeIndex source',
            'name': 'index_column',
            'source': 'index_column',
            'select_all': True,
            'default': 'index',
        },
        {
            'label': 'x1',
            'name': 'x_1',
            'source': 'x_1',
            'multi': False,
        },
        {
            'label': 'x2',
            'name': 'x_2',
            'source': 'x_2',
            'multi': False,
        },
        {
            'label': 'Window size',
            'name': 'window_size',
            'source': 'window_size',
            'default': DEFAULT_WINDOW_SIZE,
        },
        {
            'label': 'Correlation threshold',
            'name': 'corr_threshold',
            'source': 'corr_threshold',
            'default': DEFAULT_THRESHOLD,
        },
    ]

    description: str = """
        График отображает динамику корреляций. Каждому временному окну соответствует горизонтальный отрезок, 
        расположенный на уровне средней корреляции признаков за данный интервал.
    """

    params_description: dict[str, str] = {
        'index_column': 'Столбец с временными метками (`timeseries`), используемый в качестве индекса и отображаемый на оси X',
        'x_1': 'Первый признак для рассчёта корелляции',
        'x_2': 'Второй признак для рассчёта корелляции',
        'window_size': 'Размер временного окна (интервала), измеряемый в шагах временного ряда',
        'corr_threshold': """Граничное значение корреляции. Все участки графика, где корреляция ниже этого значения, 
        подсвечиваются цветом""",
    }

    styles_modificators = [
        StyleModificator(
            id_='scheme_color',
            value=BaseMethod.COLOR_SCHEME_DEFAULT,
            options={scheme: scheme for scheme in BaseMethod.COLOR_SCHEMES_DISCRETE},
        ),
    ]

    def __init__(self):
        super().__init__()

        self.x1_column: str | None = None
        self.x2_column: str | None = None

        self.window_size: int = self.DEFAULT_WINDOW_SIZE
        self.corr_threshold: float = self.DEFAULT_THRESHOLD
        self.xaxis_name_default = 'Data'
        self.yaxis_name_default = 'Correlation coefficient'

    @staticmethod
    def timestamp_to_index(
        time_index: pd.Series | list[Timestamp],
    ) -> tuple[list[int], dict[Timestamp, int]]:
        """
        Функция для перевода Timestamp в порядковые индексы

        Parameters:
        -----------
        time_index : array-like
            Временной индекс

        Returns:
        --------
        int_index: list
            Целочисленный индекс
        time_dict: dict
            Словарь для обратной конвертации
        """
        # создаём словарь дат
        all_dates = sorted(time_index)
        time_dict = {key: value for (key, value) in enumerate(all_dates)}
        time_to_int_dict = {
            value: key for key, value in time_dict.items()
        }  # словарь для обратной конвертации на следующих итерациях
        int_index = [time_to_int_dict[date] for date in time_index]

        return sorted(int_index), time_dict

    @staticmethod
    def bad_corr_periods_search(
        corr_dict: dict[int, dict[str, Any]],
    ) -> list[list[int]]:
        """
        Нахождение участков с низкой корреляцией для дальнейшего отображения на графике

        Parameters:
        -----------
        corr_dict : dict
            Словарь с корреляциями, где ключи - индексы окон с низкой корреляцией, значения - левые и правые границы окон.

        Returns:
        --------
        bad_periods_filtered : list
            Список участков с корреляциями ниже минимального порога.
        """
        if not corr_dict:
            return []  # Возвращаем пустой список, если нет плохих периодов

        bad_periods = []
        indices = list(corr_dict.keys())

        period = [indices[0]]
        for i in range(len(indices) - 1):
            if indices[i + 1] - indices[i] > 1:
                bad_periods.append([corr_dict[period[0]]['s'], corr_dict[period[-1]]['f']])
                period = []
            period.append(indices[i + 1])

        bad_periods.append([corr_dict[period[0]]['s'], corr_dict[period[-1]]['f']])  # the last iteration

        # Теперь объединим все пересекающиеся участки
        previous = bad_periods[0]
        bad_periods_filtered = []

        if len(bad_periods) > 1:
            for start, finish in bad_periods[1:]:
                if start in range(*previous) or start == previous[1]:
                    previous[1] = finish if finish > previous[1] else previous[1]
                else:
                    bad_periods_filtered.append(previous)
                    previous = [start, finish]

        bad_periods_filtered.append(previous)

        return bad_periods_filtered

    def update_properties(self, properties: dict[str, Any]):
        self.x1_column = properties.get('x_1')
        self.x2_column = properties.get('x_2')

        if self.x1_column is None:
            raise PlottingAbsentColumnExceptions('x1')

        if self.x2_column is None:
            raise PlottingAbsentColumnExceptions('x2')

        if self.x1_column == self.x2_column:
            raise PlottingParametersException('Column values must not be the same')

        self.index_column = properties.get('index_column', 'index')
        if self.index_column is None:
            raise PlottingAbsentColumnExceptions('index_column')

        if window_size := properties.get('window_size'):
            window_size = try_to_number(window_size)
            if not isinstance(window_size, int):
                raise PlottingParametersException('window_size must be an integer')
            if window_size <= 1:
                raise PlottingParametersException('Window size should be greater than 1')

            self.window_size = window_size

        corr_threshold = properties.get('corr_threshold', self.DEFAULT_THRESHOLD)

        if not isinstance(corr_threshold, (int, float)):
            raise PlottingParametersException('corr_threshold must be a float')
        self.corr_threshold = corr_threshold

    def plot(self, ddf: DictDataFrame, properties: dict[str, Any], styles: dict[str, Any], **kwargs) -> dict[str, Any]:
        scipy_import.check()

        self.update_styles(styles=styles)
        self.update_properties(properties=properties)

        df = ddf.view()

        if self.index_column != 'index':
            df = df.set_index(self.index_column)

        self.check_columns_type(df, [self.x1_column, self.x2_column])

        df_reduced = self.get_series(df, kwargs, [self.x1_column, self.x2_column])
        df_reduced = df_reduced[0][1].sort_index()

        traces = []
        corrs = []
        corr_low = {}

        # Проверка типа индекса:
        if isinstance(df_reduced.index[0], pd.Timestamp):
            int_index, time_dict = self.timestamp_to_index(df_reduced.index)
        else:
            int_index, time_dict = df_reduced.index, {}

        marker_colors = getattr(px.colors.qualitative, self.color_scheme)

        for i in range(len(df_reduced) - self.window_size):
            showlegend = True if i == 0 else False
            window = slice(i, i + self.window_size)
            corr = pearsonr(df_reduced[self.x1_column][window], df_reduced[self.x2_column][window])[
                0
            ]  # рассчёт корелляции
            corrs.append(corr)

            if corr <= self.corr_threshold:
                corr_low[i] = {'s': int_index[i], 'f': int_index[i + self.window_size]}
            traces.append(
                {
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Window correlation',
                    'line': {'width': 2, 'color': marker_colors[0]},
                    'x': df_reduced.index[window].tolist(),
                    'y': [corr] * self.window_size,
                    'hovertemplate': str(corr),
                    'showlegend': showlegend,
                }
            )

        # Теперь подсветим участки с корреляцией ниже, чем corr_threshold
        bad_corr_periods = self.bad_corr_periods_search(corr_low) if corr_low else []

        if time_dict:
            # Конвертируем обратно в Timestamp формат
            bad_corr_periods = [[time_dict[i], time_dict[j]] for (i, j) in bad_corr_periods]

        corr_min, corr_max = min(corrs), max(corrs)
        if len(bad_corr_periods) != 0:
            for i, bad_period in enumerate(bad_corr_periods):
                showlegend = True if i == 0 else False
                traces.append(
                    {
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'X1 and X2 window correlation <=' + str(self.corr_threshold),
                        'line': {'color': marker_colors[1], 'width': 0.1},
                        'fill': 'toself',
                        'fillcolor': marker_colors[1],
                        'opacity': 0.3,
                        'x': [
                            bad_period[0],
                            bad_period[0],
                            bad_period[1],
                            bad_period[1],
                            bad_period[0],
                        ],
                        'y': [corr_min, corr_max, corr_max, corr_min, corr_min],
                        'legendgroup': 'X1 and X2 window correlation <=' + str(self.corr_threshold),
                        'showlegend': showlegend,
                    }
                )

        layout = self.compile_layout()
        self.xaxis_name_default = 'Data'
        self.yaxis_name_default = 'Correlation coefficient'

        # Возвращаем данные и разметку
        return self.clean_nan_from_plot_data({'data': traces[::-1], 'layout': layout})

    @classmethod
    def get_columns(cls, data_types=None, columns=None, instance=None, **kwargs):
        num_columns = cls.filter_not_proba_columns(
            cls.filter_dtypes_columns(data_types, methods=['number'], columns=columns)
        )
        ts_columns = ['index'] + cls.filter_dtypes_columns(data_types, methods=['datetime'], columns=columns)

        if instance:
            selected_index_column = instance.index_column
            selected_column1, selected_column2 = instance.x_1, instance.x_2
            window_size = instance.window_size
            corr_threshold = instance.corr_threshold

        else:
            selected_index_column = ts_columns[-1]
            selected_column1 = num_columns[0] if num_columns else None
            selected_column2 = num_columns[1] if len(num_columns) > 1 else None
            window_size, corr_threshold = cls.DEFAULT_WINDOW_SIZE, cls.DEFAULT_THRESHOLD

        index_column = dict(map(lambda col: (col, col == selected_index_column), ts_columns))
        column1 = dict(map(lambda col: (col, col == selected_column1), num_columns))
        column2 = dict(map(lambda col: (col, col == selected_column2), num_columns))

        return {
            'index_column': index_column,
            'x_1': column1,
            'x_2': column2,
            'window_size': window_size,
            'corr_threshold': corr_threshold,
        }

    @classmethod
    def check_to_show(cls, *args: Any, columns=None, **kwargs: Any) -> Any:
        if columns is None:
            columns = []

        data_types = kwargs.get('data_types', {})
        columns_dtype_filtered = cls.filter_dtypes_columns(
            data_types,
            methods=['datetime'],
            columns=['index'] + columns,
        )
        number_columns = cls.filter_dtypes_columns(data_types, methods=['number'], columns=columns)

        return len(columns_dtype_filtered) > 0 and len(number_columns) > 1
