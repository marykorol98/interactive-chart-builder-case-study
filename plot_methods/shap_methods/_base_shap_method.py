from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd
from lazy_imports import try_import
import plotly
from sklearn.base import ClassifierMixin

from smile_ml_core.data.exceptions import PlottingParameterTypeException, PlottingSmallNumberValuesExceptions
from smile_ml_core.data.structures import DictDataFrame
from smile_ml_core.models._ml_model_protocol import MLModel
from smile_ml_core.plot_methods import BaseMethod
from smile_ml_core.plot_methods.base_method import convert_ndarray
from smile_ml_core.plot_methods.styles.style_modification import StyleModificator

# Патчим np.bool, если отсутствует
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int

with try_import() as shap_import:
    import shap
    from shap import Explanation


class PlotBaseShapMethod(BaseMethod, ABC):
    styles_fields_exclude = ['scheme_color', 'marker_size']
    min_values_count = 10
    xaxis_title_default = 'SHAP Values'
    yaxis_title_default = 'Features'

    MAX_DISPLAY_DEFAULT = 10
    RANDOM_STATE_DEFAULT = 42
    N_SAMPLES = 100  # рекомендуется примерно такой минимальный размер рекомендательного сэмпла.
    # Чем больше это число, тем быстрее и качественнее сработает эксплейнер

    columns_templates = [
        {
            'label': 'Max features displayed',
            'name': 'max_display',
            'source': 'max_display',
            'default': MAX_DISPLAY_DEFAULT,
        },
        {
            'label': 'Random State',
            'name': 'random_state',
            'source': 'random_state',
            'default': RANDOM_STATE_DEFAULT,
        },
    ]

    params_description: dict[str, str] = {
        'max_display': 'Максимальное количество отображаемых признаков',
        'random_state': 'random state для shap сэмплера',
    }

    styles_modificators = [
        StyleModificator(id_='data_slice', value='200'),
    ]

    def __init__(self):
        super().__init__()

        self.max_display: int | None = None
        self.random_state: int | None = None
        self.df_shape: tuple[int, int] = ()

    def update_properties(self, properties: dict[str, Any]):
        self.max_display = properties.pop('max_display', self.MAX_DISPLAY_DEFAULT)
        self.random_state = properties.pop('random_state', self.RANDOM_STATE_DEFAULT)

        if self.max_display is not None and type(self.max_display) is not int:
            raise PlottingParameterTypeException('max_display', type(self.max_display), ['int'])

        if self.random_state is not None and type(self.random_state) is not int:
            raise PlottingParameterTypeException('random_state', type(self.random_state), ['int'])

    def plot(
        self,
        ddf: DictDataFrame,
        properties: dict[str, Any],
        styles: dict[str, Any],
        **kwargs,
    ):
        shap_import.check()
        self.apply_plotly_configs(properties, styles)

        cover_model = self.validate_model(kwargs.get('model'))
        model = cover_model.instance_model
        x_columns = model.feature_names_in_

        self._target_column = cover_model._target_column

        y = ddf.view()[self._target_column]
        ddf, y = cover_model._prepare_fit_data(ddf, y)

        # Необходимо для применения data_slice, если он указан
        # [0][1] берём, потому что get_series возвращает list[tuple[None, Any]]
        df = self.get_series(ddf.view(), kwargs, x_columns)[0][1]
        y_reduced = y.iloc[df.index]
        self.df_shape = df.shape

        shap_values = self.generate_shap_values(df, model, self.random_state)

        fig = self.shap_plot_plotly(shap_values, df.columns, indices=df.index.tolist(), y=y_reduced, **kwargs)

        layout = self.compile_layout()

        fig.update_layout(layout)

        return convert_ndarray(fig.to_plotly_json())

    def generate_shap_values(self, df: pd.DataFrame, model: MLModel, random_state: int) -> 'Explanation':
        if df.shape[0] < self.min_values_count:
            raise PlottingSmallNumberValuesExceptions(values_count=df.shape[0], min_count=self.min_values_count)

        # Приводим bool → int, если есть
        for col in df.select_dtypes(include='bool').columns:
            df[col] = df[col].astype(int)

        X_instance = shap.utils.sample(df, nsamples=min(len(df), self.N_SAMPLES), random_state=random_state)

        predict_func = model.predict

        # Проверим, классификация ли это
        is_classification = isinstance(model, ClassifierMixin)
        if is_classification:
            predict_func = model.predict_proba

        try:
            explainer = shap.Explainer(predict_func, X_instance)
            shap_values = explainer(df)

        except AssertionError:
            raise RuntimeWarning(
                'SHAP explanation завершился с ошибкой. Вероятной причиной является запуск сервиса calculations (uvicorn) в режиме отладки'
            )

        if (
            is_classification
        ):  # если True, то надо преобразовать результат и оставить только относящиеся к выбранном классу значения
            shap_values = self.extract_shap_for_predicted_class(shap_values)

        self.validate_shap_values(shap_values)

        return shap_values

    def validate_shap_values(self, shap_values: 'Explanation'):
        shap_values_shape = shap_values.values.shape
        error_msg = f'Shap значения сгенерированы некорректно. Текущий shape {shap_values_shape} не соответсвует формату (n_samples, n_features)'
        # Должно быть 2 измерения
        if shap_values_shape != self.df_shape:
            raise ValueError(error_msg)

    def extract_shap_for_predicted_class(self, shap_values: 'Explanation') -> 'Explanation':
        """
        Извлекает SHAP-значения только для предсказанного класса в задачах многоклассовой классификации.
        Возвращает новый shap.Explanation формы (n_samples, n_features).
        """
        values = shap_values.values  # (n_samples, n_features, n_classes) без predict
        base_values = shap_values.base_values  # (n_samples, n_classes) без predict
        data = shap_values.data
        feature_names = shap_values.feature_names
        output_names = shap_values.output_names

        if values.ndim == 2:
            # SHAP уже в виде (n_samples, n_features) — просто вернуть как есть
            return shap_values

        # Найдём индекс класса с наибольшей суммой SHAP + base_value — т.е. предсказанный класс
        predicted_classes = np.argmax(values.sum(axis=1) + base_values, axis=1)  # shape: (n_samples,)

        # Собираем SHAP values только для предсказанного класса
        selected_values = np.array([values[i, :, predicted_classes[i]] for i in range(values.shape[0])])

        selected_base_values = np.array([base_values[i, predicted_classes[i]] for i in range(base_values.shape[0])])

        return shap.Explanation(
            values=selected_values,
            base_values=selected_base_values,
            data=data,
            feature_names=feature_names,
            output_names=output_names,
        )

    @abstractmethod
    def shap_plot_plotly(
        self,
        shap_values: 'Explanation',
        feature_names: list[str],
        indices: list[int],
        y: pd.Series | None = None,
        **kwargs,
    ) -> plotly.graph_objs.Figure:
        raise NotImplementedError()

    @classmethod
    def get_columns(
        cls,
        instance: Any | None = None,
        **kwargs,
    ):
        selected_max_display = instance.max_display if instance else cls.MAX_DISPLAY_DEFAULT
        selected_random_state = instance.random_state if instance else cls.RANDOM_STATE_DEFAULT

        columns_info = {'max_display': selected_max_display, 'random_state': selected_random_state}

        return columns_info

    @classmethod
    def check_to_show(cls, *args, data_types: Any = None, columns=None, **kwargs):
        columns = cls.filter_dtypes_columns(data_types, methods=['number'], columns=columns)
        return len(columns) != 0
