import colorsys
from typing import Any, Optional

import numpy as np

from smile_ml_core.constants import TREE_MODELS
from smile_ml_core.data.exceptions import PlottingParametersException
from smile_ml_core.data.structures import DictDataFrame
from smile_ml_core.models import BaseModel
from smile_ml_core.models._ml_model_protocol import MLModel

from smile_ml_core.plot_methods.base_method import BaseMethod, convert_ndarray

import plotly.graph_objects as go


def hsl_to_hex(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))


class TreeGraphVisualizer(BaseMethod):
    styles_fields: list | str = ['font_size', 'title']
    MAX_DEPTH_DEFAULT = 3
    MAX_NODE_SIZE = 400

    columns_templates = [
        {
            'label': 'Tree index (if Forest)',
            'name': 'tree_index',
            'source': 'tree_index',
            'default': '',
        },
        {'label': 'Max depth', 'name': 'max_depth', 'source': 'max_depth', 'default': MAX_DEPTH_DEFAULT},
    ]

    description = """
    Визуализация представляет собой изображение дерева решений, построенного на основе обученной модели. Если модель является ансамблем (например, Random Forest), можно выбрать конкретное дерево по его индексу.

    📊 Что показывает граф:
    - Узлы дерева отображают условия, по которым данные разделяются на каждом уровне.
    - Ветви показывают направление разделения в зависимости от значений признаков.
    - Листовые узлы содержат итоговые предсказания класса и распределение по классам.
    - Цвета узлов соответствуют предсказанным классам, а насыщенность цвета указывает на уверенность модели в этом классе (чем насыщеннее — тем выше уверенность).
    - В легенде указано, какому классу соответствует каждый цвет узла.

    ⚙️ Пользовательские параметры:
    - Tree index — выбор конкретного дерева в ансамбле (например, в лесу из 100 деревьев можно выбрать 0–99).
    - Max depth — ограничение глубины дерева, полезно для фокусировки на ключевых уровнях логики.

    🧠 Как интерпретировать:
    - Чтение дерева начинается с корня (самый верхний узел) и движется вниз по ветвям.
    - Каждое условие в узле указывает, как делятся данные (например, `feature_3 <= 1.5`). 
    - Левый дочерний узел представляет собой выборку при кейсе, если условия выполняется со значением True, правый - со значением False.
    - Чем глубже узел, тем более специфичным становится разделение.
    - Конечные узлы (листы) всегда имеют вероятность, равную единице в одном из классов
    - В аннотациях при узлах (при наведении курсора) всегда можно увидеть вероятности каждого из классов.

    📌 Назначение:
    - Анализировать, как модель принимает решения.
    - Проверять, какие признаки наиболее важны.
    - Упрощённая отладка и объяснение модели в терминах логики.
    - Используется в Explainable AI (XAI) для визуального обоснования выводов модели.
    """

    params_description: dict[str, str] = {
        'tree_index': 'Индекс дерева в ансамбле (например, в Random Forest). Позволяет выбрать конкретное дерево для визуализации.',
        'max_depth': 'Максимальная глубина визуализируемого дерева. Помогает ограничить количество уровней для улучшения читаемости графа.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tree_index: int | None = None
        self.max_depth: int = self.MAX_DEPTH_DEFAULT

    def update_properties(self, properties: dict[str, Any]):
        self.tree_index = None
        if tree_index := properties.get('tree_index'):
            try:
                self.tree_index = int(tree_index)
            except ValueError as e:
                raise PlottingParametersException("Parameter 'tree_index' must be an integer") from e

        self.max_depth = properties.get('max_depth', self.MAX_DEPTH_DEFAULT)

    def validate_model(self, model: BaseModel | None) -> BaseModel:
        model = super().validate_model(model)
        instance = model.instance_model
        if not hasattr(instance, 'tree_') and not hasattr(instance, 'estimators_'):
            raise PlottingParametersException(
                'The model should have such attrubutes as `tree_` or `estimators_` (as '
                'DecisionTreeClassifier, for example). '
            )

        return model

    def plot(
        self,
        ddf: DictDataFrame,
        properties: dict[str, Any],
        styles: dict[str, Any],
        **kwargs,
    ):
        self.apply_plotly_configs(properties, styles)

        model: MLModel = self.validate_model(kwargs.get('model')).instance_model

        # Обычно в моделях sklearn есть эта информация
        columns: list[str] = getattr(model, 'feature_names_in_', None)  # type: ignore

        cls_ = model if hasattr(model, 'tree_') else model.estimators_[self.tree_index or 0]
        class_names = getattr(cls_, 'classes_', np.ndarray([])).astype(str)

        fig = self.plot_tree_with_plotly(cls_, feature_names=columns, class_names=class_names, max_depth=self.max_depth)

        return convert_ndarray(fig.to_plotly_json())

    def plot_tree_with_plotly(
        self,
        model: Any,
        max_depth: int,
        feature_names: Optional[list[str]] = None,
        class_names: Optional[list[str]] = None,
    ) -> 'go.Figure':
        tree = model.tree_
        children_left, children_right = tree.children_left, tree.children_right
        features, thresholds, values = tree.feature, tree.threshold, tree.value

        node_x, node_y, node_labels, hovertexts, colors, node_sizes = [], [], [], [], [], []
        edges_x, edges_y = [], []
        max_depth = min(max_depth, tree.max_depth)

        # --- Scaling settings ---
        max_size = self.MAX_NODE_SIZE / max_depth
        min_size = 1
        scaling_factor = max_depth * 2

        def get_node_label(node_id: int, is_leaf: bool) -> str:
            n_samples = tree.n_node_samples[node_id]
            if is_leaf:
                return f'<b>end leaf</b><br>samples={n_samples}'
            fname = feature_names[features[node_id]] if feature_names is not None else f'f{features[node_id]}'
            return f'<b>{fname} ≤ {thresholds[node_id]:.2f}</b><br>samples={n_samples}'

        def build_nodes(node_id=0, depth=0, x=0.0, dx=1.0):
            """
            Рекурсивно строим узлы дерева вместе с ребрами
            """
            if max_depth is not None and depth > max_depth:
                return

            y = depth
            size = max(min_size, max_size * np.exp(-depth / scaling_factor))
            is_leaf = children_left[node_id] == children_right[node_id]

            node_x.append(x)
            node_y.append(y)
            node_sizes.append(size)

            label = get_node_label(node_id, is_leaf)
            hovertexts.append(self.get_hovertext(values[node_id], label, size))
            node_labels.append('' if size < 60 else label)
            colors.append(self.get_node_color(values[node_id]))

            if not is_leaf:
                next_d = depth + 1
                for child_id, child_x in zip([children_left[node_id], children_right[node_id]], [x - dx, x + dx]):
                    edges_x.extend([x, child_x, None])
                    edges_y.extend([y, next_d, None])
                    build_nodes(child_id, next_d, child_x, dx / 2)

        build_nodes()

        # Flip y-axis
        node_y = [-y for y in node_y]
        edges_y = [-y if y is not None else None for y in edges_y]

        fig = go.Figure(
            data=[
                go.Scatter(x=edges_x, y=edges_y, mode='lines', line=dict(width=1, color='#888'), showlegend=False),
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    marker=dict(color=colors, size=node_sizes, symbol='circle', line=dict(color='black', width=1)),
                    text=node_labels,
                    textposition='middle center',
                    textfont=dict(color='black'),
                    hoverinfo='text',
                    hovertext=hovertexts,
                    showlegend=False,
                ),
            ]
        )

        layout = self.compile_layout()

        fig.update_layout(layout)

        # Add legend if class names provided
        if class_names is not None:
            fig = self.add_legend(fig, class_names=class_names)

        return fig

    def add_legend(self, fig: 'go.Figure', class_names: list[str]) -> 'go.Figure':
        for i, class_name in enumerate(class_names):
            color = hsl_to_hex((i * 0.3) % 1.0, 0.7, 0.5)
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=15, color=color, line=dict(color='black', width=1)),
                    legendgroup=class_name,
                    showlegend=True,
                    name=class_name,
                )
            )

        return fig

    def compile_layout(self, **kwargs) -> dict[str, Any]:
        layout = super().compile_layout()
        layout.update(
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
        return layout

    def get_node_color(self, value: np.ndarray) -> str:
        value_sum = np.sum(value)
        class_id = int(np.argmax(value))
        confidence = value[0][class_id] / value_sum if value_sum > 0 else 0.0
        hue = (class_id * 0.3) % 1.0
        lightness = 1.0 - 0.5 * confidence
        return hsl_to_hex(hue, 0.7, lightness)

    def get_hovertext(self, value: np.ndarray, label: str, size: float) -> str:
        probas = ' '.join(f'{v:.2f}' for v in value[-1])
        return f'{label}<br>probas: {probas}' if size < 60 else f'probas: {probas}'

    @classmethod
    def check_to_show(cls, *args, columns=None, **kwargs):
        model_path = kwargs.get('model_path')
        # Строится только при деревьях и лесах
        # Проверяем по названию -> не очень красиво, но лучшего способа пока не придумано
        for model_pattern in TREE_MODELS:
            if model_pattern in model_path:
                return True

        return False

    @classmethod
    def default_properties(cls) -> dict[str, Any]:
        return {'tree_index': ''}

    @classmethod
    def get_columns(
        cls,
        data_types: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        instance: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        plots_data: dict[str, Any] = {}
        if instance is None:
            plots_data |= cls.default_properties()
            max_depth = cls.MAX_DEPTH_DEFAULT
        else:
            plots_data['tree_index'] = str(instance.tree_index)
            max_depth = instance.max_depth

        return plots_data | {'max_depth': max_depth}
