import io
from typing import Any, Callable

import pandas as pd
from lazy_imports import try_import

from smile_ml_core.data.structures import DictDataFrame
from smile_ml_core.plot_methods import BaseMethod

from smile_ml_core.data.exceptions import PlottingAbsentColumnExceptions

with try_import() as geopandas_import:
    import geopandas
    from geopandas.array import GeometryDtype

with try_import() as pyplot_import:
    import plotly.graph_objects as go


class GeoDataVisualizer(BaseMethod):
    styles_fields: list | str = []
    MAP_STYLES: list[str] = ['carto-positron', 'open-street-map', 'white-bg', 'carto-darkmatter']
    MAP_STYLE_DEFAULT: str = MAP_STYLES[0]

    columns_templates = [
        {
            'label': 'Columns',
            'name': 'columns',
            'source': 'columns',
            'multi': True,
        },
        {
            'label': 'Map style',
            'name': 'map_style',
            'source': 'map_style',
            'default': MAP_STYLE_DEFAULT,
            'multi': False,
        },
    ]

    def __init__(self):
        super().__init__()
        self.map_style: str = self.MAP_STYLE_DEFAULT

    def update_properties(self, properties: dict[str, Any]):
        self.columns = properties.get('columns', [])
        self.map_style = properties.get('map_style', self.MAP_STYLE_DEFAULT)
        if not self.columns:
            raise PlottingAbsentColumnExceptions('columns')

    def plot(
        self,
        ddf: DictDataFrame,
        properties: dict[str, Any],
        styles: dict[str, Any],
        boto_handler: Callable[[io.BytesIO, str], Any] | None = None,
        **kwargs,
    ):
        geopandas_import.check()
        pyplot_import.check()

        if boto_handler is None:
            raise ValueError('The plot method requires a boto handler')

        self.apply_plotly_configs(properties, styles)

        df = ddf.view()
        if isinstance(df, pd.DataFrame):
            df = geopandas.GeoDataFrame(df)  # to convert type

        df = df.set_geometry(self.columns[-1])
        df = df.set_crs(epsg=3857)

        fig = go.Figure()
        colors = self.get_colors()

        # Извлекаем широту и долготу из POINT объектов
        lats = []
        lons = []

        for point_list in df[self.columns].values:
            point = point_list[0]
            lons.append(point.x)  # Долгота
            lats.append(point.y)  # Широта

        fig.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',  # это точки, если нужны линии lines
                marker=go.scattermapbox.Marker(
                    size=17,
                    color=colors[0],
                    opacity=0.7,
                ),  # также если линии, то вместо marker указываешь line
                hoverinfo='text',  # то что будет во всплывающем окне
            )
        )

        fig.update_layout(
            mapbox={
                'center': {'lon': 30.282751, 'lat': 59.944652},
                # указываем точку, относительно которой надо центровать карту
                'style': self.map_style,  # стиль карты
                'zoom': 12,
            }
        )  # масштаб

        return fig.to_plotly_json()

    @classmethod
    def check_to_show(cls, *args, data_types=None, **kwargs):
        return any(map(lambda v: isinstance(v, GeometryDtype), data_types.values()))

    @classmethod
    def get_columns(cls, data_types=None, columns=None, instance=None, **kwargs):
        extra_checking_methods = {'geometry': lambda dt: isinstance(dt, GeometryDtype)}
        columns_geometry: list = cls.filter_dtypes_columns(
            data_types,
            methods=['geometry'],
            columns=columns,
            extra_checking_methods=extra_checking_methods,
        )
        if instance is None:
            columns = dict(map(lambda col: (col, True), columns_geometry))  # по дефолту выбраны все
            map_style = dict(
                map(lambda col: (col, True) if col == cls.MAP_STYLE_DEFAULT else (col, False), cls.MAP_STYLES)
            )
        else:
            columns = dict(map(lambda col: (col, col in instance.columns), columns_geometry))
            map_style = dict(map(lambda col: (col, col in instance.map_style), cls.MAP_STYLES))

        return {'columns': columns, 'map_style': map_style}
