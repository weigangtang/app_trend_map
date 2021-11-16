import numpy as np
import pandas as pd

import re
import geopandas as gpd

import plotly.express as px
import plotly.graph_objs as go

from dash import dcc
from dash import html

from dash import callback_context as ctx
from dash.dependencies import Input, Output, State

px.set_mapbox_access_token(open('.mapbox_token').read())

df_rsid = pd.read_csv('data/regional_sid.csv', index_col=0)
sr_rsid = df_rsid['Name']
sr_rsid.name = 'Region'

glacier = pd.read_csv('data/watershed_glacier_area.csv', index_col=0)
glacier = glacier[['GLA AREA', 'GLA PERC']]

gauges = pd.read_csv('data/Station_Info_ALL.csv', index_col=0)
gauges = gauges.join(glacier)

gauges = gauges.join(sr_rsid)
gauges['Region'] = gauges['Region'].fillna('-')

gdf = gpd.read_file('data/watershed_shapefile/ALL_REF_watersheds.shp')
gdf.index = [
    '{}-{}'.format(prov, sid)
    for prov, sid in gdf[['PROV', 'STATION ID']].values
]

df_hys = pd.read_csv('data/sel_hys.csv', index_col=[0, 1])
df_hys.columns = np.arange(1, 366)

# keep hydrometric order
hym_list = [
    'mean', 'median', 'std', 'skew', 'range',
    '10p', '25p', '75p', '90p',
    'min', 'min7d', 'max', 'max7d',
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'jan_p', 'feb_p', 'mar_p', 'apr_p', 'may_p', 'jun_p',
    'jul_p', 'aug_p', 'sep_p', 'oct_p', 'nov_p', 'dec_p',
    'cen_jd', '25p_jd', '50p_jd', '75p_jd',
    'max7d_jd', 'min7d_jd', 'max14d_jd', 'min14d_jd',
    'low_count', 'low_dur', 'high_count', 'high_dur',
    'si',
    'spr', 'spr_p', 'smr', 'smr_p',
    'aut', 'aut_p', 'win', 'win_p',
    'smr2', 'smr2_p', 'aut2', 'aut2_p',
    'cold', 'cold_p', 'cold2', 'cold2_p',
    'spr_max14d_jd', 'spr_onset_jd',
]

hym_name = pd.read_csv('data/hym_name_table.csv', index_col=0)
hym_name = hym_name.loc[hym_list]

mktest_options = [
    {
        'label': 'Original Mann-Kendall Test',
        'value': 'original'},
    {
        'label': 'Hamed and Rao Modified MK Test',
        'value': 'rao'},
    {
        'label': 'Yue and Wang Modified MK Test',
        'value': 'yue'},
    {
        'label': 'Modified MK Test using Pre-Whitening Method',
        'value': 'prewhiten'},
    {
        'label': 'Modified MK Test using Trend Free Pre-Whitening Method',
        'value': 'trendfree'},
]

hym_options = [
    {'label': val, 'value': idx}
    for idx, val in hym_name['Name'].items()
]

region_options = [
    {'label': item, 'value': item} for item in
    [
        'North America', 'WNA',
        'Northwest', 'CRM', 'USRM', 'CPNW', 'CPMW', 'Southwest',
    ]
]

canadian_western_prov = ['YT', 'BC', 'NT', 'AB']
us_western_prov = [
    'AK', 'WA', 'OR', 'CA', 'NV', 'ID',
    'UT', 'AZ', 'MT', 'WY', 'CO', 'NM'
]
western_prov = canadian_western_prov + us_western_prov

# ----------------------------------------------------------------------------
fig_trend_map = go.Figure(go.Scattermapbox())
fig_trend_map.update_layout(
    title='Trend Map of Hydrometric',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox={
        'accesstoken': open('.mapbox_token').read(),
        'bearing': 0,
        'center': {'lat': 50, 'lon': -105},
        'zoom': 2.2,
        'style': 'open-street-map',
    },
    height=490,
    margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
)

trend_summary_frame = """
    ||||
    | ------------- | ------------- | ------------- |
    | MK-Test P-Value: {:.3f} | Net Change: {:.3f} | Initial:  {:.3f} |
    | Sen's Slope: {:.3f} | Change Rate: {:.3f}| Last:  {:.3f} |
    | \\# Valid Year: {:.0f}|||
"""

trend_summary_empty = re.sub('{\\S*}', '', trend_summary_frame)

trend_plot_layout = {
    'font_size': 14,
    'title': {
        'text': 'Trend Line:',
        'x': .05,
        'y': .98,
        'font': {'size': 18},
    },
    'hovermode': 'closest',  # closest
    'showlegend': True,
    'autosize': False,
    'xaxis': dict(
        title='Year',
        zeroline=False,
        domain=[0., .98],
        showgrid=False,
        automargin=True
    ),
    'yaxis': dict(
        zeroline=False,
        domain=[0., .98],
        showgrid=False,
        automargin=True
    ),
    'plot_bgcolor': 'rgba(200,200,200,0.1)',
    # 'paper_bgcolor': '#F2F2F2',
    # 'width': 800,
    'height': 240,
    'margin': dict(l=2, r=2, t=30, b=2),
}
trend_plot_empty = go.Figure(layout=trend_plot_layout)

gts_layout = {
    'font_size': 14,
    'title': {
        'text': 'Annual Daily Hydrographs: ',
        'x': .05,
        'y': .98,
        'font': {'size': 18},
    },
    'hovermode': 'closest',  # closest
    'showlegend': False,
    'autosize': False,
    'xaxis': dict(
        title='Day of Year',
        zeroline=False,
        domain=[0., .98],
        showgrid=False,
        automargin=True
    ),
    'yaxis': dict(
        title='Flow',
        zeroline=True,
        domain=[0., .98],
        showgrid=False,
        automargin=True
    ),
    'plot_bgcolor': 'rgba(200,200,200,0.1)',
    # 'paper_bgcolor': '#F2F2F2',
    'height': 300,
    'margin': dict(l=2, r=2, t=30, b=2),
}
gts_plot_empty = go.Figure(layout=gts_layout)

trend_config_dict = {
    'all': {
        'name': 'All',
        'size': 8,
        'color': 'rgba(100,100,100,0.5)',
    },
    'na': {
        'name': 'NA',
        'size': 8,
        'color': 'rgba(100,100,100,0.5)',
    },
    'neg': {
        'name': 'Positive',
        'size': 8,
        'color': 'rgba(0,0,256,0.3)',
    },
    'pos': {
        'name': 'Negative',
        'size': 8,
        'color': 'rgba(256,0,0,0.3)',
    },
    'sig_neg': {
        'name': 'Significant Negative',
        'size': 10,
        'color': 'rgba(0,0,256,0.9)',
    },
    'sig_pos': {
        'name': 'Significant Positive',
        'size': 10,
        'color': 'rgba(256,0,0,0.9)',
    },
}

mkout_param_options = [
    {'label': 'Net Change', 'value': 'chg'},
    {'label': 'Change Rate', 'value': 'chg_r'},
    {'label': 'Initial Value', 'value': 'init'},
    {'label': 'Last Value', 'value': 'last'},
]

# Support Functions ----------------------------------------------------------
# ----------------------------------------------------------------------------
def find_trend_group(df_mkout, trend_type, pthr):
    if trend_type == 'all':
        return df_mkout.index
    elif trend_type == 'pos':
        return df_mkout[df_mkout['slp'] > 0].index
    elif trend_type == 'neg':
        return df_mkout[df_mkout['slp'] < 0].index
    elif trend_type == 'sig_pos':
        idx = (df_mkout['slp'] > 0) & (df_mkout['pvalue'] <= pthr)
        return df_mkout[idx].index
    elif trend_type == 'sig_neg':
        idx = (df_mkout['slp'] < 0) & (df_mkout['pvalue'] <= pthr)
        return df_mkout[idx].index
    else:
        return df_mkout.index


def add_watershed_polygon(sel_sid):

    gdf_sel = gdf[gdf.index == sel_sid]

    gjs_sel = eval(gdf_sel.to_json())
    coords = gjs_sel['features'][0]['geometry']['coordinates'][0]

    # get watershed layer
    geojd = {'type': 'FeatureCollection', 'features': []}
    geojd['features'].append(
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]}
        }
    )
    watershed_layer = {
        'sourcetype': 'geojson',
        'source': geojd,
        'below': '',
        'type': 'fill',
        'color': 'rgba(0,0,256,0.2)',
    }
    return watershed_layer


def add_gauges_points(fig, df_sel, size, rgba):

    hovertext_frame = '<b>{}</b><br>' + 'Name: {}<br>' + \
        'Area: {:.1f} sqkm<br>' + 'Glacier Coverage (%): {:.2f}'

    text_list = [
        hovertext_frame.format(
            idx,
            val['STATION NAME'],
            val['DRAINAGE AREA'],
            val['GLA PERC'],
        )
        for idx, val in df_sel.iterrows()
    ]

    fig.add_trace(
        go.Scattermapbox(
            lat=df_sel['LATITUDE'],
            lon=df_sel['LONGITUDE'],
            mode='markers',
            marker={'size': size, 'color': rgba},
            text=text_list,
            hoverinfo='text'
        )
    )


def extract_sid_from_click(click_data):
    text = click_data['points'][0]['text']
    text = text.split('<br>')[0]
    sel_sid = text.replace('<b>', '').replace('</b>', '')
    return sel_sid


# App Layout -----------------------------------------------------------------
# ----------------------------------------------------------------------------
def create_layout(app):
    return html.Div(
        className='row',
        children=[

            html.Div(
                className='seven columns',
                children=[

                    html.Div(
                        children=[
                            html.Label(
                                'Basemap:',
                                style={
                                    'font-size': 16,
                                    'font-weight': 'bold',
                                    'margin-bottom': '10px',
                                },
                            ),
                            dcc.RadioItems(
                                id='radio-basemap',
                                value='open-street-map',
                                options=[
                                    {'label': 'Open Street', 'value': 'open-street-map'},
                                    {'label': 'Satellite', 'value': 'satellite'},
                                ],
                                labelStyle={
                                    'display': 'inline-block',
                                    'margin-right': '10px'
                                },
                                inputStyle={'margin-right': '5px'},
                            ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '28%',
                            'height': '70px',
                            'margin-left': '1%',
                            'margin-bottom': '10px',
                            'vertical-align': 'middle',
                            'border': '1px rgb(200,200,200) solid',
                            'border-radius': '5px',
                            'padding': '5px 15px 5px 15px',
                        }
                    ),


                    html.Div(
                        children=[
                            html.Label(
                                'Region:',
                                style={
                                    'font-size': 16,
                                    'font-weight': 'bold',
                                    'margin-bottom': '5px',
                                },
                            ),
                            dcc.Dropdown(
                                id='dropdown-select-region',
                                value='All',
                                options=region_options,
                            ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '20%',
                            'height': '70px',
                            'margin-left': '1%',
                            'margin-bottom': '10px',
                            'vertical-align': 'middle',
                            'border': '1px rgb(200,200,200) solid',
                            'border-radius': '5px',
                            'padding': '5px 15px 5px 15px',
                        }
                    ),

                    html.Div(
                        children=[
                            html.Label(
                                'Glacial Type:',
                                style={
                                    'font-size': 16,
                                    'font-weight': 'bold',
                                    'margin-bottom': '10px',
                                },
                            ),
                            dcc.RadioItems(
                                id='radio-glacial',
                                value='all',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Glacial', 'value': 'gla'},
                                    {'label': 'Non-Glacial', 'value': 'non'},
                                ],
                                labelStyle={
                                    'display': 'inline-block',
                                    'margin-right': '10px'
                                },
                                inputStyle={'margin-right': '5px'},
                            ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '35%',
                            'height': '70px',
                            'margin-left': '1%',
                            'margin-bottom': '10px',
                            'vertical-align': 'middle',
                            'border': '1px rgb(200,200,200) solid',
                            'border-radius': '5px',
                            'padding': '5px 15px 5px 15px',
                        }
                    ),

                    dcc.Graph(
                        id='graph-trend-map',
                        figure=fig_trend_map,
                    ),

                    html.Label(
                        id='trend-count-string_pos',
                        style={
                            'color': 'rgba(250,0,0,.9)',
                            'width': '40%',
                            'margin-top': '10px',
                            'margin-left': '10%',
                            'display': 'inline-block',
                        },
                    ),
                    html.Label(
                        id='trend-count-string_neg',
                        style={
                            'color': 'rgba(0,0,150,.9)',
                            'width': '40%',
                            'margin-top': '10px',
                            'display': 'inline-block',
                        },
                    ),
                    html.Div(
                        dcc.Graph(
                            id='graph-pie',
                        ),
                        style={
                            'width': '60%',
                            'margin-top': '20px',
                            'display': 'inline-block'
                        },
                    ),
                    html.Div(
                        children=[
                            dcc.Dropdown(
                                id='dropdown-mktout-params',
                                options=mkout_param_options,
                                value='chg',
                            ),
                            dcc.Graph(
                                id='graph-boxplot',
                            ),
                        ],
                        style={
                            'width': '40%',
                            'margin-top': '20px',
                            'display': 'inline-block'
                        },
                    ),

                    dcc.ConfirmDialog(
                        id='confirm-no-shapefile',
                        message='Watershed Boundary Unavailable!'
                    ),

                    dcc.Store(
                        id='store-mkout-data',
                        storage_type='session',
                    ),
                ],
            ),

            html.Div(
                className='five columns',
                children=[

                    html.Div(
                        children=[
                            html.Label(
                                'Select Hydrometric:',
                                style={
                                    'font-size': 16,
                                    'font-weight': 'bold',
                                    'margin-bottom': '5px',
                                },
                            ),
                            dcc.Dropdown(
                                id='dropdown-select-hydrometric',
                                placeholder='Select Hydrometric',
                                value='mean',
                                options=hym_options,
                            ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '54%',
                            'height': '70px',
                            'margin-right': '2%',
                            'margin-bottom': '10px',
                            'vertical-align': 'middle',
                            'border': '1px rgb(200,200,200) solid',
                            'border-radius': '5px',
                            'padding': '5px 15px 5px 15px',
                        }
                    ),

                    html.Div(
                        children=[
                            html.Label(
                                'Significance Level:',
                                style={
                                    'font-size': 16,
                                    'font-weight': 'bold',
                                    'margin-bottom': '10px',
                                },
                            ),
                            dcc.Slider(
                                id='slider-pvalue-thr',
                                value=0.10,
                                min=0.01,
                                max=0.10,
                                step=None,
                                marks={
                                    0.01: {
                                        'label': '0.01',
                                        'style': {'font-size': 14}
                                    },
                                    0.05: {
                                        'label': '0.05',
                                        'style': {'font-size': 14}
                                    },
                                    0.10: {
                                        'label': '0.10',
                                        'style': {'font-size': 14}
                                    },
                                },
                            ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '30%',
                            'height': '70px',
                            'margin-bottom': '10px',
                            'vertical-align': 'middle',
                            'border': '1px rgb(200,200,200) solid',
                            'border-radius': '5px',
                            'padding': '5px 15px 5px 15px',
                        }
                    ),

                    html.Div(
                        children=[
                            html.Label(
                                'Select MK-test Method: ',
                                style={
                                    'font-size': 16,
                                    'font-weight': 'bold',
                                    'margin-bottom': '5px',
                                },
                            ),
                            dcc.Dropdown(
                                id='dropdown-select-mktest',
                                placeholder='Select MK-test',
                                value='trendfree',
                                options=mktest_options,
                            ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '93%',
                            'height': '70px',
                            'margin-bottom': '10px',
                            'vertical-align': 'middle',
                            'border': '1px rgb(200,200,200) solid',
                            'border-radius': '5px',
                            'padding': '5px 15px 5px 15px',
                        }
                    ),

                    dcc.Graph(
                        id='graph-trend-plot',
                        figure=trend_plot_empty,
                        style={'margin-top': '10px'}
                    ),
                    dcc.Markdown(
                        id='markdown-trend-summary',
                        children=trend_summary_empty,
                        style={'height': '180px', 'margin-top': '10px'}
                    ),

                    dcc.Graph(
                        id='graph-gts',
                        figure=gts_plot_empty,
                        style={'width': '100%', 'margin-top': '20px'},
                    ),
                ],
            ),

        ],
    )


def demo_callbacks(app):

    # @app.callback(
    #     Output('confirm-no-shapefile', 'displayed'),
    #     Input('graph-trend-map', 'clickData'),
    # )
    # def display_confirm(click_data):
    #     if click_data:
    #         sel_sid = extract_sid_from_click(click_data)
    #         if sel_sid not in gdf.index:
    #             return True
    #     return False

    @app.callback(
        Output('store-mkout-data', 'data'),
        Input('dropdown-select-region', 'value'),
        Input('radio-glacial', 'value'),
        Input('dropdown-select-hydrometric', 'value'),
        Input('dropdown-select-mktest', 'value'),
        Input('slider-pvalue-thr', 'value'),
    )
    def select_mkout_data(region, gla_type, sel_hym, method, pthr):

        df_mkout = pd.read_csv(
            'data/mktest/{}/{}.csv'.format(method, sel_hym), index_col=0)
        df = df_mkout.join(gauges, how='left')

        if region == 'North America':
            df = df.copy()
        elif region == 'WNA':
            df = df[df['Region'] != '-']
        else:
            df = df[df['Region'] == region]

        if gla_type == 'gla':
            df = df[df['GLA PERC'] > 0.]
        if gla_type == 'non':
            df = df[df['GLA PERC'] == 0.]

        df['type'] = 'na'
        for trend_type in ['pos', 'sig_pos', 'neg', 'sig_neg']:
            df.loc[find_trend_group(df, trend_type, pthr), 'type'] = trend_type
        return df.to_dict()

    @app.callback(
        Output('trend-count-string_pos', 'children'),
        Output('trend-count-string_neg', 'children'),
        Input('store-mkout-data', 'data'),
    )
    def count_trend_type(data):

        df = pd.DataFrame.from_dict(data)

        n_pos = np.sum((df['type'] == 'pos') | (df['type'] == 'sig_pos'))
        n_neg = np.sum((df['type'] == 'neg') | (df['type'] == 'sig_neg'))
        n_sig_pos = np.sum(df['type'] == 'sig_pos')
        n_sig_neg = np.sum(df['type'] == 'sig_neg')

        pos_count_str = 'Positive: {:4}'.format(n_pos) + ' | ' + \
            'Signficiant Positive: {:4}'.format(n_sig_pos)
        neg_count_str = 'Negative: {:4}'.format(n_neg) + ' | ' + \
            'Significant Negative: {:4}'.format(n_sig_neg)
        return pos_count_str, neg_count_str

    @app.callback(
        Output('graph-trend-map', 'figure'),
        Input('store-mkout-data', 'data'),
        Input('radio-basemap', 'value'),
        Input('graph-trend-map', 'clickData'),
        State('graph-trend-map', 'figure'),
    )
    def plot_trend_map(data, basemap, click_data, fig_config):

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        fig = go.Figure(fig_config)  # fig_config = {data, layout}

        if trigger_id == 'store-mkout-data':
            df = pd.DataFrame.from_dict(data)
            fig.data = []
            for trend_type in ['pos', 'sig_pos', 'neg', 'sig_neg']:
                size = trend_config_dict[trend_type]['size']
                color = trend_config_dict[trend_type]['color']
                df_sel = df[df['type'] == trend_type]
                add_gauges_points(fig, df_sel, size, color)

        else:
            sel_sid = extract_sid_from_click(click_data)
            if sel_sid in gdf.index:
                watershed_layer = add_watershed_polygon(sel_sid)
                fig.update_layout(
                    mapbox_style=basemap,
                    mapbox_layers=[watershed_layer]
                )

        return fig

    @app.callback(
        Output('graph-trend-plot', 'figure'),
        Output('markdown-trend-summary', 'children'),
        Input('graph-trend-map', 'clickData'),
        Input('store-mkout-data', 'data'),
        State('dropdown-select-hydrometric', 'value'),
    )
    def plot_trend(click_data, data, sel_hym):

        if click_data:

            df = pd.DataFrame.from_dict(data)

            text = click_data['points'][0]['text']
            text = text.split('<br>')[0]
            sel_sid = text.replace('<b>', '').replace('</b>', '')

            df_hym = pd.read_csv(
                'data/hydrometrics/{}.csv'.format(sel_hym), index_col=0)
            sel_hym_name = hym_name.loc[sel_hym, 'Name']

            ts = df_hym.loc[sel_sid]
            t = ts.index.values.astype(int)
            y = ts.values

            mkout = df.loc[sel_sid]
            slp = mkout['slp']
            intp = mkout['intp']
            trend_type = mkout['type']

            y2 = slp * t + intp

            data = [
                go.Scatter(
                    x=t, y=y, name='Time Series', mode='markers', text=t,
                    showlegend=False, marker={'color': 'grey', 'size': 6},
                    hovertemplate='Year: %{text}' + '<br>Value: %{y:.3f}</br>'
                )
            ]
            data.append(
                go.Scatter(
                    x=t, y=y2, name='Sens Slope', mode='lines',
                    showlegend=False, line={'color': 'green', 'width': 2})
            )

            bgcolor = 'white'  # can't use rgba for white
            if trend_type == 'sig_pos':
                bgcolor = 'rgba(256,0,0,.2)'
            if trend_type == 'sig_neg':
                bgcolor = 'rgba(0,0,256,.2)'

            layout = trend_plot_layout.copy()
            layout.update({
                'title': {
                    'text': 'Trend Line: {}'.format(sel_sid),
                    'x': .05,
                    'y': .98,
                    'font': {'size': 18},
                },
                'plot_bgcolor': bgcolor,
                'yaxis_title': sel_hym_name,
            })

            fig = {'data': data, 'layout': layout}

            trend_summary = trend_summary_frame.format(
                mkout['pvalue'], mkout['chg'], mkout['init'],
                mkout['slp'], mkout['chg_r'], mkout['last'],
                mkout['n'],
            )
            return fig, trend_summary

        else:
            return trend_plot_empty, trend_summary_empty

    @app.callback(
        Output('graph-gts', 'figure'),
        Input('graph-trend-map', 'clickData'),
    )
    def plot_gts(click_data):

        if click_data:

            sel_sid = extract_sid_from_click(click_data)
            sel_hys = df_hys.loc[sel_sid]

            x = np.arange(365)
            y2 = sel_hys.median(axis=0).values

            data = []
            for year in sel_hys.index:
                y = sel_hys.loc[year].values
                data.append(
                    go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        line_width=1.5,
                        marker_color='rgba(100,100,100,0.2)',
                        hovertemplate=str(year))
                )
            data.append(
                go.Scatter(
                    x=x, y=y2,
                    mode='lines',
                    marker_color='rgba(0,0,256,.9)',
                    line_width=2.5,
                    hovertemplate='Average')
            )

            layout = gts_layout.copy()
            layout.update({
                'title': {
                    'text': 'Annual Daily Hydrographs: {}'.format(sel_sid),
                    'x': .05,
                    'y': .98,
                    'font': {'size': 18},
                }
            })

            return {'data': data, 'layout': layout}

        else:
            return {'data': [], 'layout': gts_layout}

    @app.callback(
        Output('graph-pie', 'figure'),
        Input('store-mkout-data', 'data'),
    )
    def plot_pie(data):

        df = pd.DataFrame.from_dict(data)

        trend_type_list = ['na', 'pos', 'sig_pos', 'neg', 'sig_neg']
        count = df.groupby('type').count()['slp']
        count = count.reindex(trend_type_list).dropna()

        name_list = [
            trend_config_dict[trend_type]['name']
            for trend_type in count.index
        ]
        color_list = [
            trend_config_dict[trend_type]['color']
            for trend_type in count.index
        ]

        fig_pie = go.Figure(
            go.Pie(
                values=count.values,
                labels=name_list, marker_colors=color_list,
                textinfo='percent+label', textfont_size=16,
                direction='clockwise', sort=False, hole=.3,
            )
        )
        fig_pie.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(200,200,200,0.1)',
            height=300,
            margin=dict(l=2, r=2, t=2, b=2),
        )
        return fig_pie

    @app.callback(
        Output('graph-boxplot', 'figure'),
        Input('store-mkout-data', 'data'),
        Input('dropdown-mktout-params', 'value'),
    )
    def plot_boxplot(data, mkout_param):

        fig_box = go.Figure()

        if mkout_param:

            df = pd.DataFrame.from_dict(data)

            for trend_type in ['all', 'pos', 'sig_pos', 'neg', 'sig_neg']:
                if trend_type == 'all':
                    df_sel = df.copy()
                else:
                    df_sel = df[df['type'].str.endswith(trend_type)]
                fig_box.add_trace(
                    go.Violin(
                        y=df_sel[mkout_param],
                        name=trend_config_dict[trend_type]['name'],
                        marker_color=trend_config_dict[trend_type]['color'],
                    ),
                )

        fig_box.update_layout(
            showlegend=False,
            xaxis_showticklabels=False,
            plot_bgcolor='rgba(200,200,200,0.1)',
            height=270,
            margin=dict(l=2, r=2, t=2, b=2),
        )
        return fig_box