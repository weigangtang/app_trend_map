import numpy as np
import pandas as pd

import re

import plotly.express as px
import plotly.graph_objs as go

from dash import dcc
from dash import html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

px.set_mapbox_access_token(open('.mapbox_token').read())

gauges = pd.read_csv('data/Station_Info_ALL.csv', index_col=0)

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
    'aut', 'aut_p', 'spr', 'spr_p',
    'smr', 'smr_p', 'smr2', 'smr2_p',
    'win', 'win_p', 'win2', 'win2_p',
    'spr_max14d_jd', 'spr_onset_jd',
]

hym_name = pd.read_csv('data/Hydrometric_Name_Table.csv', index_col=0)
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

trend_summary_frame = """
    |Trend Analysis||
    | ------------- | ------------- | ------------- |
    | MK-Test P-Value: {:.3f} | Net Change: {:.3f} | Initial:  {:.3f} |
    | Sen's Slope: {:.3f} | Change Rate: {:.3f}| Last:  {:.3f} |
    | \\# Valid Year: {:.0f}||
"""

canadian_western_prov = ['YT', 'BC', 'NT', 'AB']
us_western_prov = [
    'AK', 'WA', 'OR', 'CA', 'NV', 'ID',
    'UT', 'AZ', 'MT', 'WY', 'CO', 'NM'
]
western_prov = canadian_western_prov + us_western_prov

trend_summary_empty = re.sub('{\\S*}', '', trend_summary_frame)

trend_plot_empty = go.Figure(
    layout={
        'height': 320,
        'margin': dict(l=2, r=2, t=50, b=2),
    },
)

# App Layout -----------------------------------------------------------------
# ----------------------------------------------------------------------------
def create_layout(app):
    return html.Div(
        className='row',
        children=[

            html.Div(
                className='row',
                # style={'height': '450px'},
                children=[
                    html.Div(
                        className='seven columns',
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
                                    'width': '30%',
                                    'height': '75px',
                                    'margin-right': '1%',
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
                                    dcc.RadioItems(
                                        id='radio-region',
                                        value='all',
                                        options=[
                                            {'label': 'All', 'value': 'all'},
                                            {'label': 'West Only', 'value': 'west'},
                                        ],
                                        labelStyle={
                                            'display': 'inline-block',
                                            'margin-right': '20px'
                                        },
                                        inputStyle={'margin-right': '10px'},
                                    ),
                                ],
                                style={
                                    'display': 'inline-block',
                                    'width': '25%',
                                    'height': '75px',
                                    'margin-right': '1%',
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
                                        value=0.05,
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
                                    'width': '28%',
                                    'height': '75px',
                                    'margin-right': '1%',
                                    'margin-bottom': '10px',
                                    'vertical-align': 'middle',
                                    'border': '1px rgb(200,200,200) solid',
                                    'border-radius': '5px',
                                    'padding': '5px 15px 5px 15px',
                                }
                            ),

                            dcc.Graph(
                                id='graph-trend-map',
                            ),
                        ],
                    ),

                    html.Div(
                        className='five columns',
                        children=[

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
                                        value='original',
                                        options=mktest_options,
                                    ),
                                ],
                                style={
                                    'display': 'inline-block',
                                    'width': '90%',
                                    'height': '75px',
                                    'margin-bottom': '10px',
                                    'vertical-align': 'middle',
                                    'border': '1px rgb(200,200,200) solid',
                                    'border-radius': '5px',
                                    'padding': '5px 15px 5px 15px',
                                }
                            ),

                            dcc.Markdown(
                                id='markdown-trend-summary',
                                children=trend_summary_empty,
                                style={'width': '100%', 'height': '180px'}
                            ),
                            dcc.Graph(
                                id='graph-trend-plot',
                                figure=trend_plot_empty,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def add_trace(fig, df_sel, size, rgba):

    hovertext_frame = '<b>{}</b><br>Name: {}<br>Area: {:.1f} sqkm'

    text_list = [
        hovertext_frame.format(
            idx, val['STATION NAME'], val['DRAINAGE AREA']
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


def demo_callbacks(app):

    @app.callback(
        Output('graph-trend-map', 'figure'),
        Input('dropdown-select-hydrometric', 'value'),
        Input('dropdown-select-mktest', 'value'),
        Input('slider-pvalue-thr', 'value'),
        Input('radio-region', 'value'),
    )
    def plot_trend_map(sel_hym, method, thr, region):

        df_mkout = pd.read_csv(
            'data/{}/{}.csv'.format(method, sel_hym), index_col=0)
        df = df_mkout.join(gauges, how='left')

        if region == 'west':
            df = df[df['PROV'].isin(western_prov)]

        fig = go.Figure()

        df_sel = df[(df['pvalue'] < thr) & (df['slp'] > 0)]
        add_trace(fig, df_sel, 10, 'rgba(255,0,0,.9)')

        df_sel = df[(df['pvalue'] > thr) & (df['slp'] > 0)]
        add_trace(fig, df_sel, 8, 'rgba(255,0,0,.2)')

        df_sel = df[(df['pvalue'] < thr) & (df['slp'] < 0)]
        add_trace(fig, df_sel, 10, 'rgba(0,0,255,.8)')

        df_sel = df[(df['pvalue'] > thr) & (df['slp'] < 0)]
        add_trace(fig, df_sel, 8, 'rgba(0,0,255,.2)')

        fig.update_layout(
            title='Trend Map of Hydrometric',
            autosize=True,
            hovermode='closest',
            showlegend=False,
            mapbox={
                'accesstoken': open('.mapbox_token').read(),
                'bearing': 0,
                'center': {'lat': 50, 'lon': -105},
                'zoom': 2.2,
                'style': 'light'
            },
            height=500,
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        )
        return fig

    @app.callback(
        Output('graph-trend-plot', 'figure'),
        Output('markdown-trend-summary', 'children'),
        Input('graph-trend-map', 'clickData'),
        Input('dropdown-select-mktest', 'value'),
        State('dropdown-select-hydrometric', 'value'),
        State('slider-pvalue-thr', 'value'),
    )
    def plot_trend(click_data, method, sel_hym, thr):

        if click_data:

            text = click_data['points'][0]['text']
            text = text.split('<br>')[0]
            sel_sid = text.replace('<b>', '').replace('</b>', '')

            df_hym = pd.read_csv(
                'data/hym/{}.csv'.format(sel_hym), index_col=0)

            df_mkout = pd.read_csv(
                'data/{}/{}.csv'.format(method, sel_hym), index_col=0)

            sel_hym_name = hym_name.loc[sel_hym, 'Name']

            ts = df_hym.loc[sel_sid]
            t = ts.index.values.astype(int)
            y = ts.values

            mkout = df_mkout.loc[sel_sid]
            slp = mkout['slp']
            intp = mkout['intp']
            pvalue = mkout['pvalue']

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

            if pvalue < thr:
                if slp > 0:
                    bgcolor = 'rgba(256,0,0,.2)'
                else:
                    bgcolor = 'rgba(0,0,256,.2)'
            else:
                bgcolor = 'rbga(0,0,0,0)'

            layout = {
                'font_size': 14,
                'title': 'Station: {}'.format(sel_sid),
                'title_font_size': 30,
                'title_font_family': "Times New Roman",
                'hovermode': 'closest',  # closest
                'plot_bgcolor': bgcolor,
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
                    title=sel_hym_name,
                    zeroline=True,
                    domain=[0., .98],
                    showgrid=False,
                    automargin=True
                ),
                # 'paper_bgcolor': '#F2F2F2',
                # 'width': 800,
                'height': 320,
                'margin': dict(l=2, r=2, t=50, b=2),
            }

            figure = {'data': data, 'layout': layout}

            trend_summary = trend_summary_frame.format(
                pvalue, mkout['chg'], mkout['init'],
                slp, mkout['chg_r'], mkout['last'],
                mkout['n'],
            )

            return figure, trend_summary

        else:
            return trend_plot_empty, trend_summary_empty
