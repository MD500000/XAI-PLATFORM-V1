import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html, dash_table
from dash.dependencies import Input, Output, State
from app import app
from apps import function_1, pages
import pandas as pd
import dash_table
import matplotlib
import fontawesome as fa
import dash_daq as daq
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular
import shap
import matplotlib
import matplotlib.pyplot as pl
import matplotlib.pylab as plt
from io import BytesIO
import base64
###########################################################
# değişken tanımlamaları
#iris_kayipverili.csv
#iris_aykiridegerli.csv
eniyimodel=None
#output-veri-analiz
#style={'display': 'none'}
new_df = None
new_oz_df = None
new_hedef_df = None
pozitif_sinif_ismi = None
verisetiozellikleri = None
new_oz_onehot=None
onehot_liste = None

###########################################################
#sayfa içerikleri
giris_metni= "Some methods were needed in order to make the results obtained as a result of modeling with machine learning methods more interpretable and explainable. Based on these requirements, the concept of explicable artificial intelligence was introduced. It is a set of methods developed to make the model more understandable by revealing the relationships between output and input variables. The use of classification models to diagnose disease in the field of health largely depends on the ability of the models created to be interpreted and explained by the researcher. There are many different ways to increase the explainability of artificial intelligence models created in the field of health and variable significance is one of them.Explainable AI methods used for this purpose provide a patient-specific explanation for a particular classification so that any i allows for a simpler explanation of a complex classifier in the clinical setting."
lime_metni = "LIME is a post-hoc model-free annotation technique that aims to approximate any black box machine learning model with a native, interpretable model to explain each individual prediction. As a result, LIME works locally, which means it is observation-specific and provides explanations for the prediction for each observation. What LIME does is try to fit a local model using sample data points similar to the observation described."
shap_metni = "The main idea of SHAP is to calculate the Shapley values for each feature of the sample to be interpreted, where each Shapley value represents the predictive impact of the feature to which it is associated."
nav_style={
    "color": "#ffffff",
    "link": "#ffffff",
    "background-color": "#F9A825",
	"width": "20rem",
	#"position": "fixed",
	"padding": "0.5rem 1rem",
	"left": 0,
	"line-height": 15.82857143,
	"font-size": "15px",

    #"font-family": "Bahnschrift"
}
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 60,
    "left": 0,
    "right": 0,
    #"bottom": 0,
    "width": "23rem",
    "height": "100%",
    "z-index": 2,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#ECEFF1",
	"line-height": 40.82857143,
	#"padding": "4rem 1rem 2rem",
	"font-family": "Bahnschrift"
}
SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 55,
    "left": "20rem",
    "bottom": 0,
    "width": "20rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#ECEFF1",
    "font-family": "Bahnschrift"
}

# login card
giris_card = dbc.Card(
    [
        dbc.CardImg(src="/assets/classification.jpg", top=True),
        dbc.CardBody(
            [
                html.H4("Explainable Artificial Intelligence (xAI) ", className="card-title",style={"width": "100%", 'text-align':'justify',"font-family": "Bahnschrift"},),
                html.P(
                    giris_metni,style={"width": "100%", 'text-align':'justify',"font-family": "Bahnschrift"},
                    className="card-text",
                ),

            ]
        ),
    ],
    style={"width": "30rem"},
    className="card text-white bg-info mb-3",
)



lime_card = dbc.Card(
    [

        dbc.CardBody(
            [
                html.H4("Local Interpretable Model-Agnostic Explanations (LIME)", className="card-title",style={"width": "100%", 'text-align':'justify',"font-family": "Bahnschrift"},),
                html.P(
                    lime_metni,style={"width": "100%", 'text-align':'justify',"font-family": "Bahnschrift"},
                    className="card-text",
                ),
            ]
        ),
        dbc.CardImg(src="/assets/lime.png", top=True),
    ],
    style={"width": "25rem"},
    className="card  mb-3 border-info text-white bg-primary ",
)


shap_card = dbc.Card(
    [
        dbc.CardImg(src="/assets/shap.png", top=True),
        dbc.CardBody(
            [
                html.H4(" Shapley Additive Explanations (SHAP) ", className="card-title",style={"width": "100%", 'text-align':'justify',"font-family": "Bahnschrift"},),
                html.P(
                    shap_metni,style={"width": "100%", 'text-align':'justify',"font-family": "Bahnschrift"},
                    className="card-text",
                ),
            ]
        ),
    ],
    style={"width": "20rem"},
    className="card text-white bg-primary mb-3",
)

# landing page
giris_sayfasi = dbc.Container([

            dbc.Row([
                dbc.Col(html.Br(), width=12),
                dbc.Col(giris_card, xs=12, sm=12, md=12, lg=5, xl=5),
                dbc.Col(lime_card,xs=12, sm=12, md=12, lg=4, xl=4),html.Br(),

                dbc.Col([   dbc.Col(html.Br(), ),
                            dbc.Col(html.Br(), ),
                            dbc.Col(html.Br(), ),
                            dbc.Col(html.Br(), ),
                            dbc.Col(shap_card ) ,
                  ], xs=12, sm=12, md=12, lg=3, xl=3),dbc.Button("Start ->", color="primary", href="/Veri_Yukle1" , n_clicks=0)])


                ])


data_upload2 =dbc.Card(
    [
        dbc.CardBody(
            [
                html.H6("This software supports data files with .xls, .xlsx, .sav, .csv and .txt extensions.", className="card-title"),
                dcc.Upload( id="upload-data1",  children=html.Div([dbc.Button("Choose ", color="primary", active=True, className="mb-3")]),
                        multiple=False, ),
            ]
        ),
    ],
    style={"width": "100%", 'textAlign':'center'},
    className="card text-white bg-info mb-3",
)


veri_analiz2 = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H6("Before the preprocessing steps, you need to analyze the data.", className="card-title"),
                html.Br(),
                dbc.Button("Analyze ", id = "analizbuton1" ,color="primary", n_clicks=0),
            ]
        ),
    ],
    style={"width": "100%"},
    className="card text-white bg-secondary mb-3",
)



offcanvas = html.Div(
    [ dbc.Offcanvas([
        dbc.Nav(
            [html.Div(
                                    [html.H6("Choose Model:",style={"font-family": "Bahnschrift"},),
                                dcc.Checklist(
    options=[
                {'label': 'Support Vector Machine', 'value': 'SVM'},
                {'label': 'Logistic Regression', 'value': 'LR'},
                {'label': 'Random Forest', 'value': 'RF'},
                {'label': 'Decision Tree', 'value': 'DT'},
                {'label': 'LightGBM', 'value': 'LGBM'},
                {'label': 'Gaussian Naive Bayes', 'value': 'GNB'},
                {'label': 'AdaBoost', 'value': 'ADA'},
                {'label': 'GradientBoosting', 'value': 'GBT'},
                {'label': 'CatBoost', 'value': 'CB'},
                {'label': 'XgBoost', 'value': 'XGB'},
                {'label': 'Multilayer perceptron(MLP) ', 'value': 'MLP'}

            ],
    value=['SVM', 'LR'],id="modelsecici1",
    labelStyle = {'display': 'block'}, className='radiobutton-group',
    style={"line-height":"28px","font-family": "Bahnschrift"},
)
],className="row flex-display",
                                ),html.Br(),

                                html.Div(
    [html.H6("Parameter optimization:",style={"font-family": "Bahnschrift"}),
        dcc.RadioItems(
            options=[
                {"label": "Yes", "value": '0'},
                {"label": "No", "value": '1'},

            ],id="optimizasyonslider1",
            value="1",
            inputStyle={"vertical-align":"middle", "margin":"auto"},
            labelStyle={"vertical-align":"middle"},
            style={"display":"inline-flex", "flex-wrap":"wrap", "justify-content":"space-between","line-height":"28px"},
            #labelStyle = {'display': 'block'},className='radiobutton-group',
            #style={"line-height":"28px","font-family": "Bahnschrift"},


        ),
        html.Div(id="optimizasyonsecenek1"),
            html.H6("Validation method:",style={"font-family": "Bahnschrift"}),

            dcc.RadioItems(  options=[
            {'label': 'Hold-out', 'value': '0'},
            {'label': 'Repeated Hold-out', 'value': '1'}, {'label': 'Stratified K Fold Cross Validation', 'value': '2'},{'label': 'Leave one out Cross Validation', 'value': '3'},{'label': 'Repeated Cross Validation', 'value': '4'} , {'label': 'Nested Cross Validation', 'value': '5'} ],id="veriseti-bolum-secici1",
            value='0',   labelStyle = {'display': 'block'},className='radiobutton-group',style={"line-height":"25px","font-family": "Bahnschrift"}, ),
                html.Br(),
                html.Div(id='bolme1'),


html.Br()

,

        # dbc.Button("Uygula", color="warning", size="md",active=True,className="fas fa-cogs",style={ 'color':'white' , 'border':'1.5px black solid',"font-family": "Bahnschrift"})
    ]
)



            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="offcanvas1",
    is_open=False,
    title="",
    style=SIDEBAR_STYLE,
    ),]
)

@app.callback(
    Output("bolme1", "children"),

    Input("veriseti-bolum-secici1", "value"),
    #State("veriseti-bolum-secici", "value" ),

    )
def validation(split):

    bolumleme=html.Div([html.H6("Select the training dataset percentage:", className="control_label",style={"font-family": "Bahnschrift"}),
                      dcc.Slider(50, 100, 5, value=80, id='traintestslider1', marks=None,
                      tooltip={"placement": "bottom", "always_visible": True} ),
                      html.Div(id="kfoldsecenegigoster1"),
                      ],className="row flex-display," )

    cv=html.Div(      [      html.H6("Select k fold:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(2, 10, 1, value=2, id='kfoldslider1', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} ) ])

    nsplit= html.Div(      [      html.H6("Select the number of repeat:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 50, 1, value=5, id='ntekrarslider1', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} ),html.Br(), html.H6("Select split size:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(0, 100, 5, value=50, id='ntekrarsplitslider1', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} )        ] )
    repeatcv= html.Div(      [      html.H6(" Select k fold:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 10, 1, value=5, id='repatkfoldslider1', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} ),html.Br(), html.H6("Select the number of repeat:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 10, 1, value=50, id='ntkrarslider1', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} )] )
    nestedcv= html.Div(      [      html.H6("Select Inner k fold:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 10, 1, value=5, id='nestedickfoldslider1', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} ),html.Br(), html.H6("Select Outer k fold:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 10, 1, value=50, id='diskfoldslider1', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} )] )





    if split== str(0):
        a = html.Div([ html.Div(bolumleme),
                       html.Div([nsplit, cv, repeatcv, nestedcv], style={'display': 'none'} ),
                        ])
    if split==str(1):
        a = html.Div([ html.Div(nsplit),
                       html.Div([cv, bolumleme, repeatcv, nestedcv], style={'display': 'none'} ),])
    if split==str(2):
        a = html.Div([
                  html.Div([bolumleme,nsplit, repeatcv, nestedcv], style={'display': 'none'} ),
                  html.Div(cv)
                  ])
    if split==str(3):
        a = html.Div([
                  html.Div([bolumleme,cv, nsplit, repeatcv, nestedcv], style={'display': 'none'} ),
                  ])
    if split==str(4):
        a = html.Div([
                  html.Div(bolumleme, style={'display': 'none'} ),
                  html.Div(nsplit, style={'display': 'none'} ),
                  html.Div(cv,style={'display': 'none'}),
                  html.Div(nestedcv,style={'display': 'none'}),
                  html.Div(repeatcv)
                  ])
    if split==str(5):
        a = html.Div([
                  html.Div([bolumleme, repeatcv], style={'display': 'none'} ),
                  html.Div(nsplit, style={'display': 'none'} ),
                  html.Div(cv,style={'display': 'none'}),
                  html.Div(nestedcv)
                  ])

    return a


veri_yukleme = dbc.Container([

            dbc.Row([
                dbc.Col(html.Br(), width=12),
                dbc.Col(data_upload2, xs=3, sm=3, md=3, lg=3, xl=3,  width={'offset': 0}),
                dbc.Col(html.Div(id="output-data-upload-21"),  xs=9, sm=9, md=9, lg=9, xl=9, width={'offset': 0})
])  ])


veri_onisleme = dbc.Container([

            dbc.Row([
                dbc.Col(html.Br(), width=12),
                dbc.Col(veri_analiz2, xs=3, sm=3, md=3, lg=3, xl=3),
                dbc.Col(html.Div(id="output-veri-analiz1"),  xs=9, sm=9, md=9, lg=9, xl=9 ),
                dbc.Col(html.Div(id="output-veri-analiz21"),  xs=9, sm=9, md=9, lg=9, xl=9 )
])  ])

################################################################################


parametre1 = dbc.Toast([
        offcanvas,
         dbc.Button("<<<<", id="open-offcanvas1", n_clicks=1, color="danger"),

], header="Choose modeling methods", className="mb-3", style={'width': '100%'})


parametre2 = dbc.Toast([
            html.Div(["This process may take a long time depending on the number of models you choose. Please wait."], className="h6"),
         dbc.Button( ">>>>" , id="sonuchesapla1", n_clicks=0 , color="success",  className="me-1", ),

], header="Calculate Results", className="mb-3", style={'width': '100%','margin-right' : '200px',})





modelsonuclari2 =   dbc.Spinner(children=[ dbc.Toast([], header="Model Results", id="modelsonucgoster1", className="mb-3", style={'width': '200%', 'height':'200%'} )
                        ], size="lg", color="primary", type="border" )


modelleme = dbc.Container([

            dbc.Row([
                dbc.Col(html.Br(), width=20),
                dbc.Col([parametre1, parametre2], xs=6, sm=6, md=3, lg=3, xl=3,  width={'offset': 0}),
                dbc.Col([html.Div([modelsonuclari2])],  xs=14, sm=14, md=17, lg=17, xl=17 ),
])  ])











################################################################################
aciklanabilir1 =   dbc.Spinner(children=[ dbc.Toast([], header="LIME", id="limegoster1", className="mb-3", style={'width': '100%'} )
                        ], size="lg", color="primary", type="border" )

aciklanabilir2 =   dbc.Spinner(children=[ dbc.Toast([], header="SHAP", id="shapgoster1", className="mb-3", style={'width': '130%'} )
                        ], size="xxl", color="primary", type="border" )

aciklanabilir3 =   dbc.Spinner(children=[ dbc.Toast(["Analysis may take a long time depending on the size of the dataset. "], header="General Information", id="shapgoster21", className="mb-3", style={'width': '130%'} )
                        ], size="lg", color="primary", type="border" )

aciklanabilir = dbc.Container([
html.Div([
            dbc.Row([
                dbc.Col(html.Br(), width=30),
                dbc.Col([aciklanabilir3], xs=30, sm=30, md=30, lg=30, xl=30 ),
                dbc.Col([aciklanabilir1], xs=30, sm=30, md=30, lg=30, xl=30,  style={'height': '1500%'}),
                
                
]) ]) ])

aciklanabilir_SHAP = dbc.Container([
html.Div([
            dbc.Row([dbc.Col(html.Br(), width=30),
dbc.Col([aciklanabilir2],  xs=30, sm=30, md=30, lg=30, xl=30,style={'height': '2000%'})]) ]) ])
################################################################################

navbar1 = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    dbc.Col(dbc.NavbarBrand("      XAI: Explainable Artificial Intelligence Interface ",style=dict(fontSize = 20,fontWeight='bold',fontFamily= "Bahnschrift", lineHeight= 1), className="ms-2")),
                    align="center",
                    className="g-0",
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.Row(
                [
                    dbc.NavbarToggler(id="navbar-toggler1"),
                    dbc.Collapse(
                        dbc.Nav(
                            [
                                dbc.NavItem(dbc.NavLink("      Introduction", href="/page_1", id="page-1-link11",active="exact",className="fas fa-home")),
                                dbc.NavItem(dbc.NavLink("      File Upload ", href="/page_2" ,id="page-2-link11",active="exact",className="fas fa-folder-open")),
                                dbc.NavItem(dbc.NavLink("      Data Preprocessing", href="/page_3", id="page-3-link11",active="exact",className="fas fa-cogs")),
                                dbc.NavItem(dbc.NavLink("      Modelling", href="/page_4", id="page-4-link11",active="exact",className="fas fa-laptop")),
                                dbc.NavItem(dbc.NavLink(       "LIME ", href="/page_5",id="page-5-link11",active="exact",className="fas fa-laptop")),
                                dbc.NavItem(dbc.NavLink(       "SHAP ", href="/page_6",id="page-6-link11",active="exact",className="fas fa-laptop")),

                                dbc.NavItem(
                                    dbc.NavLink("     Citation ", href="/page_7", id="page-7-link11",active="exact",className="far fa-chart-bar"),

                                    className="me-auto",
                                ),


                                dbc.NavItem(dbc.Button("Turkish", href='/apps/app1',color="success", size="sm",active=True,className="mr-1",id='btn-nclicks-11',style={ 'color':'white' , 'border':'1.5px black solid'}),),
                                dbc.NavItem(dbc.Button("English", href='/apps/app2',color="success", size="sm",active=True,className="mr-1",id='btn-nclicks-21',style={ 'color':'white' , 'border':'1.5px black solid'}),
  ),
                            ],
                            # make sure nav takes up the full width for auto
                            # margin to get applied
                            className="w-100",
                        ),
                        id="navbar-collapse1",
                        is_open=False,
                        navbar=True,
                    ),
                ],
                # the row should expand to fill the available horizontal space
                className="flex-grow-1",
            ),
        ],
        fluid=True,
    ),
    dark=True,
    #brand_href="/apps/app1",
    color="#F57F17",
)

gereksiz= html.Div([
dcc.Slider(50, 100, 5, value=80, id='traintestslider1', marks=None, tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(0, 10, 1, value=0, id='kfoldslider1', marks=None,   tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(5, 50, 1, value=5, id='ntekrarslider1', marks=None,      tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(0, 100, 5, value=50, id='ntekrarsplitslider1', marks=None,    tooltip={"placement": "bottom", "always_visible": True} ) ,
dcc.Slider(5, 10, 1, value=5, id='nestedickfoldslider1', marks=None,  tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(5, 10, 1, value=50, id='diskfoldslider1', marks=None, tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(5, 10, 1, value=5, id='repatkfoldslider1', marks=None, tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(5, 10, 1, value=50, id='ntkrarslider1', marks=None, tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Dropdown([0],id='pozitif_deger1'),
dcc.RadioItems( options=[],  value='5', id = 'smoteradio1', inline=False,),
dcc.Slider(1, 10, 1, value=2, id='optimizekfoldslider1', marks=None,    tooltip={"placement": "bottom", "always_visible": True} ),
])

content = html.Div( children=giris_sayfasi, id="page-content11",)

layout = html.Div([ offcanvas,
                    html.Div([gereksiz], style={'display': 'none'}),
                    navbar1,     dcc.Location(id="urla1"),     content,
                    html.Div(id='verideposu1', style={'display': 'none'}),
                    html.Div(id='ozellikdf1', style={'display': 'none'}),
                    html.Div(id='hedefdf1', style={'display': 'none'}),])


################################################################################
@app.callback(
    [Output('page-1-link11', 'active'),Output('page-2-link11', 'active'),Output('page-3-link11', 'active'),Output('page-4-link11', 'active'),Output('page-5-link11', 'active'),Output('page-6-link11', 'active'),Output('page-7-link11', 'active')],
    [Input('urla1', 'pathname')],
)
def toggle_active_links(pathname):
    if pathname == '/':
        # Treat page 1 as the homepage / index
        return True, False, False,False, False,False, False
    return [pathname == '/page-{i}' for i in range(1, 8)]
#sayfa geçişleri
@app.callback(Output("page-content11", "children"), [Input("urla1", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page_1"]:
        return giris_sayfasi
    elif pathname == "/page_2":
        return veri_yukleme
    elif pathname == "/page_3":
        return veri_onisleme
    elif pathname == "/page_4":
        return modelleme
    elif pathname == "/page_5":
        return aciklanabilir
    elif pathname == "/page_6":
        return aciklanabilir_SHAP
    elif pathname == "/page_7":
        return html.Div()

    else:
        return giris_sayfasi

################################################################################
 #canvas

@app.callback(
    Output("offcanvas1", "is_open"),
    Input("open-offcanvas1", "n_clicks"),
    [State("offcanvas1", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

################################################################################
#data uploaded

@app.callback(Output('verideposu1', 'children'),
              Input('upload-data1', 'contents'),
              State('upload-data1', 'filename'),  )


def updateupload(contents, filename):

    if contents is not None:
        jsonfile = function_1.dosyaoku(contents, filename)
        return jsonfile


################################################################################

@app.callback(  Output('output-data-upload-21', 'children'),
                Input('verideposu1', 'children') )

def alanupdate(veri):
    if veri is not None:
        try:
            dff=pd.read_json(veri , orient='split')
            oznitelik_secici= dcc.Dropdown(dff.columns, id='oznitelik_column1',
            multi=True, className="", value=dff.columns, placeholder="Select only one variable for the target/output attribute:")
            hedef_secici= dcc.Dropdown(dff.columns, id='hedef_column1',  placeholder="",
            multi=False, className="")
            # pozitif_secici= dcc.Dropdown(dff.columns, id='pozitif_deger',  placeholder="İlgilenilen Sınıfı Seçiniz:",
            # multi=False, className="")

            toast0 = dbc.Toast([function_1.veriyazdir(veri)], header="Uploaded Dataset", className="mb-3", style={'width': '100%'})
            toast1 = dbc.Toast(   [oznitelik_secici], header="Select predictive attributes", className="mb-3", style={'width': '100%'})
            toast2 = dbc.Toast(   [hedef_secici],    header="Select target/output attribute", className="mb-3", style={'width': '100%'})
            toast3 = dbc.Toast(   [html.Div(id="pozitifsecici1") ],   header="Select the class of interest", className="mb-3", style={'width': '100%'})
            ozelliksecici = dbc.Row([   dbc.Col(toast1), dbc.Col(toast2), dbc.Col(toast3), ])
            bilgiver = html.Div( [  dbc.Button(    "Save",      id="veri_kaydet1",   color="info",
                                                className="mb-3",     n_clicks=0,   ),
                                    dbc.Toast(    [html.P("You can proceed to the data preprocessing step", className="mb-0"),
                                    html.Br(),
                                    dbc.Button('Data Preprocessing->', href="/page_3", color="success")],     id="simple-toast1",
                                                header="Successfully saved",
                                                dismissable=True,     is_open=False,
                                                style={"position": "fixed", "top": 20, "right": 10, "width": 250},   ),       ] )



            return html.Div([toast0, ozelliksecici, html.Hr(), bilgiver ])




        except Exception as e:
            print(e)
            return html.Div([html.Hr(),
                         html.H4('File entry is not made.'),])


################################################################################
@app.callback(
    Output("pozitifsecici1", "children"),
    Input("hedef_column1","value"),
    State("verideposu1", "children"), )

def pozitifsinifsec(sinif, veri):
    if sinif is not None:

        dff=pd.read_json(veri , orient='split')
        print(dff[sinif].unique())
        a = html.Div(dcc.Dropdown(dff[sinif].unique(), id='pozitif_deger1',  placeholder="Select the class of interest:", multi=False, className=""))
        return a






######################################################
@app.callback(
    Output("simple-toast1", "is_open"),
    Output("ozellikdf1","children"),
    Output("hedefdf1","children"),
    Input("veri_kaydet1", "n_clicks"),
    State("oznitelik_column1", "value"),
    State("hedef_column1", "value"),
    State("verideposu1", "children"),
    State("pozitif_deger1","value")
)
def open_toast(n, ozellik, hedef, veri, psinif):
    global df
    global oz_df
    global hedef_df
    global pozitif_sinif_ismi
    if n != 0 and ozellik is not None and hedef is not None and veri is not None and psinif is not None:
        pozitif_sinif_ismi = psinif
        dff=pd.read_json(veri , orient='split')
        ozellikdf = dff[ozellik]
        try:
            ozellikdf = ozellikdf.to_frame()
        except:
            ozellikdf = ozellikdf

        hedefdf= dff[hedef]

        df=dff
        oz_df=ozellikdf
        hedef_df = hedefdf

        ozellikdf = ozellikdf.to_json(date_format='iso', orient='split' )
        hedefdf = hedefdf.to_json(date_format='iso', orient='split' )
        print(pozitif_sinif_ismi)
        return True , ozellikdf, hedefdf
    return False , "" ,""



################################################################################

modal = html.Div(
    [
        dcc.Download(id="download-dataframe-xlsx1"),
        dbc.Button("Download preprocessed dataset", id="open1", n_clicks=0),
        dbc.Modal(
            [
               dbc.ModalHeader(dbc.ModalTitle("")),
                dbc.ModalBody("This is the content of the modal", id= "onislemverisi1"),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close1", className="ms-auto", n_clicks=0
                    )
                ),
            ],
            id="modal1",
            size="xl",
            is_open=False,
        ),
    ]
)



@app.callback(
    Output("output-veri-analiz1", "children"),
    Input("analizbuton1", "n_clicks"), )
def analizfonksiyon(n):
    global df, oz_df, hedef_df
    global new_df, new_oz_df, new_hedef_df
    OZ_YENİ=oz_df.copy()
    OZ_YENİ , onehot_liste = function_1.categoric1(OZ_YENİ)
    if n != 0 and df is not None and oz_df is not None and hedef_df is not None:
        # missing data
        kayipverisonuc=function_1.is_kayipveri(oz_df)
        # is an outlier
        aykiriverisonuc = function_1.is_aykiriveri(df)
        
        dengesizverisonuc = function_1.isdengesiz(hedef_df)
        #analizverileri = function_1.verianalizi(df, oz_df, hedef_df)
        kayipveri1= html.Div([dbc.Card([  dbc.CardBody([ html.H6("Missing Data Analysis Results: ", className="card-title"),
                                html.P( str(kayipverisonuc) + " missing data found. ", className="card-text"),
                                html.Div( dcc.RadioItems( options=[  {'label': 'Remove rows with missing values from the dataset. ', 'value': '0'}, {'label': 'Let the assignment be made with the Random Forest method.', 'value': '1'} ],
                                            value='1', id = 'kayipdegerradio1', inline=False,
                                            className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                            style={"width": "100%"},  className="card text-white bg-danger mb-3", ) ]  )
        kayipveri2= dbc.Card([  dbc.CardBody([ html.H6("Missing Data Analysis Result: ", className="card-title"),
                                        html.P( "There is no missing data in the data set.", className="card-text"), ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-success mb-3", )

        aykiriveri1= dbc.Card([  dbc.CardBody([ html.H6("Outlier Value Analysis Result: ", className="card-title"),
                                        html.P( str(aykiriverisonuc) + " outlier values were found. Delete outliers values?.", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'Yes', 'value': '0'}, {'label': 'No', 'value': '1'} ],
                                                    value='1', id = 'aykiridegerradio1', inline=False,
                                                    className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-danger mb-3", )
        aykiriveri2= dbc.Card([  dbc.CardBody([ html.H6("Outlier Value Analysis Result:  ", className="card-title"),
                                                html.P( "There is no outlier value in the dataset.", className="card-text"), ] ),  ],
                                                            style={"width": "100%"},  className="card text-white bg-success mb-3", )




        veridonusum1= dbc.Card([  dbc.CardBody([ html.H6("Transformation methods", className="card-title"),
                                        html.P( "Please choose one of the following methods for data transformation.", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'Normalization', 'value': '0'},
                                                                             {'label': 'Min-Max Standardization', 'value': '1'},
                                                                             {'label': 'Standardization', 'value': '2'},
                                                                             {'label': 'Robust Standardization', 'value': '3'},
                                                                             {'label': 'None', 'value': '4'}, ],
                                                    value='4', id = 'veridonusumradio1', inline=False,
                                                    className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-info mb-3", )



        ozsecimi1= dbc.Card([  dbc.CardBody([ html.H6("Attribute Selection", className="card-title"),
                                        html.P( "Select one of the following methods for attribute selection", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'Recursive Feature Elimination (RFE)', 'value': '0'},
                                                                             {'label': 'Based on Extra Trees Classifier', 'value': '1'},
                                                                             {'label': 'Based on Random Forest Classifier', 'value': '2'},
                                                                             {'label': 'LASSO', 'value': '3'},
                                                                             {'label': 'Boruta', 'value': '4'},
                                                                              {'label': 'None', 'value': '5'},],
                                                        value='5', id = 'ozscmradio1', inline=True,
                                                       className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-info mb-3", )

        print(onehot_liste)
        print("smoteöncesionehotliste")
        if dengesizverisonuc==True and len(onehot_liste) == 0:
            smote1= dbc.Card([  dbc.CardBody([ html.H6("Class Imbalance Analysis", className="card-title"),
                                        html.P( "There is a class imbalance problem in the dataset. Select one of the following methods to resolve the class imbalance problem.", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'SMOTE', 'value': '0'},{'label': 'SMOTETomek', 'value': '1'},{'label': 'None', 'value': '2'}],

                                                    value='2', id = 'smoteradio1', inline=False,
                                                    className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-info mb-3", )
        if dengesizverisonuc==True and len(onehot_liste) !=0:
            smote1= dbc.Card([  dbc.CardBody([ html.H6("Class Imbalance Analysis", className="card-title"),
                                        html.P( "There is a class imbalance problem in the dataset. Select one of the following methods to resolve the class imbalance problem.", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'None', 'value': '2'},{'label': 'SMOTE-NC', 'value': '3'}],

                                                    value='2', id = 'smoteradio1', inline=False,
                                                    className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-info mb-3", )
        if dengesizverisonuc ==False:
            smote1= dbc.Card([  dbc.CardBody([ html.H6("Class Imbalance Analysis", className="card-title"),
                                        html.P( "There is no class imbalance problem in the dataset.", className="card-text") ]) ])


        if kayipverisonuc ==0:
            kayipveri3=html.Div([
                        html.Div([kayipveri1], style={'display': 'none'} ),
                        html.Div([kayipveri2] )  ])
        else:
            kayipveri3=html.Div([
                                    html.Div([kayipveri1] ),
                                    html.Div([kayipveri2] , style={'display': 'none'} )  ])


        if aykiriverisonuc ==0:
            aykiriveri3=html.Div([
                        html.Div([aykiriveri1], style={'display': 'none'} ),
                        html.Div([aykiriveri2] )  ])
        else:
            aykiriveri3=html.Div([
                                    html.Div([aykiriveri1] ),
                                    html.Div([aykiriveri2] , style={'display': 'none'} )  ])






        verionsilemekaydet = html.Div([dbc.Button('Save', id="verionislemkaydet1", color="primary", n_clicks=0,className="me-1" ),

                                        dbc.Toast(    [html.P("You can move on to the modeling phase.", className="mb-0"),
                                            html.Br(),
                                            dbc.Button('Modeling->', href="/page_4", color="success"),html.Br(),html.Br(),
                                            dbc.Row([dbc.Col(modal)])],     id="simple-toast21",
                                            header="Successfully saved.",
                                            dismissable=True,     is_open=False,
                                            style={"position": "fixed", "top": 66, "right": 10, "width": 250},  ),html.Br(),


                                        ])


        return dbc.Toast([kayipveri3, aykiriveri3, veridonusum1, ozsecimi1, smote1, verionsilemekaydet ], header="Data Analysis Results", style={'width': '100%'})
    elif n != 0:
        return dbc.Toast(['Data Entry Failed Return to Data Loading Page!!'], header="Data Analysis Results", style={'width': '100%'})





@app.callback(
    Output("modal1", "is_open"),
    Output("onislemverisi1", "children"),
    Input("open1", "n_clicks"),
    Input("close1", "n_clicks"),
    State("modal1", "is_open"), )
def toggle_modal(n1, n2, is_open):
    global new_df
    table = dash_table.DataTable(  data=new_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in new_df.columns],editable=False,

                        page_action="native",
                        #page_current= 0,
                        #page_size= 10, style_table={'overflowX': 'auto'},  )
                        style_table={
            'width': '100%',
                 'height': '300px',
            'overflowY': 'scroll',
            'overflowX': 'scroll',
            'textAlign': 'center',

        },style_header=
                                {
                                    'fontWeight': 'bold',
                                    'border': 'thin lightgrey solid',
                                    'backgroundColor': 'rgb(100, 100, 100)',
                                    'color': 'white',
                                    'textAlign': 'center',
                                },style_cell={
                                    "font-family": "Bahnschrift",
                                    'textAlign': 'center',
                                    'width': '150px',
                                    'minWidth': '180px',
                                    'maxWidth': '180px',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',

                                })
    downloadbuton=dbc.Button("Download" , id="excelkaydet1", n_clicks=0)
    if n1 or n2:
        return not is_open, html.Div([table, downloadbuton] )
    return is_open, html.Div([table, downloadbuton] )


@app.callback(
    Output("download-dataframe-xlsx1", "data"),
    Input("excelkaydet1", "n_clicks") )

def veridownload(n):
    global new_df
    if n!=0:
        return dcc.send_data_frame(new_df.to_excel, "mydf.xlsx", sheet_name="Sheet_name_1")







@app.callback(
    Output("simple-toast21", "is_open"),
    Input("verionislemkaydet1", "n_clicks"),
    State("kayipdegerradio1", "value" ),
    State("aykiridegerradio1", "value"),
    State("veridonusumradio1", "value"),
    State("ozscmradio1", "value"),
    State("smoteradio1", "value"),
    )
def onislemkaydetfonk(n, kayipsecim, aykirisecim, donusum, ozelliksecim,smote):
    global df, oz_df, hedef_df
    global new_df, new_oz_df, new_hedef_df
    global new_oz_onehot, onehot_liste
    new_oz_df = oz_df.copy()
    new_hedef_df = hedef_df.copy()
    new_df= pd.concat([new_oz_df,new_hedef_df],axis=1)
    print("ilk hali")
    print(new_df.info())
    new_df = function_1.stringtofloat(new_df)
    new_hedef_df = new_df[hedef_df.name]
    new_oz_df = new_df.drop([hedef_df.name], axis=1)
    new_oz_df , onehot_liste = function_1.categoric1(new_oz_df)
    print("ikinci hali")
    print(new_df.info())



    if n != 0:
        allcolumns = new_df.columns.values
        xcolumns = new_oz_df.columns.values
        y_name = hedef_df.name

        if kayipsecim == str(0):
            adf = new_df.dropna()
            new_df = adf
            new_df = new_df.reset_index(drop=True)
            new_hedef_df = new_df[hedef_df.name]
            new_oz_df = new_df.drop([hedef_df.name], axis=1)
            print("üçüncü hali")
            print(new_df.info())

        if kayipsecim == str(1):
            new_hedef_df = new_df[hedef_df.name]
            new_oz_df = new_df.drop([hedef_df.name], axis=1)
            new_oz_df = function_1.kayipveritamamla(new_oz_df)

            aaa = pd.DataFrame(new_oz_df)
            for i in range(len(xcolumns)):
                aaa= aaa.rename(columns={i:xcolumns[i]})
            new_oz_df = aaa
            new_df= pd.concat([new_oz_df,new_hedef_df],axis=1)
            print("4 hali")
            print(new_df.info())
            # new_df = new_df.reset_index(drop=True)

        if aykirisecim== str(0):
            new_df = function_1.aykiriveritamamla(new_df)
            new_hedef_df = new_df[hedef_df.name]
            new_oz_df = new_df.drop([hedef_df.name], axis=1)
            print("5 hali")
            print(new_df.info())

        if donusum == str(0) or donusum== str(1) or donusum== str(2) or donusum== str(3):

            new_hedef_df = new_df[hedef_df.name]
            new_oz_df = new_df.drop([hedef_df.name], axis=1)
            new_oz_df = function_1.veridonusumfonk(new_oz_df, donusum)
            aaa = pd.DataFrame(new_oz_df)
            for i in range(len(xcolumns)):
                aaa= aaa.rename(columns={i:xcolumns[i]})
            new_oz_df = aaa

            new_df= pd.concat([new_oz_df,new_hedef_df],axis=1)
            print("6 hali")
            print(new_df.info())


        if ozelliksecim == str(0) or ozelliksecim == str(1) or ozelliksecim ==str(2) or ozelliksecim ==str(3) or ozelliksecim ==str(4):
            new_columns = function_1.ozelliksecimfonk(new_df,new_oz_df,new_hedef_df, ozelliksecim)
            new_oz_df = new_oz_df[new_columns]
            new_df= pd.concat([new_oz_df,new_hedef_df],axis=1)
            print("6 hali")
            print(new_df.info())


        if smote == str(0) or smote == str(1) or smote == str(3):
            xcolumns_1 = new_oz_df.columns.values
            new_oz_df,new_hedef_df = function_1.smotefonk(new_df,new_oz_df,new_hedef_df, smote)
            aaa = pd.DataFrame(new_oz_df)
            for i in range(len(xcolumns_1)):
                aaa= aaa.rename(columns={i:xcolumns_1[i]})
            new_oz_df = aaa
            new_df= pd.concat([new_oz_df,new_hedef_df],axis=1)
            print("smote sonrası")
            print(new_df.info())


        new_hedef_df = new_df[hedef_df.name]
        new_oz_df = new_df.drop([hedef_df.name], axis=1)

        new_oz_df , onehot_liste = function_1.categoric1(new_oz_df)
        new_oz_onehot = function_1.onehotfonk(new_oz_df, onehot_liste )
        
        print("son hali")
        print(new_oz_onehot.info())


        # function_1.veriyazdir1(new_df)
        return True




x_train= None
x_test= None
y_train = None
y_test = None
modeller = {}
model_cm = []
model_puanlari = {}
model_puanlari1 = {}
model_puanlari2 = {}
model_puanlari3 = {}
model_puanlari4 = {}
model_puanlari5 = {}
model_puanlari6 = {}
model_puanlari7 = {}
shaptype = None
agactabanli={}







################################

@app.callback(
    Output("optimizasyonsecenek1", "children"),
    Input("optimizasyonslider1", "value"), )

def opt(n):

    if n!=str(0):
        return html.Div([dcc.Slider(1, 10, 1, value=2, id='optimizekfoldslider1')], style={'display': 'none'} )
    if n==str(0):
        aa = dcc.Slider(2, 10, 1, value=2, id='optimizekfoldslider1', marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True} ),
        return aa











################################################################################
#sonuç hesaplamak
import numpy as np
import time
@app.callback(
    Output("modelsonucgoster1", "children"),
    Input("sonuchesapla1", "n_clicks"),
    State("optimizasyonslider1","value"),
    State("veriseti-bolum-secici1", "value" ),
    State("traintestslider1", "value"),
    State("ntekrarslider1", "value"),
    State("ntekrarsplitslider1", "value"),
    State("kfoldslider1", "value"),
    State("repatkfoldslider1", "value"),
    State("ntkrarslider1", "value"),
    State("nestedickfoldslider1", "value"),
    State("diskfoldslider1", "value"),
    State("modelsecici1", "value"),
    State("optimizekfoldslider1", "value"),

    )

def sonuchesaplafonk(n, optimizasyonslider, method, trainslider, rholdslider,rholdsplitsizeslider, skfoldslider,kfold, nrepeat,nestedickfoldslider,nesteddiskfoldslider,   models, optsecenek):
    #print(method, trainslider, rholdslider,rholdsplitsizeslider, skfoldslider,nestedickfoldslider,nesteddiskfoldslider, models, optimizasyonslider)
    #print(str(optimizasyonslider[0]))
    if n!=0:
        global new_df, new_oz_df, new_hedef_df,new_oz_onehot, onehot_liste
        global modeller,  model_puanlari,model_puanlari1,model_puanlari2,model_puanlari3,model_puanlari4,model_puanlari5,model_puanlari6,model_puanlari7
        modeller = {}
        model_puanlari = {}
        model_puanlari1 = {}
        model_puanlari2 = {}
        model_puanlari3 = {}
        model_puanlari4 = {}
        model_puanlari5 = {}
        model_puanlari6 = {}
        model_puanlari7 = {}


        global shaptype

        if new_df is None or new_oz_df is None or new_hedef_df is None:
            return html.H5("The preprocessing steps are not completed. Return to data preprocessing page.")
        global x_train, x_test, y_train, y_test
        if method==str(0) and optimizasyonslider ==str(0):
            shaptype=1
            x_train, x_test, y_train, y_test = train_test_split(new_oz_onehot, new_hedef_df, test_size = (100-int(trainslider))/100, random_state = 0)
            for i in models:
                   modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i]= function_1.siniflandirma_2(x_train,y_train,x_test,y_test,optsecenek,i)
            print("optimizeholdout çalıştı")
        elif method==str(0) and optimizasyonslider !=str(0):
            shaptype=2
            x_train, x_test, y_train, y_test = train_test_split(new_oz_onehot, new_hedef_df, test_size = (100-int(trainslider))/100, random_state = 0)
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i]= function_1.siniflandirma_3(x_train,y_train,x_test,y_test,i)
            print("tekholdoutcalıştı")
        elif method==str(1) and optimizasyonslider ==str(0):
            shaptype=3
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i]= function_1.repeatholdoutoptimize(new_oz_onehot, new_hedef_df, optsecenek, rholdslider, rholdsplitsizeslider, i)
            print("optimizerepetedholdout çalıştı")
        elif method==str(1) and optimizasyonslider !=str(0):
            shaptype=4
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i]= function_1.repeatedholdout(new_oz_onehot, new_hedef_df, rholdslider, rholdsplitsizeslider, i)
            print("repetedholdout çalıştı")
        elif method==str(2) and optimizasyonslider ==str(0):
            shaptype=5
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i]= function_1.skfoldcrossoptimize(new_oz_onehot, new_hedef_df,optsecenek,skfoldslider,i)
        elif method==str(2) and optimizasyonslider !=str(0):
            shaptype=6
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i]= function_1.stratifiedcrossvalidation(new_oz_onehot, new_hedef_df, skfoldslider, i)
        elif method==str(3) and optimizasyonslider ==str(0):
            shaptype=7
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i] = function_1.loocvoptimize(new_oz_onehot, new_hedef_df,optsecenek,i)
        elif method==str(3) and optimizasyonslider !=str(0):
            shaptype=8
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i] = function_1.leaveoneout(new_oz_onehot, new_hedef_df, i)
        elif method==str(4) and optimizasyonslider ==str(0):
            shaptype=9
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i] = function_1.repeatedcvoptimize(new_oz_onehot, new_hedef_df,optsecenek,kfold,nrepeat, i)
        elif method==str(4) and optimizasyonslider !=str(0):
            shaptype=10
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i] = function_1.repeatedcrossvalidation(new_oz_onehot, new_hedef_df,kfold,nrepeat, i)
        elif method==str(5):
            shaptype=11
            for i in models:
                modeller[i], model_puanlari[i], model_puanlari1[i], model_puanlari2[i], model_puanlari3[i],model_puanlari4[i],model_puanlari5[i],model_puanlari6[i],model_puanlari7[i]= function_1.nestedcross(new_oz_onehot, new_hedef_df,nestedickfoldslider,nesteddiskfoldslider,i)



        adf = pd.DataFrame( {"Model": model_puanlari.keys(), "Accuracy": model_puanlari.values(),"F1_score": model_puanlari1.values(),"Precision": model_puanlari2.values(),"Recall": model_puanlari3.values(),"AUC": model_puanlari4.values(), "FPR": model_puanlari5.values(), "TPR": model_puanlari6.values(),"NPV": model_puanlari7.values()} )



        sonuclar = html.Div([dash_table.DataTable(
                data=adf.to_dict('records'),
                columns=[{"name": i, "id": i} for i in adf.columns],style_table={
            'width': '100%',
                 'height': '500px',

            'textAlign': 'center',

        },style_header=
                                {
                                    'fontWeight': 'bold',
                                    'border': 'thin lightgrey solid',
                                    'backgroundColor': 'rgb(100, 100, 100)',
                                    'color': 'white',
                                    'textAlign': 'center',
                                },
                                style_cell={
                                    "font-family": "Bahnschrift",
                                    'textAlign': 'center',
                                    'width': '60px',
                                    'minWidth': '60px',
                                    'maxWidth': '60px',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',

                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(248, 248, 248)'
                                    },
                                    {
                                        'if': {'column_id': 'country'},
                                        'backgroundColor': '#FF8F00',
                                        'color': 'black',
                                        'fontWeight': 'bold',
                                        'textAlign': 'center'
                                    }],
            ),
            # html.Br(), html.Div([figur]),
            dbc.Button("LIME >>", href="/page_5"),
            ]),

        print(model_puanlari)

        return sonuclar

#####################################

@app.callback(Output('limegoster1', 'children'),
              Input('urla1', 'pathname'))
def display_page(pathname):
    global new_df, new_oz_df, new_hedef_df,new_oz_onehot, onehot_liste
    global modeller,  model_puanlari,model_puanlari1,model_puanlari2,model_puanlari3,model_puanlari4,model_puanlari5,model_puanlari6,model_puanlari7
    if pathname == '/page_5':
        try:
            if modeller == {} or new_df is None:
                return html.Div("The classifier did not work. First, do the classification.")
            else:
                modellistesi = list(modeller.keys())
                limesecici= dcc.Dropdown( modellistesi , id='limemodel1', clearable=False, multi=False, className="", value=modellistesi[0])
                limetablo = dash_table.DataTable( id='limetablo11',
                    columns=[ {"name": i, "id": i} for i in new_df.columns ],
                        data=new_df.to_dict('records'),
                        editable=False,
                        row_selectable="single",
                        selected_rows=[0],
                        page_action="native",
                        #page_current= 0,
                        #page_size= 10, style_table={'overflowX': 'auto'},  )
                        style_table={
            'width': '100%',
                 'height': '200px',
            'overflowY': 'scroll',
            'overflowX': 'scroll',
            'textAlign': 'center',

        },style_header=
                                {
                                    'fontWeight': 'bold',
                                    'border': 'thin lightgrey solid',
                                    'backgroundColor': 'rgb(100, 100, 100)',
                                    'color': 'white',
                                    'textAlign': 'center',
                                },style_cell={
                                    "font-family": "Bahnschrift",
                                    'textAlign': 'center',
                                    'width': '150px',
                                    'minWidth': '180px',
                                    'maxWidth': '180px',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',

                                },)
                limesonucgoster = html.Div(id='limesonuc11')
                shapsecici= dcc.Dropdown( modellistesi , id='shapmodel1', clearable=False,  multi=False, className="", value=modellistesi[0])


                return html.Div([limesecici,html.Br(), limetablo, html.Br(), limesonucgoster])

        except:
            return html.Div("The classifier did not work. First, do the classification.")










import matplotlib.pyplot as plt


@app.callback(Output('limesonuc11', 'children'),
              Input('limetablo11',"derived_virtual_selected_rows"),
              Input('limemodel1', 'value'), )

def limehesapla(row, model):
    global new_df, new_oz_df, new_hedef_df,new_oz_onehot, onehot_liste
    global modeller,  model_puanlari,model_puanlari1,model_puanlari2,model_puanlari3,model_puanlari4,model_puanlari5,model_puanlari6,model_puanlari7
    global x_train,  x_test,  y_train ,  y_test
    global pozitif_sinif_ismi

    try:
        classifier = modeller[model]
        X = new_oz_onehot.to_numpy()
        print("X")
        Y = new_hedef_df.to_numpy()
        # df,categorical_features = function_1.categoric1(new_oz_onehot)
        # print(categorical_features)

        categorical_features = np.argwhere(np.array([len(set(X[:,x])) for x in range(X.shape[1])]) <= 10).flatten()

        print(categorical_features)
        explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=list(new_oz_onehot.columns), categorical_features=categorical_features, class_names=new_hedef_df.unique())
        print('lime çalıştı')
        exp = explainer.explain_instance(X[row[0]], classifier.predict_proba)
        print("lime hesapladı")
        exp.as_html()
        limelist=pd.DataFrame(exp.as_list())
        

        obj = html.Iframe(
            # Javascript is disabled from running in an IFrame for security reasons
            # Static HTML only!!!
        srcDoc= exp.as_html(),
            width='100%',
            height='600px',
            style={'border': '2px #d3d3d3 solid'},
        )

        sonuclar = html.Div([dash_table.DataTable(
                data=limelist.to_dict('records'),
                columns=[{"name": i, "id": i, "deletable": False, "selectable": False  } for i in limelist.columns],editable=False,
                #cols[0]["name"] == "Attribute",  
                
                
              
                        page_action="native",
                        page_current= 0,
                        page_size= 10,   
                        style_table={
            'width': '100%',
                 'height': '600px',
            'overflowY': 'scroll',
            'overflowX': 'scroll',
            'textAlign': 'center',

        },style_header=
                                {
                                    'fontWeight': 'bold',
                                    'border': 'thin lightgrey solid',
                                    'backgroundColor': 'rgb(100, 100, 100)',
                                    'color': 'white',
                                    'textAlign': 'center',
                                },style_cell={
                                    "font-family": "Bahnschrift",
                                    'textAlign': 'center',
                                    'width': '150px',
                                    'minWidth': '180px',
                                    'maxWidth': '180px',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',

                                }, 
            ),dbc.Button("SHAP >>", href="/page_6"),])
        
    


        return obj,html.Br(),html.Br(),sonuclar
    except Exception as e:
        print(e)
        return "LIME could not be calculated"
from shap.plots._beeswarm import summary_legacy


@app.callback(Output('shapgoster1', 'children'),
              Input('urla1', 'pathname'))
def display_page(pathname):
    global new_df, new_oz_df, new_hedef_df,new_oz_onehot, onehot_liste
    global modeller,  model_puanlari,model_puanlari1,model_puanlari2,model_puanlari3,model_puanlari4,model_puanlari5,model_puanlari6,model_puanlari7
    if pathname == '/page_6':
        try:
            if modeller == {} or new_df is None:
                return html.Div("The classifier did not work. First, do the classification.")
            else:
                modellistesi = list(modeller.keys())
                shapturleri = ["Shap Summary Plot", "Attribute Importance Plot"]
                shapsonucgoster = html.Div(id='shapsonuc11')



                print(new_hedef_df.unique())

                #a = html.Div(dcc.Dropdown(new_hedef_df.unique(), id='pozitif_deger2',  placeholder="Pozitif sınıfı seçiniz:", multi=False, className=""))
                shapsecici= dcc.Dropdown( modellistesi , id='shapmodel1', clearable=False,  multi=False, className="", value=modellistesi[0])
                shapsecici2= dcc.Dropdown( shapturleri , id='shaptur1', clearable=False,  multi=False, className="", value=shapturleri[0])


                return html.Div([shapsecici, shapsecici2,  html.Br(), shapsonucgoster ])

        except:
            return html.Div("The classifier did not work. First, do the classification.")

@app.callback(Output('shapsonuc11', 'children'),
              Input('shapmodel1', 'value'),
              Input('shaptur1', 'value') )

def shaphesapla(model, tur):
    global new_df, new_oz_df, new_hedef_df, new_oz_onehot, onehot_liste
    global modeller,  model_puanlari,model_puanlari1,model_puanlari2,model_puanlari3,model_puanlari4,model_puanlari5,model_puanlari6,model_puanlari7
    global x_train,  x_test,  y_train ,  y_test
    global shaptype
    global pozitif_sinif_ismi

   # pozitif_sinif_index = list(y_train.unique()).index(sinif)
    print("pozitif sınıf index")
    #print(pozitif_sinif_index)
    agactabanli = ['RF', 'DT','CB','GBT','XGB']
    kernel = ['SVM', 'LR','GNB','LGBM','ADA','MLP']


    try:
        if x_test is None:
            x_train, x_test, y_train, y_test = train_test_split(new_oz_onehot, new_hedef_df, test_size = 0.2, random_state = 0)
        classifier = modeller[model]
        if  model in agactabanli:
            print("ağaçtabanlı shap")
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(x_test)
            print(np.array(shap_values).shape)
            shap_values2= shap_values[1][:]
            shap_values2= pd.DataFrame(shap_values2)
            vals = np.abs(shap_values2.values).sum(0)
            vals=vals/vals.sum()

            a = new_oz_onehot.columns

            shap_importance = {}
            for i in range(len(a)):
                shap_importance[a[i]] = vals[i]

            adf = pd.DataFrame( {"Attribute": shap_importance.keys(), "Attribute Importance": shap_importance.values()} )

            adf = adf.sort_values(by=['Attribute Importance'] , ascending=False)


            #print(shap_importance)
            sonuclar2 = html.Div([dash_table.DataTable(
                data=adf.to_dict('records'),
                editable=False,
                        
                        page_action="native",
                        page_current= 0,
                        page_size= 10, 
                        style_table={
            'width': '100%',
                 'height': '600px',
            'overflowY': 'scroll',
            'overflowX': 'scroll',
            'textAlign': 'center',

        },style_header=
                                {
                                    'fontWeight': 'bold',
                                    'border': 'thin lightgrey solid',
                                    'backgroundColor': 'rgb(100, 100, 100)',
                                    'color': 'white',
                                    'textAlign': 'center',
                                },style_cell={
                                    "font-family": "Bahnschrift",
                                    'textAlign': 'center',
                                    'width': '150px',
                                    'minWidth': '180px',
                                    'maxWidth': '180px',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',

                                },
            )])
            fig=pl.gcf()
            if tur=="Shap Summary Plot":

                 shap.summary_plot(shap_values, x_test, class_names=new_hedef_df.unique(), color='coolwarm', feature_names = new_oz_onehot.columns, show=(False))
            elif tur=="Attribute Importance Plot":

                feat_importances = pd.Series(shap_importance.values(), index=x_test.columns)
                ax=feat_importances.nlargest(20).plot(kind='barh',figsize=(7, 6),color='#86bf91', zorder=2, width=0.3)
                ax.set_xlabel("Normalized shap significance values", labelpad=20, weight='bold', size=12)
                ax.set_ylabel("Attribute", labelpad=20, weight='bold', size=12)
            else:

                feat_importances = pd.Series(shap_importance.values(), index=x_test.columns)
                feat_importances.nlargest(20).plot(kind='barh',figsize=(7, 6),color='#86bf91', zorder=2, width=0.3)

            #adf.plot.bar(x='feature_importance_vals', y='col_name', rot=0)
            tmpfile = BytesIO()
            fig.savefig(tmpfile, format='png', bbox_inches='tight')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            html1 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
            figur = html.Iframe(srcDoc=html1,
            style={"width": "130%", "height": "700px",
            "border": 0})
            plt.clf()

        else:
            print("kernel tabanlı")
            explainer = shap.KernelExplainer(classifier.predict_proba, x_test)
            shap_values = explainer.shap_values(x_test)
            print(np.array(shap_values).shape)
            shap_values2= shap_values[1][:]
            shap_values2= pd.DataFrame(shap_values2)
            vals = np.abs(shap_values2.values).sum(0)
            vals=vals/vals.sum()

            a = new_oz_onehot.columns

            shap_importance = {}
            for i in range(len(a)):
                shap_importance[a[i]] = vals[i]

            adf = pd.DataFrame( {"Attribute": shap_importance.keys(), "Attribute Importance": shap_importance.values()} )

            adf = adf.sort_values(by=['Attribute Importance'] , ascending=False)


            #print(shap_importance)
            sonuclar2 = html.Div([dash_table.DataTable(
                data=adf.to_dict('records'),
               editable=False,
                        
                        page_action="native",
                        page_current= 0,
                        page_size= 10, 
                        style_table={
            'width': '100%',
                 'height': '600px',
            'overflowY': 'scroll',
            'overflowX': 'scroll',
            'textAlign': 'center',

        },style_header=
                                {
                                    'fontWeight': 'bold',
                                    'border': 'thin lightgrey solid',
                                    'backgroundColor': 'rgb(100, 100, 100)',
                                    'color': 'white',
                                    'textAlign': 'center',
                                },style_cell={
                                    "font-family": "Bahnschrift",
                                    'textAlign': 'center',
                                    'width': '150px',
                                    'minWidth': '180px',
                                    'maxWidth': '180px',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',

                                },
            )])
            fig=pl.gcf()
            if tur=="Shap Summary Plot":

                shap.summary_plot(shap_values[1], x_test, class_names=new_hedef_df.unique(), color='coolwarm', feature_names = new_oz_onehot.columns, show=(False))
            elif tur=="Attribute Importance Plot":

                feat_importances = pd.Series(shap_importance.values(), index=x_test.columns)
                ax=feat_importances.nlargest(20).plot(kind='barh',figsize=(7, 6),color='#86bf91', zorder=2, width=0.3)
                ax.set_xlabel("Normalized shap significance values", labelpad=20, weight='bold', size=12)
                ax.set_ylabel("Attribute", labelpad=20, weight='bold', size=12)
            else:

                feat_importances = pd.Series(shap_importance.values(), index=x_test.columns)
                feat_importances.nlargest(20).plot(kind='barh',figsize=(7, 6))
                ax.set_xlabel("Normalized shap significance values", labelpad=20, weight='bold', size=12)
                ax.set_ylabel("Attribute ", labelpad=20, weight='bold', size=12)
            #adf.plot.bar(x='feature_importance_vals', y='col_name', rot=0)
            tmpfile = BytesIO()
            fig.savefig(tmpfile, format='png', bbox_inches='tight')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            html1 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
            figur = html.Iframe(srcDoc=html1,
            style={"width": "130%", "height": "700px",
            "border": 0})
            plt.clf()

        return sonuclar2,figur



    except Exception as e:
        print(e)
        return "SHAP could not be calculated"
