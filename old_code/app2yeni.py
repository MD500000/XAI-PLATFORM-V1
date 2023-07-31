from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
from app import app
from apps import function_1
import pandas as pd
import dash_table
import numpy as np
import lime
import lime.lime_tabular
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sklearn
import matplotlib
import fontawesome as fa
#####################
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

eniyimodel=None

################################
# tasarım stilleri





SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 47,
    "left": 0,
    "right": 0,
    "bottom": 0,
    "width": "20rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#37474F",
	"line-height": 15.82857143,
	#"padding": "4rem 1rem 2rem",
	#"font-family": "Bahnschrift"
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
    "background-color": "#757575",
    #"font-family": "Bahnschrift"
}
CONTENT_STYLE = {
    "margin-left": "20rem",
    "margin-right": "20rem",
    "padding": "2rem 1rem",
	"bottom": 0,
	"font-size": "16px",

	"width": "100rem",
	"position": "center",
    #"font-family": "Bahnschrift"

}
CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    #"font-family": "Bahnschrift"

}
nav_style={
    "color": "#ffffff",
    "link": "#ffffff",
    "background-color": "#37474F",
	"width": "20rem",
	#"position": "fixed",
	"padding": "0.5rem 1rem",
	"left": 0,
	"line-height": 15.82857143,
	"font-size": "15px",
	#'font-weight': 'bold'
	#"font-family": "Arial"
    #"font-family": "Bahnschrift"

}

###############################
df = pd.DataFrame()


tabs_styles = {
    'height': '44px',
    "background-color": "#FF8F00",
    'align-items': 'center'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',

    'padding': '6px',
    'fontWeight': 'bold',
    'border-radius': '15px',
    "background-color": "#FF8F00",
    'box-shadow': '4px 4px 4px 4px lightgrey',
    "font-family": "Bahnschrift"

}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    "background-color": "#20B2AA",
    'color': 'white',
    'padding': '6px',
    'border-radius': '15px',
    "font-family": "Bahnschrift"
}

backdrop_selector = html.Div(
    [
        dbc.Label("Backdrop:"),
        dbc.RadioItems(
            id="offcanvas-backdrop-selector",
            options=[
                {"label": "True (default)", "value": True},
                {"label": "False", "value": False},
                {"label": "Static (no dismiss)", "value": "static"},
            ],
            inline=True,
            value=True,
        ),
    ],
    className="mb-2",
)
import dash_bootstrap_components as dbc
offcanvas = html.Div(
    [
        backdrop_selector,
        dbc.Button(
            "Open backdrop offcanvas", id="open-offcanvas-backdrop", n_clicks=0
        ),
        dbc.Offcanvas(
            html.P("Here is some content..."),
            id="offcanvas-backdrop",
            title="Offcanvas with/without backdrop",
            is_open=False,
        ),
    ]
)


items = html.Div(
    [dcc.RadioItems(
    options=[
        {'label': 'New York City', 'value': 'NYC'},
        {'label': 'Montréal', 'value': 'MTL'},
        {'label': 'San Francisco', 'value': 'SF'}
    ],
    value='MTL'
)
])
Tab_Ekle = html.Div([
    #html.H1('Adımları Sırasıyla Takip Ediniz'),
    dcc.Tabs(id="tabs-example-graph", value='tab-1',
children=[
        dcc.Tab(label='Veri Yükleyiniz', value='tab-1',style = tab_style, selected_style = tab_selected_style),
        dcc.Tab(label='Özellikleri Belirleyiniz', value='tab-2',style = tab_style, selected_style = tab_selected_style),
        #dcc.Tab(label='Veri Önişlem Özelliklerini belirleyiniz', value='tab-3'),
        dcc.Tab(label='Sınıflandırma', value='tab-4',style = tab_style, selected_style = tab_selected_style),
        dcc.Tab(label='LIME', value='tab-5',style = tab_style, selected_style = tab_selected_style),
        dcc.Tab(label='SHAP', value='tab-6',style = tab_style, selected_style = tab_selected_style),
    ]),
    html.Div(id='tabs-content-example-graph')
])

@app.callback(
    Output("offcanvas-backdrop", "backdrop"),
    [Input("offcanvas-backdrop-selector", "value")],
)
def select_backdrop(backdrop):
    return backdrop


@app.callback(
    Output("offcanvas-backdrop", "is_open"),
    Input("open-offcanvas-backdrop", "n_clicks"),
    State("offcanvas-backdrop", "is_open"),
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open
navbar = dbc.Navbar(
    [
    dbc.Row(  [  dbc.Col( html.Span(className="fas fa-align-justify",id="btn_sidebar1",style={"font-weight": "bold", "width": "1.5rem"}),   ), ],
                align="center",  className="ml-auto flex-nowrap mt-12 mt-md-0"),
    html.Span("xAI-TR",style=dict(fontSize = 20,  width='92%',height= "20px", verticalAlign="middle", color="white"  )),


    dbc.Button("Türkçe", href='/apps/app1',color="warning", size="sm",active=True,className="mr-1",id='btn-nclicks-12',style={ 'color':'white' , 'border':'1.5px black solid','text-align':'center'}),
    dbc.Button("English", href='/apps/app2',color="warning", size="sm",active=True,className="mr-1",id='btn-nclicks-22',style={ 'color':'white' , 'border':'1.5px black solid','text-align':'center'}),
    ],
    color="#00897B",
    dark=True,
    
)

sidebar = html.Div(
    [

        dbc.Nav(
            [

                dbc.NavLink("      Giriş", href="/Giris", id="page-1-link1",active="exact",style=nav_style,className="fas fa-home"),
				html.Br(),
                dbc.NavLink("      Analiz ", href="/Yazilim", id="page-2-link1",active="exact",style=nav_style,className="fas fa-cogs"),
				html.Br(),
				dbc.NavLink("      Alıntılama", href="/Alinti", id="page-3-link1",active="exact",style=nav_style,className="fas fa-quote-right"),
				html.Br(),

            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar1",
    style=SIDEBAR_STYLE,
)

@app.callback(
    [
        Output("sidebar1", "style"),
        Output("page-content1", "style"),
        Output("side_click1", "data"),
    ],

    [Input("btn_sidebar1", "n_clicks")],
    [
        State("side_click1", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

data_upload = [
    html.Div(
    [ html.Br(),html.Br(),html.Br(), html.H5("Bu yazılım, .xls/.xlsx, .sav ve .csv/.txt uzantılı veri dosyalarını desteklemektedir.", className="card-title"), html.Br(), dcc.Upload(
            id="upload-data",


            children=html.Div([dbc.Button("Seçiniz ", color="warning", size="sm",active=True,className="mr-1",style={ 'color':'white' ,"background-color": "#FF8F00", "height": "40px",'border':'1.5px black solid','text-align':'center',"font-family": "Bahnschrift"})]),
            # style={
                # "width": "15%",
                # "height": "20px",
                # "lineHeight": "60px",
                # "borderWidth": "1px",
                # "borderStyle": "dashed",
                # "borderRadius": "5px",
                # "textAlign": "center",
                # "margin": "10px",
                # "background-color": "#FF8F00",
                # 'border-radius': '15px',
            # },
            # Allow multiple files to be uploaded
            multiple=False,
        ),

        html.Div(id="output-data-upload"),
    ]
)  ]




giris_veri = [
            html.H5("Açıklanabilir Yapay Zeka (xAI)", className="card-title"),
            html.P(
                'Makine öğrenimi yöntemleri ile modelleme sonucu elde edilen sonuçların daha yorumlanabilir ve açıklanabilir olması amacıyla bazı yöntemlere ihtiyaç duyulmuştur. Bu gereksinimlere dayalı olarak açıklanabilir yapay zeka kavramı ortaya atılmıştır. Açıklanabilir yapay zeka modelleme sonucu elde edilen sonuçların daha anlaşılabilir ve açıklanabilir olması amacıyla her bir gözlem için çıktı ve girdi değişkenleri arasındaki ilişkileri ortaya koyarak modelin daha anlaşılabilir olması için geliştirilen yöntemler bütünüdür. Sağlık alanında hastalığı teşhis etmek için sınıflandırma modellerinin kullanımı, büyük ölçüde oluşturulan modellerin araştırıcı tarafından yorumlanabilmesine ve açıklanabilmesine bağlıdır. Sağlık alanında oluşturulan yapay zeka modellerinin açıklanabilirliğini artırmanın birçok farklı yolu vardır ve değişken önemliliği bunlardan biridir. Bu amaçla kullanılan açıklanabilir yapay zeka yöntemleri, belirli bir sınıflandırma için hastaya özel bir açıklama sağlar, böylece herhangi bir karmaşık sınıflandırıcının klinik ortamda daha basit bir şekilde açıklanmasına olanak tanır.  ',
                style={'text-align':'justify','fontSize': 17,'font-family':'Bahnschrift','font-style': 'normal'}, className="bootstrap.min",
            ),
            html.H5("Yerel Olarak Yorumlanabilir Model Agnostik Açıklamalar (LIME)", className="card-title"),
            html.P(
                "   LIME, her bir bireysel tahmini açıklamak için herhangi bir kara kutu makine öğrenimi modeline yerel, yorumlanabilir bir modelle yaklaşmayı amaçlayan bir post-hoc modelden bağımsız açıklama tekniğidir.  Yazarlar, LIME orijinal sınıflandırıcıdan bağımsız olduğundan, tahminler için kullanılan algoritmadan bağımsız olarak modelin herhangi bir sınıflandırıcıyı açıklamak için kullanılabileceğini öne sürüyorlar. Sonuç olarak, LIME yerel olarak çalışır, bu da gözleme özel olduğu anlamına gelir ve her gözlemle ilgili tahmin için açıklamalar sağlar. LIME'ın yaptığı, açıklanan gözleme benzer örnek veri noktalarını kullanarak yerel bir modele uymaya çalışmaktır.  ",
                style={'text-align':'justify','fontSize': 17,'font-family':'Bahnschrift','font-style': 'normal'}, className="bootstrap.min",
            ),
            html.H5("Shapley Katkı Açıklamaları (SHAP)", className="card-title"),
            html.P(
                "  SHAP'ın ana fikri, yorumlanacak örneğin her bir özelliği için Shapley değerlerini hesaplamaktır; burada her Shapley değeri, ilişkilendirildiği özelliğin tahminde yarattığı etkiyi temsil eder.     ",
                style={'text-align':'justify','fontSize': 17,'font-family':'Bahnschrift','font-style': 'normal'}, className="bootstrap.min",
            ),
]
alinti_veri = [
    html.Div([ offcanvas, html.Br(), html.Br(),html.Br(),html.Br(),items]) ]

#fonksiyon deneme





content = html.Div(

    id="page-content1",

    style=CONTENT_STYLE)

snfsec = html.Div([

        dcc.Dropdown(
        id="snfdropdown",
    options=[
                {'label': 'svm', 'value': 'SVM'},
                {'label': 'Logistic Regression', 'value': 'LR'},
                {'label': 'Random Forest', 'value': 'RF'},
                {'label': 'Desicion Tree', 'value': 'DT'},
                {'label': 'Lightgbm', 'value': 'LGBM'},
                {'label': 'Gaussian Naive Bayes', 'value': 'GNB'},
                {'label': 'AdaBoost', 'value': 'ADA'},
                {'label': 'XgBoost', 'value': 'XGB'}

            ],
    value=['SVM', 'LR'],
    multi=True
    )
])


layout = html.Div([
    dcc.Store(id='side_click1'),
    dcc.Location(id="urla"),
    #dcc.Store(id='svmmodel', storage_type='session'),
    navbar,
    sidebar,
    content,
    html.Div(id='verideposu', style={'display': 'none'}),
    html.Div(id='ozellikdf', style={'display': 'none'}),
    html.Div(id='hedefdf', style={'display': 'none'}),
    html.Div(id='eniyimodelismi', style={'display': 'none'}),
])

@app.callback(Output("page-content1", "children"), [Input("urla", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/Giris"]:
        return giris_veri
    elif pathname == "/Yazilim":
        return Tab_Ekle
    elif pathname == "/Alinti":
        return alinti_veri
    else:
        return giris_veri



@app.callback(Output('verideposu', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),  )


def updateupload(contents, filename):

    if contents is not None:
        jsonfile = function_1.dosyaoku(contents, filename)
        return jsonfile



@app.callback(  Output('output-data-upload', 'children'),
                Input('verideposu', 'children') )

def alanupdate(veri):
    try:
        return function_1.veriyazdir(veri)
    except:

        return html.Div([html.Hr(),
                         html.H4('Dosya Girişi yapılmadı.'),])





@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'),
              State('verideposu', 'children'),
              State('ozellikdf', 'children'),
              State('hedefdf', 'children'),
              #State('eniyimodelismi', "children"),
              )

def render_content(tab,veri,ozellik,hedef):
    global eniyimodel
    modelisim=eniyimodel

    if tab == 'tab-1':
        return data_upload

    elif tab == 'tab-2':
        if veri is not None:
            dff=pd.read_json(veri , orient='split')
            return html.Div([  html.Hr(),
                         html.H5("Öznitelik Alanlarını belirleyiniz:"),
                         html.Div([ dash_table.DataTable(style_table={
                'maxHeight': '50ex',
                'overflowY': 'scroll',
                'width': '100%',
                'minWidth': '100%',
                'overflowX': 'scroll',
                "font-family": "Bahnschrift"
            },
            # style cell
            style_cell={
                "font-family": "Bahnschrift",
                'textAlign': 'center',
                'height': '20px',
                'padding': '2px 22px',
                'whiteSpace': 'inherit',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'State'},
                    'textAlign': 'left',
                    'overflow': 'hidden',
                },
            ],
            # style header
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': 'white',
            },
            style_data_conditional=[
                {
                    # stripped rows
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
                ],
                             id='özniteliktablo',
                             columns=[ {"name": i, "id": i, "deletable": False, "selectable": True} for i in dff.columns ],
                             data=dff.to_dict('records'),
                             editable=False,
                             #filter_action="native",
                             column_selectable="multi",
                             selected_columns=[i for i in dff.columns],
                             page_action="native",
                             page_current= 0,
                             page_size= 3,  ), html.Div(id='öznitelikalan') ]),
                         html.Br(),
                         html.Br(),
                         html.Br(),
                         html.Hr(),
                         html.H5("Hedef Alanı belirleyiniz:"),
                         html.Div([ dash_table.DataTable(style_table={
                'maxHeight': '50ex',
                'overflowY': 'scroll',
                'width': '100%',
                'minWidth': '100%',
                'overflowX': 'scroll',
                "font-family": "Bahnschrift"
            },
            # style cell
            style_cell={
                "font-family": "Bahnschrift",
                'textAlign': 'center',
                'height': '20px',
                'padding': '2px 22px',
                'whiteSpace': 'inherit',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'State'},
                    'textAlign': 'left',
                    'overflow': 'hidden',
                },
            ],
            # style header
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': 'white',
            },style_data_conditional=[
                {
                    # stripped rows
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
                ],
                             id='hedeftablo',
                             columns=[ {"name": i, "id": i, "deletable": False, "selectable": True} for i in dff.columns ],
                             data=dff.to_dict('records'),
                             editable=False,
                             #filter_action="native",
                             column_selectable="single",
                             selected_columns=[],
                             page_action="native",
                             page_current= 0,
                             page_size= 3,  ), html.Div(id='hedefalan') ]),

                         html.Br(),
                         html.Hr(),
                         dbc.Button('Kaydet', id='verikaydet',color="primary", className="mr-1",  n_clicks=0),
                         html.Div(id='ozellikhedefkaydet')

                         ])


    elif tab == 'tab-3':
        try:

            if ozellik is not None and hedef is not None:
                o_df=pd.read_json(ozellik , orient='split')
                h_df=pd.read_json(hedef , orient='split')
                return html.Div([  html.Hr(),
                             html.H5("bu bölümde ön işleme adımlarını ekleriz normalizasyon, kayıp veriler vs..:"),
                             html.Hr(),
                             html.H5("Öznitelik Alanları:"),
                             html.Div([ dash_table.DataTable(
                                 id='özniteliktablo',
                                 columns=[ {"name": i, "id": i} for i in o_df.columns ],
                                 data=o_df.to_dict('records'),
                                 editable=False,
                                 page_action="native",
                                 page_current= 0,
                                 page_size= 1,  ), html.Div(id='öznitelikalan') ]),
                             html.Br(),
                             html.Br(),
                             html.Br(),
                             html.Hr(),
                             html.H5("Hedef Alanları:"),
                             html.Div([ dash_table.DataTable(
                                 id='hedeftablo',
                                 columns=[ {"name": i, "id": i} for i in h_df.columns ],
                                 data=h_df.to_dict('records'),
                                 editable=False,
                                 page_action="native",
                                 page_current= 0,
                                 page_size= 1,  ), html.Div(id='hedefalan') ]),



                             ])

            else:
                return html.Div()

        except:
            return ""


    elif tab == 'tab-4':
        try:
            if ozellik is not None and hedef is not None:

                return html.Div([
                    html.Br(),
                    snfsec,
                    html.Br(),
                    dbc.Button('Hesapla', id='hesapla',color="primary", n_clicks=0, ),
                    html.Hr(),
                    html.Div(id='sonuc_1'),
                    ])

        except:
            return "olmadı"

    elif tab =='tab-5':

        if ozellik is not None and hedef is not None and modelisim is not None:


            o_df=pd.read_json(ozellik , orient='split')

            return html.Div([
                        html.Hr(),
                        dcc.Dropdown( id='limemodelsecici',
                        options=[   {'label': 'svm', 'value': 'SVM'},
                                    {'label': 'Logistic Regression', 'value': 'LR'},
                                    {'label': 'Random Forest', 'value': 'RF'},
                                    {'label': 'Desicion Tree', 'value': 'DT'},
                                    {'label': 'Lightgbm', 'value': 'LGBM'},
                                    {'label': 'Gaussian Naive Bayes', 'value': 'GNB'},
                                    {'label': 'AdaBoost', 'value': 'ADA'},
                                    {'label': 'XgBoost', 'value': 'XGB'}],
                                    value='SVM',     clearable=False ),
                                          html.Hr(),
                        html.Hr(),
                        #html.H5('Modeliniz: ' + modelisim + '  kayıt seçiniz-->'),
                        html.Hr(),
                        dash_table.DataTable( id='limetablo',
                                                 columns=[ {"name": i, "id": i} for i in o_df.columns ],
                                                 data=o_df.to_dict('records'),
                                                 editable=False,
                                                 row_selectable="single",
                                                 selected_rows=[0],
                                                 page_action="native",
                                                 page_current= 0,
                                                 page_size= 10,  ), html.Div(id='öznitelikalan'),
                        html.Br(),
                        html.Hr(),

                        html.Div(id='limegoster'),
                        dcc.Loading(id='explainer-obj', type="default" ),

            ])

    elif tab=='tab-6':
        if ozellik is not None and hedef is not None and modelisim is not None:
            return html.Div( [
                    html.Hr(),
                  dcc.Dropdown( id='shapmodelsecici',
                                   options=[   {'label': 'svm', 'value': 'SVM'},
                                               {'label': 'Logistic Regression', 'value': 'LR'},
                                               {'label': 'Random Forest', 'value': 'RF'},
                                               {'label': 'Desicion Tree', 'value': 'DT'},
                                               {'label': 'Lightgbm', 'value': 'LGBM'},
                                               {'label': 'Gaussian Naive Bayes', 'value': 'GNB'},
                                               {'label': 'AdaBoost', 'value': 'ADA'},
                                               {'label': 'XgBoost', 'value': 'XGB'}],

                                   value='SVM',     clearable=False ),
                  html.Hr(),
                  dcc.Loading(id='shapgoster' ),
                  dcc.Store(   id='figure_store'),
                  html.Div(id='shapgoster1')

                ])


    else:
        return(html.Div("Adımları Sırasıyla Takip Ediniz"))


    # else:
    #     return data_upload

####################################################################################

# tüm sınıflandırma sonuçları
@app.callback(
    Output('sonuc_1', 'children'),
    #Output('eniyimodelismi', 'children'),
    Input('hesapla', 'n_clicks'),
    State('ozellikdf', 'children'),
    State('hedefdf', 'children'),
    State('snfdropdown','value') )
def sonuchesapla( btn, ozellik, hedef, modellist):
    try:
        if modellist == []:
            return html.Div('En az 1 model seçmelisiniz')
        if ozellik is not None and hedef is not None and btn!=0:
            X=pd.read_json(ozellik, orient='split')
            X=function_1.kategorikveriler(X)
            X=function_1.kayipveriler(X)

            Y=pd.read_json(hedef, orient='split')
            try:
                Y=Y.to_frame()
            except:
                Y=Y
            Y=function_1.kategorikveriler(Y)
            Y=function_1.kayipveriler(Y)
            Y = np.ravel(Y)
            normalizasyon = StandardScaler()
            X = normalizasyon.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
            print("veriyi böldük")
            model = []
            acc = []
            for i in modellist:
                score= function_1.siniflandirma_1(X_train, y_train, X_test, y_test, i)
                acc.append(score)
                model.append(i)
            data = {'model': model, 'accuracy':acc}
            df = pd.DataFrame.from_dict(data)
            max = acc[0]
            max_isim = model[0]
            for i in range(len(acc)):
                if max<acc[i]:
                    max=acc[i]
                    max_isim=model[i]
            global eniyimodel
            eniyimodel=max_isim
            return dash_table.DataTable(
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),  )
    except:
        return "Bir Aksilik Oluştu Tekrar Deneyiniz"


#####################################################################################
# özellik ve hedeflerin kayıt edilmesi
@app.callback(Output('ozellikdf', 'children'),
              Output('hedefdf', 'children'),
              Input('verikaydet', 'n_clicks'),
              State('verideposu', 'children'),
              State('özniteliktablo', "derived_viewport_selected_columns"),
              State('hedeftablo', "derived_viewport_selected_columns"),

    )
def verisetiolustur1(btn1, veri, kaynaklar, hedef):
    try:
        if kaynaklar is not None and veri is not None:
            dff=pd.read_json(veri , orient='split', encoding='utf-8')
            kaynakalanlari =dff[kaynaklar]
            hedefalanlari =dff[hedef]
            o_df = kaynakalanlari.to_json(date_format='iso', orient='split' )
            h_df = hedefalanlari.to_json(date_format='iso', orient='split' )
            return o_df, h_df
    except:
        return "",""

@app.callback(Output('ozellikhedefkaydet', 'children'),
              Input('verikaydet', 'n_clicks'),
              State('hedefdf', 'children'),
              State('ozellikdf', 'children') )
def ozellikkayitupdate( click, hedef, kaynak):
    try:
        if hedef is not None and kaynak is not None:
            return html.Br(), html.H6('Özellik ve Hedef alanları seçildi. Eğitim Adımına gidebilirsiniz.')
    except: return ('')
#####################################################################################
#shap

@app.callback(
    Output('shapgoster', 'children'),
    Input('shapmodelsecici', 'value'),
    State('ozellikdf', 'children'),
    State('hedefdf', 'children'))
def shapfonksiyon(model_ismi, ozellik, hedef ):

    try:
        if ozellik is not None and hedef is not None:
            A=pd.read_json(ozellik, orient='split')
            B=pd.read_json(hedef, orient='split')


        fig= function_1.shap_fonksiyon(A,B,model_ismi)


        return html.Div([fig,html.H5("Grafikteki noktalar, hastanın değişken seviyelerinin normalize edilmiş değerlerine göre renklendirildi. Maviye yaklaştıkça ilgili değişken değeri azalırken, pembeye yaklaştıkça artmaktadır. Bir özelliğin SHAP değeri ne kadar yüksek verilirse, pozitif sınıfın olasılığı o kadar yüksek olur." )])

    except:
        return "hesaplanamadı"










######################################################################################
#lime sonuç göster
model = None
limemodelisim=None
@app.callback(Output('explainer-obj', 'children'),
              Input('limetablo',"derived_virtual_selected_rows"),
              Input('limemodelsecici', "value"),
              #State('eniyimodelismi', "children"),
              State('ozellikdf', 'children'),
              State('hedefdf', 'children') )
def limefonksiyon(i,model_isim, ozellik, hedef):
    #global eniyimodel
    #model_isim=eniyimodel

    ozellikveri=pd.read_json(ozellik, orient='split')
    hedefveri=pd.read_json(hedef, orient='split')


    try:
        A=hedefveri.to_frame()
    except:
        A=hedefveri
    #print(A)
    A= A.squeeze()
    #print(A)
    global model, X, Y, limemodelisim

    if model is None or model_isim !=limemodelisim:
        limemodelisim=model_isim
        model, X, Y=function_1.siniflandirma(ozellikveri, hedefveri, model_isim)

    explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=list(ozellikveri.columns), class_names=A.unique(), discretize_continuous=True)
    print('lime çalıştı')
    exp = explainer.explain_instance(X[i[0]], model.predict_proba, num_features=5, top_labels=1)
    print("lime hesapladı")

    exp.as_html()

    obj = html.Iframe(
            # Javascript is disabled from running in an IFrame for security reasons
            # Static HTML only!!!
        srcDoc=exp.as_html(),
            width='100%',
            height='200px',
            style={'border': '2px #d3d3d3 solid'},
        )



    return obj
