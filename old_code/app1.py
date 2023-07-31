import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import dash_table
from dash import html
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
giris_metni="Makine öğrenimi yöntemleri ile modelleme sonucu elde edilen sonuçların daha yorumlanabilir ve açıklanabilir olması amacıyla bazı yöntemlere ihtiyaç duyulmuştur. Bu gereksinimlere dayalı olarak açıklanabilir yapay zeka kavramı ortaya atılmıştır. Açıklanabilir yapay zeka modelleme sonucu elde edilen sonuçların daha anlaşılabilir ve açıklanabilir olması amacıyla her bir gözlem için çıktı ve girdi değişkenleri arasındaki ilişkileri ortaya koyarak modelin daha anlaşılabilir olması için geliştirilen yöntemler bütünüdür. Sağlık alanında hastalığı teşhis etmek için sınıflandırma modellerinin kullanımı, büyük ölçüde oluşturulan modellerin araştırıcı tarafından yorumlanabilmesine ve açıklanabilmesine bağlıdır. Sağlık alanında oluşturulan yapay zeka modellerinin açıklanabilirliğini artırmanın birçok farklı yolu vardır ve değişken önemliliği bunlardan biridir. Bu amaçla kullanılan açıklanabilir yapay zeka yöntemleri, belirli bir sınıflandırma için hastaya özel bir açıklama sağlar, böylece herhangi bir karmaşık sınıflandırıcının klinik ortamda daha basit bir şekilde açıklanmasına olanak tanır."
lime_metni = "LIME, her bir bireysel tahmini açıklamak için herhangi bir kara kutu makine öğrenimi modeline yerel, yorumlanabilir bir modelle yaklaşmayı amaçlayan bir post-hoc modelden bağımsız açıklama tekniğidir. Yazarlar, LIME orijinal sınıflandırıcıdan bağımsız olduğundan, tahminler için kullanılan algoritmadan bağımsız olarak modelin herhangi bir sınıflandırıcıyı açıklamak için kullanılabileceğini öne sürüyorlar. Sonuç olarak, LIME yerel olarak çalışır, bu da gözleme özel olduğu anlamına gelir ve her gözlemle ilgili tahmin için açıklamalar sağlar. LIME'ın yaptığı, açıklanan gözleme benzer örnek veri noktalarını kullanarak yerel bir modele uymaya çalışmaktır."
shap_metni = "SHAP'ın ana fikri, yorumlanacak örneğin her bir özelliği için Shapley değerlerini hesaplamaktır; burada her Shapley değeri, ilişkilendirildiği özelliğin tahminde yarattığı etkiyi temsil eder."
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

giris_card = dbc.Card(
    [
        dbc.CardImg(src="/assets/classification.jpg", top=True),
        dbc.CardBody(
            [
                html.H4("Açıklanabilir Yapay Zeka (xAI) ", className="card-title",style={"width": "100%", 'text-align':'justify',"font-family": "Bahnschrift"},),
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
                html.H4("Yerel Olarak Yorumlanabilir Model Agnostik Açıklamalar (LIME) ", className="card-title",style={"width": "100%", 'text-align':'justify',"font-family": "Bahnschrift"},),
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
                html.H4("Shapley Katkı Açıklamaları (SHAP) ", className="card-title",style={"width": "100%", 'text-align':'justify',"font-family": "Bahnschrift"},),
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
                  ], xs=12, sm=12, md=12, lg=3, xl=3),dbc.Button("Başlayın ->", color="primary", href="/Veri_Yukle" , n_clicks=0)])


                ])


data_upload2 =dbc.Card(
    [
        dbc.CardBody(
            [
                html.H6("Bu yazılım, .xls,.xlsx, .sav, .csv ve.txt uzantılı veri dosyalarını desteklemektedir.", className="card-title"),
                dcc.Upload( id="upload-data",  children=html.Div([dbc.Button("Seçiniz ", color="primary", active=True, className="mb-3")]),
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
                html.H6("Ön işleme adımlarından önce veriyi analiz etmeniz gerekmektedir.", className="card-title"),
                html.Br(),
                dbc.Button("Analiz Et ", id = "analizbuton" ,color="primary", n_clicks=0),
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
                                    [html.H6("Model Seçiniz:",style={"font-family": "Bahnschrift"},),
                                dcc.Checklist(
    options=[
                {'label': 'Destek Vektör Makinesi', 'value': 'SVM'},
                {'label': 'Lojistik Regresyon', 'value': 'LR'},
                {'label': 'Rastgele Orman', 'value': 'RF'},
                {'label': 'Karar Ağacı', 'value': 'DT'},
                {'label': 'Lightgbm', 'value': 'LGBM'},
                {'label': 'Gaussian Naive Bayes', 'value': 'GNB'},
                {'label': 'AdaBoost', 'value': 'ADA'},
                {'label': 'GradientBoosting', 'value': 'GBT'},
                {'label': 'CatBoost', 'value': 'CB'},
                {'label': 'XgBoost', 'value': 'XGB'},
                {'label': 'Çok Katmanlı Algılayıcı (MLP) ', 'value': 'MLP'}

            ],
    value=['SVM', 'LR'],id="modelsecici",
    labelStyle = {'display': 'block'}, className='radiobutton-group',
    style={"line-height":"28px","font-family": "Bahnschrift"},
)
],className="row flex-display",
                                ),html.Br(),

                                html.Div(
    [html.H6("Parametre optimizasyonu yapılsınmı:",style={"font-family": "Bahnschrift"}),
        dcc.RadioItems(
            options=[
                {"label": "Evet", "value": '0'},
                {"label": "Hayır", "value": '1'},

            ],id="optimizasyonslider",
            value="1",
            inputStyle={"vertical-align":"middle", "margin":"auto"},
            labelStyle={"vertical-align":"middle"},
            style={"display":"inline-flex", "flex-wrap":"wrap", "justify-content":"space-between","line-height":"28px"},
            #labelStyle = {'display': 'block'},className='radiobutton-group',
            #style={"line-height":"28px","font-family": "Bahnschrift"},


        ),
        html.Div(id="optimizasyonsecenek"),
            html.H6("Doğrulama yöntemi:",style={"font-family": "Bahnschrift"}),

            dcc.RadioItems(  options=[
            {'label': 'Hold-out', 'value': '0'},
            {'label': 'Repeated Hold-out', 'value': '1'}, {'label': 'Stratified K Fold Cross Validation', 'value': '2'},{'label': 'Leave one out Cross Validation', 'value': '3'},{'label': 'Repeated Cross Validation', 'value': '4'} , {'label': 'Nested Cross Validation', 'value': '5'} ],id="veriseti-bolum-secici",
            value='0',   labelStyle = {'display': 'block'},className='radiobutton-group',style={"line-height":"25px","font-family": "Bahnschrift"}, ),
                html.Br(),
                html.Div(id='bolme'),


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
    id="offcanvas",
    is_open=False,
    title="",
    style=SIDEBAR_STYLE
    )]
)

@app.callback(
    Output("bolme", "children"),

    Input("veriseti-bolum-secici", "value"),
    #State("veriseti-bolum-secici", "value" ),

    )
def validation(split):

    bolumleme=html.Div([html.H6("Eğitim veri seti yüzdesini seçiniz:", className="control_label",style={"font-family": "Bahnschrift"}),
                      dcc.Slider(50, 100, 5, value=80, id='traintestslider', marks=None,
                      tooltip={"placement": "bottom", "always_visible": True} ),
                      html.Div(id="kfoldsecenegigoster"),
                      ],className="row flex-display," )

    cv=html.Div(      [      html.H6("k kat seçiniz:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(2, 10, 1, value=2, id='kfoldslider', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} ) ])

    nsplit= html.Div(      [      html.H6("Tekrar sayısını seçiniz:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 50, 1, value=5, id='ntekrarslider', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} ),html.Br(), html.H6("Select split size:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(0, 100, 5, value=50, id='ntekrarsplitslider', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} )        ] )
    repeatcv= html.Div(      [      html.H6(" k kat seçiniz:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 10, 1, value=5, id='repatkfoldslider', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} ),html.Br(), html.H6("Tekrar sayısını seçiniz:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 10, 1, value=50, id='ntkrarslider', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} )] )
    nestedcv= html.Div(      [      html.H6("İç k kat seçiniz:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 10, 1, value=5, id='nestedickfoldslider', marks=None,
                         tooltip={"placement": "bottom", "always_visible": True} ),html.Br(), html.H6("Dış k kat seçiniz:",style={"font-family": "Bahnschrift"}, className="control_label"),
                    dcc.Slider(5, 10, 1, value=50, id='diskfoldslider', marks=None,
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
                dbc.Col(html.Div(id="output-data-upload-2"),  xs=9, sm=9, md=9, lg=9, xl=9, width={'offset': 0})
])  ])


veri_onisleme = dbc.Container([

            dbc.Row([
                dbc.Col(html.Br(), width=12),
                dbc.Col(veri_analiz2, xs=3, sm=3, md=3, lg=3, xl=3),
                dbc.Col(html.Div(id="output-veri-analiz"),  xs=9, sm=9, md=9, lg=9, xl=9 ),
                dbc.Col(html.Div(id="output-veri-analiz2"),  xs=9, sm=9, md=9, lg=9, xl=9 )
])  ])

################################################################################


parametre1 = dbc.Toast([
        offcanvas,
         dbc.Button("<<<<", id="open-offcanvas", n_clicks=1, color="danger"),

], header="Modelleme yöntemlerini seçiniz", className="mb-3", style={'width': '100%'})


parametre2 = dbc.Toast([
            html.Div(["Bu işlem seçtiğiniz model sayısına göre uzun sürebilir. Lütfen bekleyiniz."], className="h6"),
         dbc.Button( ">>>>" , id="sonuchesapla", n_clicks=0 , color="success",  className="me-1", ),

], header="Sonuçları Hesapla", className="mb-3", style={'width': '100%','margin-right' : '200px',})





modelsonuclari2 =   dbc.Spinner(children=[ dbc.Toast([], header="Model Sonuçları", id="modelsonucgoster", className="mb-3", style={'width': '200%', 'height':'200%'} )
                        ], size="lg", color="primary", type="border" )


modelleme = dbc.Container([

            dbc.Row([
                dbc.Col(html.Br(), width=20),
                dbc.Col([parametre1, parametre2], xs=6, sm=6, md=3, lg=3, xl=3,  width={'offset': 0}),
                dbc.Col([html.Div([modelsonuclari2])],  xs=14, sm=14, md=17, lg=17, xl=17 ),
])  ])











################################################################################
aciklanabilir1 =   dbc.Spinner(children=[ dbc.Toast([], header="LIME", id="limegoster", className="mb-3", style={'width': '100%'} )
                        ], size="lg", color="primary", type="border" )

aciklanabilir2 =   dbc.Spinner(children=[ dbc.Toast([], header="SHAP", id="shapgoster", className="mb-3", style={'width': '130%'} )
                        ], size="xxl", color="primary", type="border" )

aciklanabilir3 =   dbc.Spinner(children=[ dbc.Toast(["Veri setinin boyutuna bağlı olarak analiz uzun sürebilir "], header="Genel Bilgi", id="shapgoster2", className="mb-3", style={'width': '130%'} )
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
                    dbc.Col(dbc.NavbarBrand("      xAI: Açıklanabilir Yapay Zeka Yazılımı ",style=dict(fontSize = 20,fontWeight='bold',fontFamily= "Bahnschrift", lineHeight= 1), className="ms-2")),
                    align="center",
                    className="g-0",
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.Row(
                [
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse(
                        dbc.Nav(
                            [
                                dbc.NavItem(dbc.NavLink("      Giriş", href="/pages_1", id="page-1-link1",active="exact",className="fas fa-home")),
                                dbc.NavItem(dbc.NavLink("      Dosya Yükleme ", href="/pages_2" ,id="page-2-link1",active="exact",className="fas fa-folder-open")),
                                dbc.NavItem(dbc.NavLink("      Veri Önişleme", href="/pages_3", id="page-3-link1",active="exact",className="fas fa-cogs")),
                                dbc.NavItem(dbc.NavLink("      Modelleme", href="/pages_4", id="page-4-link1",active="exact",className="fas fa-laptop")),
                                dbc.NavItem(dbc.NavLink(       "LIME ", href="/pages_5",id="page-5-link1",active="exact",className="fas fa-laptop")),
                                dbc.NavItem(dbc.NavLink(       "SHAP ", href="/pages_6",id="page-6-link1",active="exact",className="fas fa-laptop")),

                                dbc.NavItem(
                                    dbc.NavLink("     Alıntılama ", href="/pages_7", id="page-7-link1",active="exact",className="far fa-chart-bar"),

                                    className="me-auto",
                                ),


                                dbc.NavItem(dbc.Button("Türkçe", href='/apps/app1',color="success", size="sm",active=True,className="mr-1",id='btn-nclicks-1',style={ 'color':'white' , 'border':'1.5px black solid'}),),
                                dbc.NavItem(dbc.Button("English", href='/apps/app2',color="success", size="sm",active=True,className="mr-1",id='btn-nclicks-2',style={ 'color':'white' , 'border':'1.5px black solid'}),
  ),
                            ],
                            # make sure nav takes up the full width for auto
                            # margin to get applied
                            className="w-100",
                        ),
                        id="navbar-collapse",
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
dcc.Slider(50, 100, 5, value=80, id='traintestslider', marks=None, tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(0, 10, 1, value=0, id='kfoldslider', marks=None,   tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(5, 50, 1, value=5, id='ntekrarslider', marks=None,      tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(0, 100, 5, value=50, id='ntekrarsplitslider', marks=None,    tooltip={"placement": "bottom", "always_visible": True} ) ,
dcc.Slider(5, 10, 1, value=5, id='nestedickfoldslider', marks=None,  tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(5, 10, 1, value=50, id='diskfoldslider', marks=None, tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(5, 10, 1, value=5, id='repatkfoldslider', marks=None, tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Slider(5, 10, 1, value=50, id='ntkrarslider', marks=None, tooltip={"placement": "bottom", "always_visible": True} ),
dcc.Dropdown([0],id='pozitif_deger'),
dcc.RadioItems( options=[],  value='5', id = 'smoteradio', inline=False,),
dcc.Slider(1, 10, 1, value=2, id='optimizekfoldslider', marks=None,    tooltip={"placement": "bottom", "always_visible": True} ),
])

content = html.Div( children=giris_sayfasi, id="page-content1",)

layout = html.Div([ offcanvas,
                    html.Div([gereksiz], style={'display': 'none'}),
                    navbar1,     dcc.Location(id="urla5"),     content,
                    html.Div(id='verideposu', style={'display': 'none'}),
                    html.Div(id='ozellikdf', style={'display': 'none'}),
                    html.Div(id='hedefdf', style={'display': 'none'}),])


################################################################################
@app.callback(
    [Output('page-1-link1', 'active'),Output('page-2-link1', 'active'),Output('page-3-link1', 'active'),Output('page-4-link1', 'active'),Output('page-5-link1', 'active'),Output('page-6-link1', 'active'),Output('page-7-link1', 'active')],
    [Input('urla5', 'pathname')],
)
def toggle_active_links(pathname):
    if pathname == '/':
        # Treat page 1 as the homepage / index
        return True, False, False,False, False,False, False
    return [pathname == '/pages-{i}' for i in range(1, 8)]
#sayfa geçişleri
@app.callback(Output("page-content1", "children"), [Input("urla5", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/pages_1"]:
        return giris_sayfasi
    elif pathname == "/pages_2":
        return veri_yukleme
    elif pathname == "/pages_3":
        return veri_onisleme
    elif pathname == "/pages_4":
        return modelleme
    elif pathname == "/pages_5":
        return aciklanabilir
    elif pathname == "/pages_6":
        return aciklanabilir_SHAP
    elif pathname == "/pages_7":
        return html.Div()

    else:
        return giris_sayfasi

################################################################################
 #canvas

@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

################################################################################
#data uploaded

@app.callback(Output('verideposu', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),  )


def updateupload(contents, filename):

    if contents is not None:
        jsonfile = function_1.dosyaoku(contents, filename)
        return jsonfile


################################################################################

@app.callback(  Output('output-data-upload-2', 'children'),
                Input('verideposu', 'children') )

def alanupdate(veri):
    if veri is not None:
        try:
            dff=pd.read_json(veri , orient='split')
            oznitelik_secici= dcc.Dropdown(dff.columns, id='oznitelik_column',
            multi=True, className="", value=dff.columns, placeholder="Hedef/sonuç alanı için yalnızca bir değişken seçiniz:")
            hedef_secici= dcc.Dropdown(dff.columns, id='hedef_column',  placeholder="En az 1 alan seçiniz:",
            multi=False, className="")
            # pozitif_secici= dcc.Dropdown(dff.columns, id='pozitif_deger',  placeholder="İlgilenilen Sınıfı Seçiniz:",
            # multi=False, className="")

            toast0 = dbc.Toast([function_1.veriyazdir(veri)], header="Yüklenen Veri Seti", className="mb-3", style={'width': '100%'})
            toast1 = dbc.Toast(   [oznitelik_secici], header="Öznitelik alanlarını seçiniz", className="mb-3", style={'width': '100%'})
            toast2 = dbc.Toast(   [hedef_secici],    header="Hedef/Sonuç alanı seçiniz", className="mb-3", style={'width': '100%'})
            toast3 = dbc.Toast(   [html.Div(id="pozitifsecici") ],   header="İlgilenilen sınıfı seçiniz", className="mb-3", style={'width': '100%'})
            ozelliksecici = dbc.Row([   dbc.Col(toast1), dbc.Col(toast2), dbc.Col(toast3), ])
            bilgiver = html.Div( [  dbc.Button(    "Kaydet",      id="veri_kaydet",   color="info",
                                                className="mb-3",     n_clicks=0,   ),
                                    dbc.Toast(    [html.P("Veri önişleme adımına geçebilirsiniz", className="mb-0"),
                                    html.Br(),
                                    dbc.Button('Veri Önişleme->', href="/pages_3", color="success")],     id="simple-toast",
                                                header="Başarılı bir şekilde kayıt yapıldı",
                                                dismissable=True,     is_open=False,
                                                style={"position": "fixed", "top": 20, "right": 10, "width": 250},   ),       ] )



            return html.Div([toast0, ozelliksecici, html.Hr(), bilgiver ])




        except Exception as e:
            print(e)
            return html.Div([html.Hr(),
                         html.H4('Dosya Girişi yapılmadı.'),])


################################################################################
@app.callback(
    Output("pozitifsecici", "children"),
    Input("hedef_column","value"),
    State("verideposu", "children"), )

def pozitifsinifsec(sinif, veri):
    if sinif is not None:

        dff=pd.read_json(veri , orient='split')
        print(dff[sinif].unique())
        a = html.Div(dcc.Dropdown(dff[sinif].unique(), id='pozitif_deger',  placeholder="Pozitif sınıfı seçiniz:", multi=False, className=""))
        return a






######################################################
@app.callback(
    Output("simple-toast", "is_open"),
    Output("ozellikdf","children"),
    Output("hedefdf","children"),
    Input("veri_kaydet", "n_clicks"),
    State("oznitelik_column", "value"),
    State("hedef_column", "value"),
    State("verideposu", "children"),
    State("pozitif_deger","value")
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
        dcc.Download(id="download-dataframe-xlsx"),
        dbc.Button("Önişlem yapılan veri setini indir", id="open", n_clicks=0),
        dbc.Modal(
            [
               dbc.ModalHeader(dbc.ModalTitle("")),
                dbc.ModalBody("This is the content of the modal", id= "onislemverisi"),
                dbc.ModalFooter(
                    dbc.Button(
                        "Kapat", id="close", className="ms-auto", n_clicks=0
                    )
                ),
            ],
            id="modal",
            size="xl",
            is_open=False,
        ),
    ]
)



@app.callback(
    Output("output-veri-analiz", "children"),
    Input("analizbuton", "n_clicks"), )
def analizfonksiyon(n):
    global df, oz_df, hedef_df
    global new_df, new_oz_df, new_hedef_df
    OZ_YENİ=oz_df.copy()
    OZ_YENİ , onehot_liste = function_1.categoric1(OZ_YENİ)
    if n != 0 and df is not None and oz_df is not None and hedef_df is not None:
        kayipverisonuc=function_1.is_kayipveri(oz_df)
        aykiriverisonuc = function_1.is_aykiriveri(df)
        dengesizverisonuc = function_1.isdengesiz(hedef_df)
        #analizverileri = function_1.verianalizi(df, oz_df, hedef_df)
        kayipveri1= html.Div([dbc.Card([  dbc.CardBody([ html.H6("Kayıp Veri Analiz Sonucu: ", className="card-title"),
                                html.P( str(kayipverisonuc) + " adet kayıp veri bulundu. ", className="card-text"),
                                html.Div( dcc.RadioItems( options=[  {'label': 'Kayıp değer içeren satırlar veri setinden çıkarılsın.  ', 'value': '0'}, {'label': 'Random Forest yöntemi ile atama yapılsın.', 'value': '1'} ],
                                            value='1', id = 'kayipdegerradio', inline=False,
                                            className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                            style={"width": "100%"},  className="card text-white bg-danger mb-3", ) ]  )
        kayipveri2= dbc.Card([  dbc.CardBody([ html.H6("Kayıp Veri Analiz Sonucu: ", className="card-title"),
                                        html.P( "Veri setinde kayıp veri bulunmamaktadır.", className="card-text"), ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-success mb-3", )

        aykiriveri1= dbc.Card([  dbc.CardBody([ html.H6("Aykırı/Aşırı Değer Analiz Sonucu: ", className="card-title"),
                                        html.P( str(aykiriverisonuc) + " adet aykırı/aşırı değer bulundu. Aykırı/aşırı değerler silinsin mi?.", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'Evet', 'value': '0'}, {'label': 'Hayır', 'value': '1'} ],
                                                    value='1', id = 'aykiridegerradio', inline=False,
                                                    className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-danger mb-3", )
        aykiriveri2= dbc.Card([  dbc.CardBody([ html.H6("Aykırı/Aşırı Değer Analiz Sonucu: ", className="card-title"),
                                                html.P( "Veri Setinde aykırı/aşırı değer bulunmamaktadır.", className="card-text"), ] ),  ],
                                                            style={"width": "100%"},  className="card text-white bg-success mb-3", )




        veridonusum1= dbc.Card([  dbc.CardBody([ html.H6("Dönüşüm yöntemleri", className="card-title"),
                                        html.P( "Veri dönüşümü için aşağıdaki yöntemlerden birini seçiniz", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'Normalizasyon', 'value': '0'},
                                                                             {'label': 'Min-Max Standardizasyonu', 'value': '1'},
                                                                             {'label': 'Standardizasyon', 'value': '2'},
                                                                             {'label': 'Robust Standardizasyonu', 'value': '3'},
                                                                             {'label': 'Hiçbiri', 'value': '4'}, ],
                                                    value='4', id = 'veridonusumradio', inline=False,
                                                    className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-info mb-3", )



        ozsecimi1= dbc.Card([  dbc.CardBody([ html.H6("Öznitelik Seçimi", className="card-title"),
                                        html.P( "Öznitelik seçimi için aşağıdaki yöntemlerden birini seçiniz", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'LogisticRegression', 'value': '0'},
                                                                             {'label': 'ExtraTreesClassifier', 'value': '1'},
                                                                             {'label': 'RandomForestClassifier', 'value': '2'},
                                                                             {'label': 'LASSO', 'value': '3'},
                                                                             {'label': 'Boruta', 'value': '4'},
                                                                              {'label': 'Hiçbiri', 'value': '5'},],
                                                        value='5', id = 'ozscmradio', inline=True,
                                                       className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-info mb-3", )

        print(onehot_liste)
        print("smoteöncesionehotliste")
        if dengesizverisonuc==True and len(onehot_liste) == 0:
            smote1= dbc.Card([  dbc.CardBody([ html.H6("Sınıf Dengesizliği Analizi", className="card-title"),
                                        html.P( "Veri setinde sınıf dengesizliği problemi bulunmaktadır. Sınıf dengesizliği problemini gidermek için  aşağıdaki yöntemlerden birini seçiniz.", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'SMOTE', 'value': '0'},{'label': 'SMOTETomek', 'value': '1'},{'label': 'Hiçbiri', 'value': '2'}],

                                                    value='2', id = 'smoteradio', inline=False,
                                                    className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-info mb-3", )
        if dengesizverisonuc==True and len(onehot_liste) !=0:
            smote1= dbc.Card([  dbc.CardBody([ html.H6("Sınıf Dengesizliği Analizi", className="card-title"),
                                        html.P( "Veri setinde sınıf dengesizliği problemi bulunmaktadır. Sınıf dengesizliği problemini gidermek için  aşağıdaki yöntemlerden birini seçiniz.", className="card-text"),
                                        html.Div( dcc.RadioItems( options=[  {'label': 'Hiçbiri', 'value': '2'},{'label': 'SMOTE-NC', 'value': '3'}],

                                                    value='2', id = 'smoteradio', inline=False,
                                                    className="md-3 btn-group gap-3"   ), )  ] ),  ],
                                                    style={"width": "100%"},  className="card text-white bg-info mb-3", )
        if dengesizverisonuc ==False:
            smote1= dbc.Card([  dbc.CardBody([ html.H6("Sınıf Dengesizliği Analizi", className="card-title"),
                                        html.P( "Veri setinde sınıf dengesizliği problemi bulunmamaktadır.", className="card-text") ]) ])


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






        verionsilemekaydet = html.Div([dbc.Button('Kaydet', id="verionislemkaydet", color="primary", n_clicks=0,className="me-1" ),

                                        dbc.Toast(    [html.P("Modelleme aşamasına geçebilirsiniz.", className="mb-0"),
                                            html.Br(),
                                            dbc.Button('Modelleme->', href="/pages_4", color="success"),html.Br(),html.Br(),
                                            dbc.Row([dbc.Col(modal)])],     id="simple-toast2",
                                            header="Başarılı bir şekilde kayıt yapıldı.",
                                            dismissable=True,     is_open=False,
                                            style={"position": "fixed", "top": 66, "right": 10, "width": 250},  ),html.Br(),


                                        ])


        return dbc.Toast([kayipveri3, aykiriveri3, veridonusum1, ozsecimi1, smote1, verionsilemekaydet ], header="Veri Analiz Sonuçları", style={'width': '100%'})
    elif n != 0:
        return dbc.Toast(['Veri Giriş İşlemi Başarısız Veri Yükleme Sayfasına Dönünüz!!'], header="Veri Analiz Sonuçları", style={'width': '100%'})





@app.callback(
    Output("modal", "is_open"),
    Output("onislemverisi", "children"),
    Input("open", "n_clicks"),
    Input("close", "n_clicks"),
    State("modal", "is_open"), )
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
    downloadbuton=dbc.Button("İndir" , id="excelkaydet", n_clicks=0)
    if n1 or n2:
        return not is_open, html.Div([table, downloadbuton] )
    return is_open, html.Div([table, downloadbuton] )


@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("excelkaydet", "n_clicks") )

def veridownload(n):
    global new_df
    if n!=0:
        return dcc.send_data_frame(new_df.to_excel, "mydf.xlsx", sheet_name="Sheet_name_1")







@app.callback(
    Output("simple-toast2", "is_open"),
    Input("verionislemkaydet", "n_clicks"),
    State("kayipdegerradio", "value" ),
    State("aykiridegerradio", "value"),
    State("veridonusumradio", "value"),
    State("ozscmradio", "value"),
    State("smoteradio", "value"),
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
    Output("optimizasyonsecenek", "children"),
    Input("optimizasyonslider", "value"), )

def opt(n):

    if n!=str(0):
        return html.Div([dcc.Slider(1, 10, 1, value=2, id='optimizekfoldslider')], style={'display': 'none'} )
    if n==str(0):
        aa = dcc.Slider(2, 10, 1, value=2, id='optimizekfoldslider', marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True} ),
        return aa











################################################################################
#sonuç hesaplamak
import numpy as np
import time
@app.callback(
    Output("modelsonucgoster", "children"),
    Input("sonuchesapla", "n_clicks"),
    State("optimizasyonslider","value"),
    State("veriseti-bolum-secici", "value" ),
    State("traintestslider", "value"),
    State("ntekrarslider", "value"),
    State("ntekrarsplitslider", "value"),
    State("kfoldslider", "value"),
    State("repatkfoldslider", "value"),
    State("ntkrarslider", "value"),
    State("nestedickfoldslider", "value"),
    State("diskfoldslider", "value"),
    State("modelsecici", "value"),
    State("optimizekfoldslider", "value"),

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
            return html.H5("Ön işlem adımları tamamlanmamıştır. Veri önişleme sayfasına dönünüz.")
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



        adf = pd.DataFrame( {"Model isimleri": model_puanlari.keys(), "Doğruluk": model_puanlari.values(),"F1_score": model_puanlari1.values(),"Precision": model_puanlari2.values(),"Recall": model_puanlari3.values(),"AUC": model_puanlari4.values(), "FPR": model_puanlari5.values(), "TPR": model_puanlari6.values(),"NPV": model_puanlari7.values()} )



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
            dbc.Button("LIME>>", href="/pages_5"),
            ]),

        print(model_puanlari)

        return sonuclar

#####################################

@app.callback(Output('limegoster', 'children'),
              Input('urla5', 'pathname'))
def display_page(pathname):
    global new_df, new_oz_df, new_hedef_df,new_oz_onehot, onehot_liste
    global modeller,  model_puanlari,model_puanlari1,model_puanlari2,model_puanlari3,model_puanlari4,model_puanlari5,model_puanlari6,model_puanlari7
    if pathname == '/pages_5':
        try:
            if modeller == {} or new_df is None:
                return html.Div("Sınıflandırıcı çalışmadı. Önce sınıflandırma yapınız.")
            else:
                modellistesi = list(modeller.keys())
                limesecici= dcc.Dropdown( modellistesi , id='limemodel', clearable=False, multi=False, className="", value=modellistesi[0])
                limetablo = dash_table.DataTable( id='limetablo1',
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
                limesonucgoster = html.Div(id='limesonuc1')
                shapsecici= dcc.Dropdown( modellistesi , id='shapmodel', clearable=False,  multi=False, className="", value=modellistesi[0])


                return html.Div([limesecici,html.Br(), limetablo, html.Br(), limesonucgoster])

        except:
            return html.Div("Sınıflandırıcı çalışmadı. Önce sınıflandırma yapınız.")



@app.callback(Output('shapgoster', 'children'),
              Input('urla5', 'pathname'))
def display_page(pathname):
    global new_df, new_oz_df, new_hedef_df,new_oz_onehot, onehot_liste
    global modeller,  model_puanlari,model_puanlari1,model_puanlari2,model_puanlari3,model_puanlari4,model_puanlari5,model_puanlari6,model_puanlari7
    if pathname == '/pages_6':
        try:
            if modeller == {} or new_df is None:
                return html.Div("Sınıflandırıcı çalışmadı. Önce sınıflandırma yapınız.")
            else:
                modellistesi = list(modeller.keys())
                shapturleri = ["Shap Özet Grafiği", "Öznitelik Önemi Grafiği"]
                shapsonucgoster = html.Div(id='shapsonuc1')



                print(new_hedef_df.unique())

                #a = html.Div(dcc.Dropdown(new_hedef_df.unique(), id='pozitif_deger2',  placeholder="Pozitif sınıfı seçiniz:", multi=False, className=""))
                shapsecici= dcc.Dropdown( modellistesi , id='shapmodel', clearable=False,  multi=False, className="", value=modellistesi[0])
                shapsecici2= dcc.Dropdown( shapturleri , id='shaptur', clearable=False,  multi=False, className="", value=shapturleri[0])


                return html.Div([shapsecici, shapsecici2,  html.Br(), shapsonucgoster ])

        except:
            return html.Div("Sınıflandırıcı çalışmadı. Önce sınıflandırma yapınız.")






import matplotlib.pyplot as plt


@app.callback(Output('limesonuc1', 'children'),
              Input('limetablo1',"derived_virtual_selected_rows"),
              Input('limemodel', 'value'), )

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
        print(limelist.columns)

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
                columns=[{"name": i, "id": i} for i in limelist.columns],editable=False,
                        
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
            ),dbc.Button("SHAP >>", href="/pages_6"),])



        return obj,html.Br(),html.Br(),sonuclar
    except Exception as e:
        print(e)
        return "LIME hesaplanamadı"
from shap.plots._beeswarm import summary_legacy

@app.callback(Output('shapsonuc1', 'children'),
              Input('shapmodel', 'value'),
              Input('shaptur', 'value') )

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

            adf = pd.DataFrame( {"Öznitelik": shap_importance.keys(), "Öznitelik Önemi": shap_importance.values()} )

            adf = adf.sort_values(by=['Öznitelik Önemi'] , ascending=False)


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
            if tur=="Shap Özet Grafiği":

                 shap.summary_plot(shap_values, x_test, class_names=new_hedef_df.unique(), color='coolwarm', feature_names = new_oz_onehot.columns, show=(False))
            elif tur=="Öznitelik Önemi Grafiği":

                feat_importances = pd.Series(shap_importance.values(), index=x_test.columns)
                ax=feat_importances.nlargest(20).plot(kind='barh',figsize=(7, 6),color='#86bf91', zorder=2, width=0.3)
                ax.set_xlabel("Normalize edilen shap önemlilik değerleri", labelpad=20, weight='bold', size=12)
                ax.set_ylabel("Öznitelik", labelpad=20, weight='bold', size=12)
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

            adf = pd.DataFrame( {"Öznitelik": shap_importance.keys(), "Öznitelik Önemi": shap_importance.values()} )

            adf = adf.sort_values(by=['Öznitelik Önemi'] , ascending=False)


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
            if tur=="Shap Özet Grafiği":

                shap.summary_plot(shap_values[1], x_test, class_names=new_hedef_df.unique(), color='coolwarm', feature_names = new_oz_onehot.columns, show=(False))
            elif tur=="Öznitelik Önemi Grafiği":

                feat_importances = pd.Series(shap_importance.values(), index=x_test.columns)
                ax=feat_importances.nlargest(20).plot(kind='barh',figsize=(7, 6),color='#86bf91', zorder=2, width=0.3)
                ax.set_xlabel("Normalize edilen shap önemlilik değerleri", labelpad=20, weight='bold', size=12)
                ax.set_ylabel("Öznitelik", labelpad=20, weight='bold', size=12)
            else:

                feat_importances = pd.Series(shap_importance.values(), index=x_test.columns)
                feat_importances.nlargest(20).plot(kind='barh',figsize=(7, 6))
                ax.set_xlabel("Normalize edilen shap önemlilik değerleri", labelpad=20, weight='bold', size=12)
                ax.set_ylabel("Öznitelik", labelpad=20, weight='bold', size=12)
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
        return "shap hesaplanamadı"
