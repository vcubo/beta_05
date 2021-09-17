import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats


import plotly.graph_objects as go
import plotly.express as px

import streamlit as st
import hydralit_components as hc

from utils import (partials, df_stats, filter_gen, const_figures, fit_distr, compute_partials, update_impact, scatter_hist)


st.set_page_config(layout='wide',initial_sidebar_state='collapsed',)

# DATA IMPORT #
@st.cache
def import_df(path):
    df = pd.read_csv(path) #Important: set raw URL
    return df

db_raw_path = 'https://raw.githubusercontent.com/vcubo/beta_0.1/main/VCDB_210828v0.csv'
df = import_df(db_raw_path) # main dataframe for general use

risk_dict = {'Social':['SOC', 'SOC_MIT', 'SOC (NM)' ],
        'Procurement':['PROC', 'PROC_MIT', 'PROC (NM)'],
        'Engineering':['ENG', 'ENG_MIT', 'ENG (NM)'],
        'Weather':['WEA', 'WEA_MIT', 'WEA (NM)'],
        'Management':['MGM', 'MGM_MIT', 'MGM (NM)']
        }

# MODEL #
## Underlying distriburion (modeling & regression results)
df_distrib = pd.DataFrame([['lognormal', 0.1, 0.3, -0.3]], columns = ['type', 'mu', 'sigma', 'shift'])
## Coefficients (modeling & regression results):
df_coef = pd.DataFrame([[0.25,0.05,0.4,0.3,0.2,0.6,0.35,0.5,0.6,0.8,0.7]],
                       columns = ['COUNTRY', 'LOB', 'SITE', 'PSIZE', 'CSIZE','SOC','PROC','ENG', 'WEA', 'MGM','MIT_ef'])
## List of variables:
df_part_index = ['Country','LoB','Site','Project Size', 'Contractor',
                               'Social', 'Procurement', 'Engineering', 'Weather', 'Management']




if __name__ =="__main__":

    st.sidebar.image('/vcubo_beta2.png',width=150)
    st.sidebar.write('Data driven QSRA - v0.5 beta')

    # specify the menu definition we'll stick in the sidebar
    side_menu_data = [
        {'id':'First page', 'icon': "bi bi-collection", 'label':"Explore database",'ttip':"Quick product intro"},
        {'id':'Second page', 'icon': "bi bi-file-bar-graph", 'label':"My Company",'ttip':"Quick product intro"},
        {'id':'Third page', 'icon': "bi bi-bar-chart-fill", 'label':"Project QRA",'ttip':"Quick product intro"},
        {'id':'Fourth page', 'icon': "bi bi-sliders", 'label':"QRA mitigation",'ttip':"Quick product intro"},
        {'id':'Fifth page', 'icon': "fas fa-brain", 'label':"Prescriptive AI",'ttip':"Quick product intro"},
    ]

    # specify the primary menu definition
    menu_data = [
        {'id':'First page', 'icon': "bi bi-collection", 'label':"Explore database",'ttip':"Quick product intro"},
        {'id':'Second page', 'icon': "bi bi-file-bar-graph", 'label':"My Company",'ttip':"Quick product intro"},
        {'id':'Third page', 'icon': "bi bi-bar-chart-fill", 'label':"Project QRA",'ttip':"Quick product intro"},
        {'id':'Fourth page', 'icon': "bi bi-sliders", 'label':"QRA mitigation",'ttip':"Quick product intro"},
        {'id':'Fifth page', 'icon': "fas fa-brain", 'label':"Prescriptive AI",'ttip':"Quick product intro"},

    ]

    # we can override any part of the primary colors of the menu
    #over_theme = {'txc_inactive': '#FFFFFF','menu_background':'','txc_active':'yellow','option_active':'blue'}
    over_theme = {'txc_inactive': '#FFFFFF', 'menu_background':'#494854'}

    menu_id = hc.nav_bar(menu_definition=menu_data,override_theme=over_theme, home_name='Home',)

    #with st.sidebar:
        #menu_id = hc.nav_bar(menu_definition=menu_data,key='sidebar',override_theme=over_theme,first_select=6) #login_name='Login',

    #get the id of the menu item clicked
    #st.info(f"{menu_id}")
    #st.info(f"{side_menu_id=}")


def main():
    # Register your pages
    pages = {
        "Home": home_page,
        "First page": page_one,
        "Second page": page_two,
        "Third page": page_three,
        "Fourth page": page_four,
        "Fifth page": page_five
    }

    # Widget to select your page, you can choose between radio buttons or a selectbox
    #page = st.sidebar.selectbox("Select your page", tuple(pages.keys()))
    #page = st.sidebar.radio("Select your page", tuple(pages.keys()))
    #page_1 = st.sidebar.button('First page')
    #page_2 = st.sidebar.button('Second page')
    #if page_2:
    #    page = 'Second page'
    #if page_1:
    #    page = 'First page'
    if menu_id:
        page = menu_id
    #if side_menu_id:
    #    page = side_menu_id
    # Display the selected page with the session state
    pages[page]()

def home_page():
    st.image('/homepage.png')
    # ...

def page_one():
    st.subheader("Projects general database")
    #gen_expand = st.expander('. . . ', expanded=False)
    #if exp_gen:
    #    gen_expand = True
    #with gen_expand:
## GENERAL FILTERS ##
    if 'select_country' not in st.session_state: st.session_state.select_country = "All"
    if 'select_lob' not in st.session_state: st.session_state.select_lob = "All"
    if 'select_site' not in st.session_state: st.session_state.select_site = "All"
    if 'select_prsize' not in st.session_state: st.session_state.select_prsize = "All"
    if 'select_csize' not in st.session_state: st.session_state.select_csize = "All"
    if 'hist_xbin_size' not in st.session_state: st.session_state.hist_xbin_size = 0.05

    gf01, gf02, gf03, gf04, gf05, gf06 = st.columns(6)
    with gf01: st.selectbox('Country',['All']+df['COUNTRY'].unique().tolist(), key='select_country')
    with gf02: st.selectbox('LoB',['All']+df['LOB'].unique().tolist(), key='select_lob')
    with gf03: st.selectbox('Site conditions',['All']+df['SITE'].unique().tolist(), key='select_site')
    with gf04: st.selectbox('Project size',['All']+df['PR_SIZE'].unique().tolist(), key='select_prsize')
    with gf05: st.selectbox('Contractor size',['All']+df['MC_SIZE'].unique().tolist(), key='select_csize')
    with gf06: st.slider('General histograms bin width', 0.01, 0.1, key='hist_xbin_size')

    selection_gen = [st.session_state.select_country, st.session_state.select_lob, st.session_state.select_site, st.session_state.select_prsize, st.session_state.select_csize] #list of filters applied
    filter_list = filter_gen(selection_gen,df)
    df_filter = df[filter_list]
    st.session_state.df_filter = df_filter
    #if 'hist_xbin_size1' not in st.session_state:
    #    st.session_state.hist_xbins_size1 = hist_xbins_size


## STATISTICS
    projecs_num = len(df_filter) #number of projects (filters aplied)
    selection_statistics = df_stats(df_filter)

## GENERAL FIGURES
    figures_general = const_figures(df, st.session_state.df_filter, st.session_state.hist_xbin_size, df_coef, df_part_index)
    st.info('Showing statistics of '+str(len(df[filter_list]))+' projects out of '+str(len(df))+' projects in database')
    gen01, gen02 = st.columns(2)
    with gen01:
        st.write("Total composed uncertainty and risk's impact distribution")
        st.plotly_chart(figures_general[0])
    with gen02:
        st.write("Decomposed uncertainty and risk's impact distribution")
        st.plotly_chart(figures_general[1])
    st.plotly_chart(figures_general[2])



    # ...

def page_two():
    @st.cache(ttl=2*60*60)
    def df_com (df):
        df_c = df[(df['COUNTRY']=='Argentina')&((df['LOB']=='O&G - Downstream')^(df['LOB']=='O&G - Upstream'))]
        return df_c

    df_c1 = df_com(df)
    figures_c001 = const_figures(df_c1, st.session_state.df_filter, st.session_state.hist_xbin_size, df_coef, df_part_index)
    figures_c001[0].data[1].name = 'Total deviation -company'
    figures_c001[1].data[0].name = 'Uncertainty -company'
    figures_c001[1].data[1].name = 'Risk events impact -company'

    st.write(str(len(df_c1))+' projects registered')
    com01, com02 = st.columns(2)
    with com01:
        st.plotly_chart(figures_c001[0])
    with com02:
        st.plotly_chart(figures_c001[1])
    st.plotly_chart(figures_c001[2])

def page_three():
    # PREDICTIVE ANALYTICS (PROJECT) #
    # Initial datafeame upload:
    df2 = import_df(db_raw_path) # secondary dataframefor individual project operations
    # Project Characterizarion:
    st.header("Project setup")
    pr_setup = st.expander("EXPAND", expanded=True)
    with pr_setup:
        prf01, prf02, prf03, prf04, prf05, prf06 = st.columns(6)
        if 'select_country2' not in st.session_state: st.session_state.select_country2 = "All"
        if 'select_lob2' not in st.session_state: st.session_state.select_lob2 = "All"
        if 'select_site2' not in st.session_state: st.session_state.select_site2 = "All"
        if 'select_prsize2' not in st.session_state: st.session_state.select_prsize2 = "All"
        if 'select_csize2' not in st.session_state: st.session_state.select_csize2 = "All"
        if 'hist_xbin_size2' not in st.session_state: st.session_state.hist_xbin_size2 = 0.05
        if 'hist_xbin_size3' not in st.session_state: st.session_state.hist_xbin_size3 = 0.05

        with prf01: st.selectbox('Country',['All']+df2['COUNTRY'].unique().tolist(), key='select_country2')
        with prf02: st.selectbox('LoB',['All']+df2['LOB'].unique().tolist(), key='select_lob2')
        with prf03: st.selectbox('Site conditions',['All']+df2['SITE'].unique().tolist(), key='select_site2')
        with prf04: st.selectbox('Project size',['All']+df2['PR_SIZE'].unique().tolist(), key='select_prsize2')
        with prf05: st.selectbox('Contractor size',['All']+df2['MC_SIZE'].unique().tolist(), key='select_csize2')
        with prf06:
            st.slider('General histograms bin width', 0.01, 0.1, key='hist_xbin_size2')
            st.slider('Fitting curve step length', 0.01, 0.1, key='hist_xbin_size3')

        selection_pro = [st.session_state.select_country2, st.session_state.select_lob2, st.session_state.select_site2, st.session_state.select_prsize2, st.session_state.select_csize2] #list of filters applied
        filter_list2 = filter_gen(selection_pro,df)


        st.session_state.df_p1b = df2[filter_list2].copy(deep=True)

        ## EXAMPLE PROJECT - SIMULATION
        df_p1 = df2[(df2['COUNTRY']=='Argentina')&(df2['LOB']=='O&G - Upstream')&(df2['PR_SIZE']=='<20')&(df2['MC_SIZE']=='Small')&(df2['SITE']=='Harsh >50% of activity duration')]

        figures_p01 = const_figures(df_p1, st.session_state.df_p1b, st.session_state.hist_xbin_size2, df_coef, df_part_index)
        figures_p1_fit = fit_distr(st.session_state.df_p1b, st.session_state.hist_xbin_size3)
        st.session_state.figures_p1_fit = figures_p1_fit
        pre_stat = df_stats(st.session_state.df_p1b)
        st.session_state.pre_stat = pre_stat

        st.subheader('Distribution of similar projects ('+str(len(st.session_state.df_p1b))+' projects):')
        pr01a, pr01b = st.columns(2)
        with pr01a: st.plotly_chart(figures_p01[0])
        with pr01b: st.plotly_chart(figures_p01[1])

    st.header("Project analysis")
    pr_analysis = st.expander('EXPAND')
    with pr_analysis:
        pr02a, pr02b = st.columns(2)
        with pr02a: st.plotly_chart(st.session_state.figures_p1_fit[0])
        with pr02b: st.plotly_chart(st.session_state.figures_p1_fit[1])
        with pr02a:
            st.subheader('Pre-mitigated project statistics')
            st.write("With a lognormal fit over delays' distribution of the "+str(len(st.session_state.df_p1b))+" projects selected.")
            st.write('Distribution mean: '+str(np.round(pre_stat['means'][2]*100,2))+'%(fit)')
            st.write('Distribution median (P50): '+str(np.round(pre_stat['median'][2]*100,2))+'%(fit)')
            st.write('Average deviation percentage attributed to Risk events: ')
        #    st.write(pre_stat)
        with pr02b:
            st.plotly_chart(figures_p01[2])


    st.header("Project's variables correlations")
    var_corr = st.expander('EXPAND')
    with var_corr:
        st.caption("Estimated total risks' impact vs risk type impact" )
        pr03a, pr03b, pr03c = st.columns((1,2,2))
        with pr03a:
            x3d_sel = st.radio('Risk event type', ['Social','Procurement','Engineering', 'Weather', 'Management'])
        with pr03b:
            partials_df_comp = compute_partials(st.session_state.df_p1b, df_part_index, df_coef)
            sel_dev_corr = sp.stats.pearsonr(partials_df_comp['DEV_EVE'],partials_df_comp[x3d_sel]) #Pearsons' correlation coefficient between composed risk events impact and selected risk type
            st.subheader("The Pearson's correlation coefficient between the total impact of risk events and the estimated impact of "+ x3d_sel+ " risk events " +str(np.round(sel_dev_corr[0],2))+".")
            st.subheader("The p-value of the correlation coefficient is "+str(np.round(sel_dev_corr[1]*100,2))+"%.")

        pr04a, pr04b = st.columns(2)
        with pr04a:
            st.plotly_chart(scatter_hist(partials_df_comp,risk_dict[x3d_sel][2]))
        with pr04b:
            st.plotly_chart(scatter_hist(partials_df_comp,x3d_sel))
    #st.plotly_chart(scatter_3dim (st.session_state.df_p1b, risk_dict[x3d_sel][0], risk_dict[x3d_sel][1], 'DEV_EVE', 'DEV_EVE', 'DEV_TOT'))

def page_four():
    st.header("Project - Predictive analytics")
    st.subheader("Analize risk mitigation impact and conduct a high-level QSRA")
    risk_mitigate = st.expander('MITIGATE', expanded=True)

    st.session_state.df_p1m = st.session_state.df_p1b.copy(deep=True)
    if 'soc_mit' not in st.session_state: st.session_state.soc_mit = 0
    if 'proc_mit' not in st.session_state: st.session_state.proc_mit = 0
    if 'eng_mit' not in st.session_state: st.session_state.eng_mit = 0
    if 'wea_mit' not in st.session_state: st.session_state.wea_mit = 0
    if 'mgm_mit' not in st.session_state: st.session_state.mgm_mit = 0

    with risk_mitigate:
        st.subheader("Risks mitigation " )
        pr05, pr05a, pr05b, pr05c,pr05c2, pr05d = st.columns((1,2,1,5,1,8))
        with pr05a:
            st.slider('Social risks',0.0, 1.0, step=0.2, help="Adjust restimated impact of risks according to projects' conditions", key='soc_mit')
            st.slider('Procurement risks',0.0, 1.0, step=0.2, key='proc_mit')
            st.slider('Engineering risks',0.0, 1.0, step=0.2, key='eng_mit')
            st.slider('Weather risks',0.0, 1.0, step=0.2, key='wea_mit')
            st.slider('Management risks',0.0, 1.0, step=0.2, key='mgm_mit')
            st.session_state.mitigation = [st.session_state.soc_mit, st.session_state.proc_mit, st.session_state.eng_mit, st.session_state.wea_mit, st.session_state.mgm_mit]
            reset_mit = st.button('Reset')
            #if reset_mit:
            st.session_state.df_p1m = update_impact(st.session_state.df_p1m, st.session_state.df_p1b, st.session_state.mitigation, df_coef)

    show_data = st.expander('DATA', expanded=False)
    with show_data:
        st.write(st.session_state.df_p1m[1])
        st.write(st.session_state.df_p1b)
        st.write(st.session_state.df_p1m[0])

        #with pr05c:
        #    st.subheader('Risk mitigation profile:')
        #    st.write("With a lognormal fit over delays' distribution of the "+str(len(st.session_state.df_p1b))+" projects selected.")
        #    st.write('Distribution mean: '+''+'(fit)')
        #    st.write('Distribution median (P50): ')
        #    st.write('Average deviation percentage attributed to Risk events: ')
    show_charts = st.expander('CHARTS', expanded=True)
    with show_charts:
        st.session_state.figures_p1m = const_figures(st.session_state.df_p1b, st.session_state.df_p1m[0], st.session_state.hist_xbin_size3, df_coef, df_part_index)
        st.session_state.figures_p1m_fit = fit_distr(st.session_state.df_p1m[0], st.session_state.hist_xbin_size3)
        st.session_state.pos_stat = df_stats(st.session_state.df_p1m[0])
        #st.write(pos_stat)
        with pr05c:
            st.header('Post-mitigation probabilities')
            st.subheader("With a lognormal fit over delays' distribution of the "+str(len(st.session_state.df_p1b))+" projects selected.")
            st.subheader('Distribution mean: '+str(np.round(st.session_state.pos_stat['means'][2]*100,2))+'%(fit) vs '+str(np.round(st.session_state.pre_stat['means'][2]*100,2))+'%(fit)')
            st.subheader('Distribution median (P50): '+str(np.round(st.session_state.pos_stat['median'][2]*100,1))+'%(fit) vs '+str(np.round(st.session_state.pre_stat['median'][2]*100,2))+'%(fit)')
            st.write('Average deviation percentage attributed to Risk events: ')
        #    df_polar =pd.DataFrame.from_dict({'Risk type':['Social', 'Procurement', 'Engineering', 'Weather', 'Management'], '% Mitigation':mitigation })
        #    polar_mit = px.line_polar(df_polar, r="% Mitigation", theta="Risk type", line_close=True, #color="% Mitigation",
        #                color_discrete_sequence=px.colors.sequential.Rainbow,
        #                template="plotly_dark")
        #    #mitigated_plot = scatter_hist(compute_partials(st.session_state.df_p1m, df_part_index),risk_dict[x3d_sel][2])
        #    st.plotly_chart(polar_mit)
        with pr05d:
            st.plotly_chart(st.session_state.figures_p1m[2])

        pr08a, pr08b, pr08c = st.columns((4,1,4))
        with pr08a: st.plotly_chart(st.session_state.figures_p1m[0])
        with pr08c: st.plotly_chart(st.session_state.figures_p1m[1])
        with pr08a:
            st.subheader('Pre-mitigation charts:')
            st.plotly_chart(st.session_state.figures_p1_fit[0])
        with pr08b:
            x = np.linspace(0,1,int(1/st.session_state.hist_xbin_size3))
            upd_mit_chart = st.button('Update mitigated fitting curve')
            cln_mit_chart = st.button('Clean fitting curve')
            mit_dist_chart = st.session_state.figures_p1_fit[1]
            if upd_mit_chart:
                st.session_state.figures_p1_fit[1].add_scatter(y = st.session_state.figures_p1m_fit[4], x = x)
            if cln_mit_chart:
                mit_dist_chart = st.session_state.figures_p1_fit[1]
        with pr08c:
            st.subheader('Pre and post-mitigation fitting curves:')
            st.plotly_chart(mit_dist_chart)



def page_five():
    st.header("Project and portfolio Prescriptive analytics")
    st.subheader("Optimize risk mitigation with the help of AI")

if __name__ == "__main__":
    main()
