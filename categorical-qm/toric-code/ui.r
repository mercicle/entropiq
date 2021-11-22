source('load-libraries.r')

addResourcePath(prefix = 'www', directoryPath = paste0(getwd(), '/www'))

card.column.width <- 4

ui <- dashboardPage(skin = "purple",
  dashboardHeader(title = tags$a(href='https://starstuffventures.com', tags$img(src='donut-logo-2.png',height=50)),
                  
                  tags$li(class = "dropdown",
                  tags$style(".main-header {max-height: 60px}"),
                  tags$style(".main-header .logo {height: 60px;}"),
                  tags$style(".sidebar-toggle {height: 20px; padding-top: 10px !important;}"),
                  tags$style(".navbar {min-height:20px !important}"),
                  tags$style("hr {
                              border: 0;
                              clear:both;
                              display:block;
                              width: 96%;               
                              background-color:#000000;
                              height: 1px;
                            }"))

                  
                  
  ),
  dashboardSidebar(
    width = 325,
    collapsed = FALSE,
    shinyjs::useShinyjs(),
    
    sidebarMenu(id='tabs',
                  menuItem("   Platform Statistics", tabName = "stats_tab", icon = icon("chart-bar")), #, "fa-2x"
                  menuItem("   Experiment Management", tabName = "data_mgmt_tab", icon = icon("database"),
                       menuSubItem("New Experiment", tabName = "data_mgmt_new_experiment", selected = T)),
                  menuItem("   Visualize", tabName = "visualize_tab", icon = icon("project-diagram"))
    )
    
  ),
  dashboardBody(
    
      tabItems(
      
      tabItem(tabName = "stats_tab",
              br(),
              fluidRow(
                column(card.column.width, offset = 2,infoBoxOutput("experiment_1_info_card", width='12')),
                column(card.column.width, infoBoxOutput("experiment_2_info_card", width='12')),
              ),
              fluidRow(
                column(card.column.width,  offset = 2, infoBoxOutput("experiment_3_info_card", width='12')),
                column(card.column.width, infoBoxOutput("experiment_4_info_card", width='12'))
              )
      ),
      
      tabItem(tabName = 'data_mgmt_tab',
              useShinyalert(),
              
              h5(div(HTML("<b> Title </b>"))),
              fluidRow(
                column(12, div(HTML("Some text ")),
                       hr()
              )),
              DT::dataTableOutput("some_table")
              
      ),
      
      tabItem(tabName = 'data_mgmt_new_experiment',
              useShinyalert(),
              
              h2(div(HTML("<b> New Lattice </b>"))),
              fluidRow(
                column(12, 
                       sliderInput("lattice_dim", "Size:", min = 2, max = 10, value = 4),
                       visNetworkOutput("lattice_display", height = "400px",width = "400px"),
                       hr()
              )),
              
              fluidRow(
                column(12, 
                      h3(div(HTML("<b> Toric Code Model: </b>"))),
                      uiOutput('toric_code_model_latex'),
                       hr()
              ))           

      ),
      
      tabItem(tabName = 'visualize_tab',
        useShinyalert(),
        h5(div(HTML("<b>Visualize</b>"))),

        fluidRow(
          column(12, div(HTML(" Foo bar baz ")),
           hr()
        )),
        DT::dataTableOutput("some_table_2")
              
      )
      
      )
  )
)