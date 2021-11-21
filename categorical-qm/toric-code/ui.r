source('load-libraries.r')

ui <- dashboardPage(skin = "black",
  dashboardHeader(title = "Toric Code"),
  dashboardSidebar(),
  dashboardBody(
    # Boxes need to be put in a row (or column)
    fluidRow(
      box(plotOutput("distPlot", height = 250)),

      box(
        title = "Controls",
        sliderInput("slider", "Number of observations:", 1, 100, 50)
      )
    )
  )
)