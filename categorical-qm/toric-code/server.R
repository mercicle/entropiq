
source('load-libraries.r')
source("r-helpers.r")
source("ui.R")

latex_toric_code <- function(in_lattice){

  n.rows <- 
    
  latex.string <- 'The Hamiltonian $$H_s(s) = -\\frac{1}{2} T(s) \\sum_{i=1}^{N} \\sigma_i^x + L(s) H_p(s)$$'
}

shinyServer(function(input, output, session) {

  addResourcePath(prefix = 'www', directoryPath = paste0(getwd(), '/www'))

  reactive_lattice_dim <- reactiveValues(df = data.frame())

  observeEvent( (input$lattice_dim), {
    print(paste("rows/columns: ", input$lattice_dim))
    reactive_lattice_dim$df <- data.frame(n_rows = input$lattice_dim, n_cols=input$lattice_dim )
  })
  
  output$lattice_display <- renderVisNetwork({
    
    lattice.dim.df <- reactive_lattice_dim$df
    lattice.n.rows <- lattice.dim.df$n_rows
    lattice.n.cols <- lattice.dim.df$n_cols
    
    if(nrow(lattice.dim.df)>0){
    
      g <- make_lattice( c(lattice.n.rows,lattice.n.cols) )
      layout.coords <- g %>% igraph::layout_on_grid(width = lattice.n.cols, height = lattice.n.rows)
      
      edges.df <- igraph::as_data_frame(g,what="edges")
      nodes.df <- data.frame(id = unique(c(edges.df$from, edges.df$to)))
      nodes.df <- nodes.df %>% mutate(label = id)
      edges.df <- edges.df %>% mutate(smooth=FALSE, width = 2)
      
      visNetwork(nodes.df, edges.df, main = "Lattice Model") %>% 
        visEdges(color = list(color = "grey", highlight = "red")) %>%
        visNodes(color = list(background = "black",border="white", highlight = "red"), size = 6, shadow = list(enabled = TRUE, size = 8)) %>%
        visOptions(highlightNearest = TRUE, 
                   nodesIdSelection = TRUE) %>%
        visLegend(main = "Lattice Encoding Legend",
                  position = "right", 
                  width = 0.25) %>%
        visNetwork::visIgraphLayout(layout = "layout.norm", layoutMatrix = layout.coords)
      
    }else{
      visNetwork(data.frame(id=NA,color='white'), data.frame(from=NA, to=NA)) %>% visNodes(size = 0)
    }
    
  })
  
   output$toric_code_model_latex <- renderUI({
    withMathJax(
      helpText('$$H = -A_{v} \\sum_{+} \\prod_{+} \\sigma_z - B_{p} \\sum_{\\square} \\prod_{\\square} \\sigma_x$$'))
  })  
  

})
