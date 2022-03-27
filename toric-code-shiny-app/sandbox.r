
lattice.n.cols <- 3
lattice.n.rows <- 3
g <- make_lattice( c(lattice.n.rows,lattice.n.cols) )
layout_on_grid(g)

layout.coords <- g %>% igraph::layout_on_grid(width = lattice.n.cols, height = lattice.n.rows)
n.edges <- gsize(g)
n.nvertices <- gorder(g)

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

      
      
A = matrix( rep(1, lattice.n.rows*lattice.n.cols), nrow=lattice.n.rows, ncol=lattice.n.cols, byrow = TRUE) 

dim.A <- dim(A)

g2 <- make_lattice( c(3,3,3) )
layout_on_grid(g2, dim = 3)

## Not run: 
plot(g, layout=layout_on_grid)

#rgl package
rglplot(g, layout=layout_on_grid(g, dim = 3))