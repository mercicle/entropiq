
lattice.n.cols <- 3
lattice.n.rows <- 3
g <- make_lattice( c(lattice.n.rows,lattice.n.cols) )
layout_on_grid(g)

layout.coords <- g %>% igraph::layout_on_grid(width = lattice.n.cols, height = lattice.n.rows)

A = matrix( rep(1, lattice.n.rows*lattice.n.cols), nrow=lattice.n.rows, ncol=lattice.n.cols, byrow = TRUE) 

dim.A <- dim(A)

g2 <- make_lattice( c(3,3,3) )
layout_on_grid(g2, dim = 3)

## Not run: 
plot(g, layout=layout_on_grid)

#rgl package
rglplot(g, layout=layout_on_grid(g, dim = 3))