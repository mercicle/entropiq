
g <- make_lattice( c(3,3) )
layout_on_grid(g)

g2 <- make_lattice( c(3,3,3) )
layout_on_grid(g2, dim = 3)

## Not run: 
plot(g, layout=layout_on_grid)

#rgl package
rglplot(g, layout=layout_on_grid(g, dim = 3))