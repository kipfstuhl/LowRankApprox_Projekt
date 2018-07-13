
# Functions for plotting
# 
# improting Gadfly is terribly slow, so avoid this whenever it is
# possible; here do it in another file

using Gadfly
function plot_sing_vals(svs::Array{Array{Float64,1},1})
    num = length(svs)
    plts = Array{Any, 1}(num)
    for i in 1:length(svs)
        plts[i]=plot(y=svs[i], x=collect(1:length(svs[i])),
                     Geom.point, Scale.x_discrete, #Scale.y_continuous,
                     Scale.y_log10,
                     # Coord.cartesian(fixed=true),
                     # Geom.subplot_grid(free_y_axis=true),
                     Guide.title(string("Mode ", i)));
    end
    # set_default_plot_size(30cm, 6cm)
    hstack(plts...)
end
