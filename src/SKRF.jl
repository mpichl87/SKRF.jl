#__precompile__()
module SKRF

import Base: string, print, show, +, -, *, /, ^, //, ==, !=, ∩
export Network, Frequency, plot_s_smith, connect, connect_s

using PyCall
using PyPlot
@pyimport skrf as rf
#
# from __init__.py:
#
# ## Shorthand Names
# F = Frequency
# M = Media
# N = Network
# NS = NetworkSet
# C = Calibration
# lat = load_all_touchstones
# saf  = save_all_figs
#
#########################################################################################################################################
#
#   U T I L 	(skrf.util) 		http://scikit-rf.readthedocs.org/en/latest/reference/util.html
#
#	Holds utility functions that are general conveniences.
#
#########################################################################################################################################
# returns a unique sortable string, representing the current time
now_string() = rf.now_string()
# Converts the output of  :func:`now_string` to a datetime object.
now_string_2_dt( ns::AbstractString ) = rf.now_string_2_dt( ns )
# find nearest value in array.
find_nearest( array::Array{ Number }, value::Number ) = rf.find_nearest( array, value )
# find index of nearest value in array.
find_nearest_index( array::Array{ Number }, value::Number ) = rf.find_nearest( array, value )
# Returns a file object, given a filename or file object
get_fid( file, args...; kwargs... ) = rf.get_fid( file, args...; kwargs... )
# Get the extension from a filename.
get_extn( filename::AbstractString ) = rf.get_extn( filename )
# gets the basename and strips extension
basename_noext( filename::AbstractString ) = rf.basename_noext( filename )
# Returns output 'git describe', executed in a module's root directory.
git_version( modname::AbstractString ) = rf.git_version( modname )
# loads the rc-params from the specified file (file must be located in skrf/data)
stylely( rc_dict::Dict = Dict(), style_file = "skrf.mplstyle" ) =
	rf.stylely( rc_dict, style_file )
# Turns a dictionary of structured keys to a record array of objects 		<-	don't work in python
# dict_2_recarray( d::AbstracString, delim::AbstractString, dtype ) =
# 	rf.dict_2_recarray( d, delim, dtype )

    
    
#########################################################################################################################################
#
#   P L O T T I N G 	(skrf.plotting) 		http://scikit-rf.readthedocs.org/en/latest/reference/plotting.html
#
#	This module provides general plotting functions.
#
#########################################################################################################################################
# Plots and Charts
##################################
# plots the smith chart of a given radius
smith( 	smithR::Number = 1, chart_type::AbstractString = "z", draw_labels::Bool = false, 
		border::Bool = false, ax = nothing ) =
	rf.smith( 	smithR, chart_type, draw_labels, border, ax )

# plot complex data on smith chart
plot_smith( s::Array{} ) = rf.plot_smith( s, 1.0 )



# , smith_r::Number = 1, chart_type::AbstractString = "z", 
# 			x_label::AbstractString = "Real", y_label::AbstractString = "Imaginary", 
# 			title::AbstractString = "Complex Plane", show_legend::Bool = true, 
# 			axis::AbstractString  = "equal2", ax = nothing, force_chart::Bool = false, 
# 			draw_vswr = nothing, args...; kwargs... ) =
# 	rf.plot_smith( s, smith_r, chart_type, x_label, y_label, title, show_legend, axis, 
# 					ax, force_chart, draw_vswr, args...; kwargs... )
# plots rectangular data and optionally label axes.
# plot_rectangular(x, y[, x_label, y_label, ...])
# plots polar data on a polar plot and optionally label axes.
# plot_polar(theta, r[, x_label, y_label, ...])
# plot complex data on the complex plane
# plot_complex_rectangular(z[, x_label, ...])
# plot complex data in polar format.
# plot_complex_polar(z[, x_label, y_label, ...])
##################################
# Misc Functions
##################################
# Save all open Figures to disk.
# save_all_figs([dir, format, replace_spaces, ...])
# adds markers to existing lings on a plot
# add_markers_to_lines([ax, marker_list, ...])
# turn off the legend for a given axes.
# legend_off([ax])
# runs a function after making all open figures current.
# func_on_all_figs(func, *args, **kwargs)
# scrapes a legend with redundant labels
# scrape_legend([n, ax])
#########################################################################################################################################
#
#   F R E Q U E N C Y 	(skrf.frequency) 		http://scikit-rf.readthedocs.org/en/latest/reference/frequency.html
#
#	Provides a frequency object and related functions.
#	Most of the functionality is provided as methods and properties of the Frequency Class.
#
#########################################################################################################################################
# Frequency "Class"
##########################################
type Frequency
	o::PyObject
end
##########################################
# Frequency Constructor
########################################## 
Frequency( pstart = 0, stop = 0, npoints = 0, unit = "ghz", sweep_type = "lin" ) =
	Frequency( rf.Frequency( pstart, stop, npoints, unit, sweep_type ) )
# Construct Frequency object from a frequency vector.
Frequency( a::Vector, args...; kwargs... ) =
	Frequency( rf.Frequency[ :from_f ]( a, args...; kwargs... ) )
##########################################
# Frequency Attributes
##########################################
for attribute in [  "center", "center_idx", "center_scaled", "df", "df_scaled", "dw", 
	"f_scaled", "multiplier", "multiplier_dict", "span", "span_scaled", 
	"start", "start_scaled", "step", "step_scaled", "stop", "stop_scaled", "t", "t_ns",
	"unit_dict", "w" ]
		@eval $(symbol( attribute ))( f::Frequency ) = f.o[ symbol( $attribute ) ]
end
##########################################
# Frequency Properties
##########################################
for (       fset,       fget,   property    ) in [
		(   :f!,        :f,     	"f"			),
		(   :npoints!,  :npoints,   "npoints"	),
		(   :unit!,     :unit,  	"unit"		)
	]
	@eval ( $fset )( f::Frequency, val ) 	= f.o[ symbol( $property ) ] = val
	@eval ( $fget )( f::Frequency ) 		= f.o[ symbol( $property ) ]
end
##########################################
# Frequency Methods
##########################################
for method in [ "copy" ]
	@eval $( symbol( method ) )( f::Frequency ) = Frequency( f.o[ symbol( $method ) ]() )
end
for method in [ "overlap" ]
	@eval $( symbol( method ) )( f1::Frequency, f2::Frequency ) = Frequency( f1.o[ symbol( $method ) ]( f2.o ) )
end
# Round off frequency values to a specfied precision.
round_to!( f::Frequency, val::AbstractString = "hz" ) =	
	f.o[ :round_to ]( val )
##########################################
# Frequency Class Methods
##########################################
# Label the x-axis of a plot.
labelXAxis( f::Frequency, ax = nothing ) = 
	f.o[ :labelXAxis ]( ax )
# Plot something vs this frequency
plot( f::Frequency, y, args...; kwargs... ) = 
	f.o[ :plot ]( y, args...; kwargs... )
##########################################
# Frequency Operators
##########################################
for ( operator,     pymethod        ) in [
	( :(==),        "__eq__"       ),
	( :!=,          "__ne__"       ),
	( :∩,          	"overlap"      ),
	# ( :*,           "__mul__"       ) don't work in python
	# ( :/,           "__div__"       ) don't work in python
	]
	@eval $operator( f1::Frequency, f2::Frequency  ) = Frequency( f1.o[ $pymethod ]( f2.o ) )
end
# TODO length = "__len__" op

#########################################################################################################################################
#
#   N E T W O R k 	(skrf.network)		http://scikit-rf.readthedocs.org/en/latest/reference/network.html
#
#	Provides a n-port network class and associated functions.
#	Most of the functionality in this module is provided as methods and properties of the Network Class.
#
#########################################################################################################################################
# http://scikit-rf.readthedocs.org/en/latest/reference/generated/skrf.network.Network.html#id2
##########################################
# Network "Class"
##########################################
type Network
	o::PyObject
end
##########################################
# Network Constructor
##########################################
function Network( file = nothing, name = nothing, 
	comments = nothing, f_unit = nothing; kwargs... )
		Network( rf.Network( file, name, comments, f_unit; kwargs... ) )
end
# function Network( frequency::Frequency, name = nothing, 
# 	comments = nothing, f_unit = nothing; kwargs... )
# 		Network( rf.Network( nothing, name, comments, f_unit; frequency = frequency.o, kwargs... ) )
# end

##########################################
# Network Properties
##########################################
for (       fset,       fget,   property    ) in [
		(   :s!,        :s,     "s"         )
		(   :z0!,       :z0,    "z0"        )
		(   :f!,        :f,     "f"         )
		(   nothing,    :s_re,  "s_re"      )
		(   nothing,    :s_im,  "s_im"      )
		(   nothing,    :s_mag, "s_mag"     )
		(   nothing,    :s_db,  "s_db"      )
		(   nothing,    :s_deg, "s_deg"     )
		(   nothing,    :s_gd,  "s_gd"      )
	]
	if fset != nothing
		@eval ( $fset )( nw::Network, val ) = nw.o[ symbol( $property ) ] = val
	end
	@eval ( $fget )( nw::Network ) = nw.o[ symbol( $property ) ]
end
##########################################
# Network Operators
##########################################
for ( operator,     pymethod        ) in [
	( :+,           "__add__"       )
	( :-,           "__sub__"       )
	( :*,           "__mul__"       )
	( :/,           "__div__"       )
	( :^,           "__pow__"       )
	( ://,          "__floordiv__"  )
	]
	@eval $operator( nw1::Network, nw2::Network  ) = Network( nw1.o[ $pymethod ]( nw2.o ) )
end
##########################################
# Network Attributes 
##########################################
for attribute in [  "a", "a_arcl", "a_arcl_unwrap", "a_db", "a_db10", "a_deg", "a_deg_unwrap", "a_gd", 
	"a_im", "a_mag", "a_rad", "a_rad_unwrap", "a_re", "a_time", "a_time_db", "a_time_mag", "a_vswr", 
	"f", "group_delay", "nports", "number_of_ports", "passivity", "port_tuples", "reciprocity", "reciprocity2", 
	"s", "s_arcl", "s_arcl_unwrap", "s_db", "s_db10", "s_deg", "s_deg_unwrap", "s_gd", "s_im", "s_mag", "s_rad", 
	"s_rad_unwrap", "s_re", "s_time", "s_time_db", "s_time_mag", "s_vswr", "sa", "t", "y", "y_arcl", 
	"y_arcl_unwrap", "y_db", "y_db10", "y_deg", "y_deg_unwrap", "y_gd", "y_im", "y_mag", "y_rad", 
	"y_rad_unwrap", "y_re", "y_time", "y_time_db", "y_time_mag", "y_vswr", "z", "z0", "z_arcl", 
	"z_arcl_unwrap", "z_db", "z_db10", "z_deg", "z_deg_unwrap", "z_gd", "z_im", "z_mag", "z_rad", 
	"z_rad_unwrap", "z_re", "z_time", "z_time_db", "z_time_mag", "z_vswr"   ]
	@eval $( symbol( attribute ) )( nw::Network ) = nw.o[ symbol( $attribute ) ]
end

for attribute in [  "inv", "s11", "s12", "s13", "s21", "s22", "s23", "s31", "s32", "s33" ]
	@eval $( symbol( attribute ) )( nw::Network ) = Network( nw.o[ symbol( $attribute ) ] )
end

# Returns a list of tuples, for each port index pair Ports start with 1
port_tuples( nw::Network ) =  nw.o[ :port_tuples ]
# frequency information for the network.
frequency( nw::Network ) = Frequency( nw.o[ :frequency ] )
##########################################
# Network Methods 
##########################################
# adds a complex zero-mean gaussian white-noise
add_noise_polar!( nw::Network, mag_dev, phase_dev; kwargs... ) = 
	nw.o[ :add_noise_polar ](  mag_dev, phase_dev; kwargs... )
# adds a flatband complex zero-mean gaussian white-noise signal of
add_noise_polar_flatband!( nw::Network, mag_dev, phase_dev; kwargs... ) =
	nw.o[ :add_noise_polar_flatband ](  mag_dev, phase_dev; kwargs... )
# Copies the contents of another Network into self
copy_from!( nw1::Network, nw2::Network ) = nw1.o[ :copy_from ]( nw2.o )
# Crop Network based on start and stop frequencies.
crop!( nw::Network, f_start, f_stop ) = nw.o[ :crop ]( f_start, f_stop )
# returns a cropped network, leaves self alone.
cropped( nw::Network, f_start, f_stop ) = Network( nw.o[ :cropped ]( f_start, f_stop ) )
# swaps the ports of a two port Network
flip!( nw::Network ) = nw.o[ :flip ]()
# returns a flipped network, leaves self alone.
flipped( nw::Network ) = Network( nw.o[ :flipped ]() )
#
# Applies a function parameter matrix, one frequency slice at a time
# TODO: python funcs needed.
# func_on_parameter 
#
# Transform network from generalized mixed mode parameters [1] to single ended parameters
# This is not fully tested, and should be considered as experimental
gmm2se!( nw::Network, p::Int ) = 
	nw.o[ :gmm2se ]( p )
gmm2se!( nw::Network, p::Int, z0_se::Array ) = 
	nw.o[ :gmm2se ]( p, z0_se )
# Interpolate a Network allong frequency axis
interpolate( nw::Network, freq::Frequency,  basis::AbstractString = "s", coords::AbstractString = "cart"; kwargs... ) =
	Network( nw.o[ :interpolate ]( freq.o,  basis, coords; kwargs... ) )  
interpolate( nw::Network, n::Int,           basis::AbstractString = "s", coords::AbstractString = "cart"; kwargs... ) = 
	Network( nw.o[ :interpolate ]( n,       basis, coords; kwargs... ) )
interpolate( nw::Network, fs::Vector,       basis::AbstractString = "s", coords::AbstractString = "cart", f_kwargs::Dict = Dict{}; kwargs... ) = 
	Network( nw.o[ :interpolate ]( fs,      basis, coords, f_kwargs; kwargs... ) )
# Interpolate network based on a new number of frequency points
interpolate!( nw::Network, npoints::Int; kwargs... ) = 
	nw.o[ :interpolate_self_npoints ]( npoints; kwargs... )
# Interpolates s-parameters given a new Frequency’ object
interpolate!( nw::Network, freq::Frequency; kwargs... ) = 
	nw.o[ :interpolate_self ]( freq.o; kwargs... )
# Alias
resample! = interpolate!
# multiplys a complex bivariate gaussian white-noise signal
multiply_noise!( nw::Network, mag_dev, phase_dev; kwargs... ) =
	nw.o[ :multiply_noise ]( mag_dev, phase_dev; kwargs... )

for method in [ "is_lossless", "is_passive", "is_reciprocal", "is_symmetric" ]
	@eval $( symbol( method ) )( nw::Network ) = nw.o[ symbol( $method ) ]() 
end

# Normalized non-reciprocity metric.
nonreciprocity( nw::Network, m::Int, n::Int, normalize::Bool = false ) =
	nw.o[ :nonreciprocity ]( m, n, normalize; kwargs... )
# Perturb s-parameters by small amount.
nudge!( nw::Network, amount::Number = 1e-12 ) = nw.o[ :nudge ]( amount )
# plot somthing vs frequency
plot( nw::Network, args...; kw... ) = nw.o[ :plot ]( args...; kw... )


const parameter = ( "a", "s", "y", "z" )

for p in parameter
	for (       method,                 attribute,          y_label             ) in [
			(   "plot_$( p )_arcl",         "$( p )_arcl",          "Arc Length"        ),
			(   "plot_$( p )_arcl_unwrap",  "$( p )_arcl_unwrap",   "Arc Length"        ),
			(   "plot_$( p )_db",           "$( p )_db",            "Magnitude (dB)"    ),
			(   "plot_$( p )_db10",         "$( p )_db10",          "Magnitude (dB)"    ),
			(   "plot_$( p )_deg",          "$( p )_deg",           "Phase (deg)"       ),
			(   "plot_$( p )_deg_unwrap",   "$( p )_deg_unwrap",    "Phase (deg)"       ),
			(   "plot_$( p )_gd",           "$( p )_gd",            "Group Delay (s)"   ),
			(   "plot_$( p )_im",           "$( p )_im",            "Imag Part"         ),
			(   "plot_$( p )_mag",          "$( p )_mag",           "Magnitude"         ),
			(   "plot_$( p )_rad",          "$( p )_rad",           "Phase (rad)"       ),
			(   "plot_$( p )_rad_unwrap",   "$( p )_rad_unwrap",    "Phase (rad)"       ),
			(   "plot_$( p )_re",           "$( p )_re",            "Real Part"         ),
			(   "plot_$( p )_time",         "$( p )_time",          "Time (real)"       ),
			(   "plot_$( p )_time_db",		"$( p )_time_db",		"Magnitude (dB)"	),
			(   "plot_$( p )_time_mag",		"$( p )_time_mag",		"Magnitude"			),
			(   "plot_$( p )_vswr",			"$( p )_vswr",			"VSWR"				)	]
		@eval function $( symbol( method ) )( nw::Network, 
			m = nothing, n = nothing, ax = nothing, show_legend::Bool = true, 
			attribute::AbstractString = $attribute, y_label::AbstractString = $y_label, 
			args...; kwargs... )
			nw.o[ symbol( $method ) ](
				m, n, ax, show_legend, 
				attribute, y_label,
				args...; kwargs... )
		end
	end

	for (       method,                 name   ) in [
			(   "plot_$( p )_complex",      "$( p )"         ),
			(   "plot_$( p )_polar",        "$( p )"         )
		]
		@eval function $( symbol( method ) )( nw::Network, m = nothing, n = nothing, ax = nothing, show_legend::Bool = true, 
			prop_name::AbstractString = $name, 			args...; kwargs... )
				nw.o[ symbol( $method ) ]( m, n, ax, show_legend, prop_name, args...; kwargs... )
		end
	end
end

# Plots dB, deg, smith, and complex in subplots
plot_it_all( nw::Network, args...; kw... ) = nw.o[ :plot_it_all ]( args...; kw... )
# Windowed version of plot_s_time_db
plot_s_db_time( nw::Network, args...; kw... ) = nw.o[ :plot_s_db_time ]( args...; kw... )

# Plot dB(diag(passivity metric)) vs frequency
plot_passivity( nw::Network, port = nothing, label_prefix = nothing, args...; kw... ) =
	nw.o[ :plot_passivity ]( port, label_prefix, args...; kw... )
# Plot reciprocity metric
plot_reciprocity( nw::Network, db::Bool = false, args...; kw... ) =
	nw.o[ :plot_reciprocity ]( db, args...; kw... )
# Plot reciprocity metric #2 this is distance of the determinant of the wave-cascading matrix from unity.
plot_reciprocity2( nw::Network, db::Bool = false, args...; kw... ) =
	nw.o[ :plot_reciprocity2 ]( db, args...; kw... )

# plot complex s-parameters on smith chart
# function plot_s_smith( nw::Network, 
# 	m = nothing, n = nothing, r = 1, ax = nothing, 
# 	show_legend = true, chart_type = "z", draw_labels = false, 
# 	label_axes = false, args...; kwargs... )
# 		nw.o[ :plot_s_smith ](
# 			m, n, r, ax, show_legend, chart_type, 
# 			draw_labels, label_axes, args...; kwargs... )
# end

plot_s_smith( nw::Network, args...; kwargs... ) = nw.o[ :plot_s_smith ]( args...; kwargs... )

for method in [ "copy" ]
	@eval $( symbol( method ) )( nw::Network ) = Network( nw.o[ symbol( $method ) ] )
end

# Read a Network from a ‘ntwk’ file
read!( nw::Network, args...; kw... ) =
	nw.o[ :read ]( args...; kw... )
# loads values from a touchstone file.
read_touchstone!( nw::Network, filename::AbstractString ) = 
		nw.o[ :read_touchstone ]( filename )
# Renormalize s-parameter matrix given a new port impedances
renormalize!( nw::Network, z_new::Array, powerwave::Bool = false ) = 
		nw.o[ :renormalize ]( z_new, powerwave )
renormalize!( nw::Network, z_new::Int, powerwave::Bool = false ) = 
		nw.o[ :renormalize ]( z_new, powerwave )
# renumbers ports of a Network  TODO: uses reified python indizies ( -1 = end )
renumber!( nw::Network, from_ports, to_ports ) = 
		nw.o[ :renumber ]( from_ports, to_ports )
# Transform network from single ended parameters to generalized mixed mode parameters
se2gmm!( nw::Network, p::Int ) = 
	nw.o[ :se2gmm ]( p )
se2gmm!( nw::Network, p::Int, z0_mm::Array ) = 
	nw.o[ :se2gmm ]( p, z0_mm )

time_gate( 	nw::Network, start = nothing, stop = nothing, center = nothing, span = nothing, 
			mode::AbstractString = "bandpass", window = ( "kaiser", 6 ) ) =
	Network( nw.o[ :time_gate ]( start, stop, center, span, mode, window ) )
# Convert attributes of a Network to a pandas DataFrame
to_dataframe( nw::Network, args...; kw... ) =
	nw.o[ :to_dataframe ]( args...; kw... )
# Return a windowed version of s-matrix. Used in time-domain analysis.
windowed( nw::Network, window = ( "kaiser", 6 ), normalize = true) =
	Network( nw.o[ :windowed ]( window, normalize ) )
# Write the Network to disk using the pickle module. (Python)
write( nw::Network, file = nothing, args...; kw... ) =
	nw.o[ :write ]( file, args...; kw... )
# Write contents of network to a spreadsheet, for your boss to use.
write_spreadsheet( nw::Network, args...; kw... ) =
	nw.o[ :write_spreadsheet ]( args...; kw... )
# Write a contents of the Network to a touchstone file.
write_touchstone( ::Network, filename = nothing, dir = nothing, write_z0 = false, skrf_comment = true, form = "ri" ) =
	nw.o[ :write_spreadsheet ]( filename, dir, write_z0, skrf_comment, form )		
##########################################
# Network "Class" Methods 
##########################################
read( ::Type{ Network }, args...; kw... ) =
	Network( rf.read( args...; kw... ) )
##########################################
# Network Functions
##########################################
#
# Connecting Networks
#
connect( A::Array, k::Int, B::Array, l::Int ) =
	rf.connect_s( A, k, B, l )
# connect two n-port networks together.
connect( nw1::Network, k::Int,  nw2::Network, l::Int, num::Int = 1 ) =
	Network( rf.connect( nw1.o, k, nw2.o, l, num ) )
# connect two ports of a single n-port network’s s-matrix.
innerconnect( A::Array, k::Int, l::Int ) =
	rf.innerconnect_s( A, k, l )
# connect ports of a single n-port network.
innerconnect( nw1::Network, k::Int, l::Int, num::Int = 1 ) =
	Network( rf.innerconnect( nw1.o, k, l, num ) )	
# Cascade two 2-port Networks together
cascade( nw1::Network, nw2::Network ) =
	Network( rf.cascade( nw1.o, nw2.o ) )
# cascade a list of 2-port networks
cascade( nws::Vector{ Network } ) =
	Network( rf.cascade_list( map( x -> x.o, nws ) ) )
# De-embed ntwkA from ntwkB.
de_embed( nw1::Network, nw2::Network ) =
	Network( rf.de_embed( nw1.o, nw2.o ) )
# invert the ports of a ( [fx]2x2 ) networks s-matrix, ‘flipping’ it over
flip( A::Array ) =
	rf.flip( A )
#
# Interpolation and Concatenation Along Frequency Axis
#
# Stitches ntwkA and ntwkB together.
stitch( nw1::Network, nw2::Network; kw... ) =
	Network( rf.stitch( nw1.o, nw2.o; kw... ) )
# Returns the overlapping parts of two Networks, interpolating if needed.
overlap( nw1::Network, nw2::Network ) =
	Network( rf.overlap( nw1.o, nw2.o ) )
#
# Combining Networks
#
# Builds a N-port Network from list of N one-ports
n_oneports_2_nport( nws::Vector{ Network }, args...; kw... ) =
	Network( rf.n_oneports_2_nport( map( x -> x.o, nws ) ), args...; kw... )
# Builds a 2-port Network from list of four 1-ports
four_oneports_2_twoport( nw1::Network, nw2::Network, nw3::Network, nw4::Network, args...; kw... ) =
	Network( rf.four_oneports_2_twoport( nw1.o, nw2.o, nw3.o, nw4.o, args... ; kw... ) )
# Creates 3-port from three 2-port Networks
three_twoports_2_threeport( nws::Vector{ Network }, auto_order::Bool = true, args...; kw... ) =
	Network( rf.three_twoports_2_threeport( map( x -> x.o, nws ) ), auto_order, args...; kw... )
# Builds a N-port Network from list of two-ports
n_twoports_2_nport( nws::Vector{ Network }, nports::Int, offby::Int = 1, ; kw... ) =
	Network( rf.n_twoports_2_nport( map( x -> x.o, nws ) ), nports, offby; kw... )


pow( ntwk1, ntwk2 ) = Network( ntwk1.o[ :__pow__ ]( ntwk2.o ))
 
end