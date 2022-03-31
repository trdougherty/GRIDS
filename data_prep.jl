### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 49d1ea64-9535-11ec-2eab-119fadb9083e
begin
	using DataFrames
	using GeoDataFrames
	using CSV
	using Plots
	using Gadfly
	using Statistics
	using Chain
	using ArchGDAL
	const AG = ArchGDAL
	using Rasters
	using GMT
	using VoronoiCells
	using GeometryBasics
	using Dates
	using StatsBase
	using PlutoUI
	using GeoFormatTypes
	using GeoJSON
	using GeoStats
	using Missings
	using LazySets
	using ProgressMeter
	using ColorSchemes: flag
	
	color_scheme = cgrad(:matter, 9, categorical = true)
end;

# ╔═╡ 800a75e5-4f56-49e9-8d77-d62cce02ebad
md"""
# Data Prep for Dense Building Representations (DBRs)
"""

# ╔═╡ 8bb0c644-ce7b-49d8-8107-fd473f281199
md"""
This file runs the general trajectory for processing historical weather data from NOAA to combine with historical energy consummption data on the monthly scale. As it stands, almost all of the functionality is compressed into this single notebook. Fantastic next steps would be:
1. Functionalize a lot of the behavior into separate files
2. Utilize Julia's multiple dispatch to provide multiple training scenarios, like annual consumption
3. Fix the monthly cleaning a bit so some months of regions aren't dropped
4. Add command like argument parsing to handle new data sources with similar formats
5. Drop the graphing dependencies for council footprints (it's custom for New York) potentially again using multiple dispatch
"""

# ╔═╡ c8ed3ef3-eb49-45c3-933d-6935d8ea14eb
# this cell is used to control the intensity of the program when trying things out
begin
	toyset = true
	if (length(ARGS) > 0 && ARGS[1] == "full")
		toyset = false
		println("Running full datset at time: ", Dates.now())
	end
end

# ╔═╡ e2e418ad-9fcf-4c8b-b20f-383efe07f094
md"""
All of the filename paths:
"""

# ╔═╡ e2ed76ac-9b7d-4e9c-9b14-48b4feb9edeb
begin
	data_directory = joinpath(".","data","nyc")
	weather_directory = joinpath(data_directory, "weather")
	footprints_directory = joinpath(data_directory, "footprints")
	energy_directory = joinpath(data_directory, "energy")
	photos_directory = joinpath(data_directory, "photos")

	weather_path = joinpath(
		weather_directory,
		"2020.csv"
	)
	council_footprints_path = joinpath(
		footprints_directory, 
		"council_districts.geojson"
	)
	building_footprints_path = joinpath(
		footprints_directory, 
		"building_footprints.geojson"
	)
	monthly_energy_path = joinpath(
		energy_directory, 
		"monthly",
		"Local_Law_84_2021__Monthly_Data_for_Calendar_Year_2020_.csv"
	)
	annual_energy_path = joinpath(
		energy_directory, 
		"annually", "Energy_and_Water_Data_Disclosure_for_Local_Law_84_2021__Data_for_Calendar_Year_2020_.csv"
	)

	satellite_photos_path = joinpath(
		photos_directory,
		"planet_3m_raw",
		"lowres.tif"
	)
	microsoft_building_footprints_path = joinpath(
		footprints_directory,
		"NewYorkCity.geojson"
	)
	streetview_metadata_path = joinpath(
		photos_directory, 
		"streetview",
		"manhattan_metadata_nodes.tsv"
	)

	### There are two versions of this data - one is for the toyset, which will just look at one council region (2) of New York, the other has data for the entire region. This is to reduce processing time when building algorithms, but be sure to switch it back to the full data before running the entire algorithm

	if toyset
		# Might want to do something later like change the data loaded
		# For the full footprints list of New York State
		# microsoft_building_footprints_path = joinpath(
		# 	footprints_directory,
		# 	"NewYork.geojson"
		# )
	else
		# nothing right now
	end
end;

# ╔═╡ 68884f19-bca4-40b0-8a6a-91783e9bc7ac


# ╔═╡ 9398797a-82f1-42b2-ae54-5392c3ffb437
begin
	# ESPG codes - 4326 is the global standard for Lat, Lng and the other is UTM
	source = ArchGDAL.importEPSG(4326, order=:trad)
	# dist_target = ArchGDAL.importEPSG(26918) # this is the unique UTM code for zone 18
	# target = ArchGDAL.importEPSG(2955) 
	target = ArchGDAL.importEPSG(3857) # this is the web mercator but meters based
	# target = ArchGDAL.importEPSG(26918)
end;

# ╔═╡ 5e8216f3-8a86-4d6e-8444-11ba63785c1a
begin
function reproject_points!(
	geom_obj::ArchGDAL.IGeometry,
	source::ArchGDAL.ISpatialRef,
	target::ArchGDAL.ISpatialRef)

	ArchGDAL.createcoordtrans(source, target) do transform
		ArchGDAL.transform!(geom_obj, transform)
	end
    return geom_obj
end

function reproject_points!(
	geom_obj::Union{
		Vector{ArchGDAL.IGeometry},
		Vector{ArchGDAL.IGeometry{ArchGDAL.wkbPoint}},
		Vector{ArchGDAL.IGeometry{ArchGDAL.wkbMultiPolygon}},
		Vector{ArchGDAL.IGeometry{ArchGDAL.wkbPolygon}}
	},
	source::ArchGDAL.ISpatialRef,
	target::ArchGDAL.ISpatialRef)

	AG.createcoordtrans(source, target) do transform
		AG.transform!.(geom_obj, Ref(transform))
	end
    return geom_obj
end
end

# ╔═╡ 1d6bd5a3-0b3d-4870-b5bb-7a39f6494394
unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))

# ╔═╡ 8ef3ef24-6ab4-4e55-a6bc-6ee4ac901a53
md"""
## 1. Weather Data Loading
"""

# ╔═╡ 4d822b90-5b31-4bb3-954b-71675e6c1dd9
begin
	weather = CSV.File(weather_path) |> DataFrame
	unique_stations = DataFrames.combine(DataFrames.groupby(weather, :STATION), first)
	locations = ArchGDAL.createpoint.(unique_stations.LONGITUDE,unique_stations.LATITUDE)

	# want it in web mercator
	reproject_points!(locations, source, target)
	
	locations_tuple = [(ArchGDAL.getx(location, 0), ArchGDAL.gety(location, 0)) for location in locations];
	inv_locations_tuple = [(ArchGDAL.gety(location, 0), ArchGDAL.getx(location, 0)) for location in locations];

	council_footprints = GeoDataFrames.read(council_footprints_path)
	council_footprints.coun_dist = Base.parse.(Int64, council_footprints.coun_dist)
	council_footprints.shape_area = Base.parse.(Float64, council_footprints.shape_area)
	council_footprints.shape_leng = Base.parse.(Float64, council_footprints.shape_leng)
	council = council_footprints

	reproject_points!(council.geom, source, target)
	weather
end

# ╔═╡ 6d921f17-a3b7-469d-9df2-6d02022fcc33


# ╔═╡ bf92bd9a-eb71-468e-b7c2-1b5980dd0a5c
md"""
A general idea of how the weather stations from NOAA are scattered around New York City"""

# ╔═╡ 08d4a328-141c-4869-88c7-09ea8ff57590
begin
council_plot = Plots.plot(
	council.geom, 
	color="transparent", 
	dpi=400,
	size=(600,600)
)
council_points_plot = Plots.plot(
	council_plot,
	locations, 
	color=color_scheme[2], 
	alpha=0.5,
	# xlims=(-74.5,-73.25),
	# ylims=(40.18, 41.2)
)
end

# ╔═╡ 464a6802-c5eb-4a42-b9f4-d0dde12e24e8


# ╔═╡ 5157cf2e-5b1c-447c-9d59-9064ecf9a771
md"""
## 2. Weather Data Preprocessing
"""

# ╔═╡ 31f66cb8-afa3-4ebc-9d7d-7d220997ba1f
md"""
Next want to build a bounding box around new york to group the different regions into "weather zones"
"""

# ╔═╡ 8594d3ad-0e04-49d3-86e1-cdc0c8f8951c
begin
	points_poly = ArchGDAL.createpolygon(inv_locations_tuple)
	# below I'm adding a small buffer layer around, 0.02 was arbitrary and may suffer from compression errors in high latitude regions - something to think about
	points_box = ArchGDAL.buffer(ArchGDAL.boundingbox(points_poly), 500)
	Plots.plot(council_points_plot, points_box, color=color_scheme[1], alpha=0.15)
end

# ╔═╡ 0d9ba991-3d5a-4a15-84fd-a9641323e0d6
begin
	buffer_radius = 800 # meters
	geom_locations = GeometryBasics.Point2.(locations_tuple)
	lons, lats = unzip(locations_tuple)
	(lat_min, lon_min), (lat_max, lon_max) = extrema(locations_tuple)
	lat_min = minimum(lats) - buffer_radius
	lon_min = minimum(lons) - buffer_radius
	lat_max = maximum(lats) + buffer_radius
	lon_max = maximum(lons) + buffer_radius
	geom_rect = Rectangle(
		GeometryBasics.Point2(lon_min, lat_min),
		GeometryBasics.Point2(lon_max, lat_max)
	)
end;

# ╔═╡ 1b9654c4-ea9a-4dcd-a610-6295732d96b3
geom_locations

# ╔═╡ 3a92bcdc-9c40-4fea-8955-f011b41a66c0
geom_rect

# ╔═╡ 9d4de2c5-c0d2-468d-b694-c678c6961857
md"""
Breaking the region into zones reveals how the weather data might interact with New York City
"""

# ╔═╡ 20886cf8-b48d-4293-b756-a894279d11f8
begin
	tess = voronoicells(geom_locations, geom_rect)
	Plots.plot(council_points_plot, tess, color=color_scheme[end-3], dpi=400)
end

# ╔═╡ e9df25ae-f0e5-4634-8b44-f40576c7fc87
md"""
So if we don't run any compression, these are the likely weather groups. But not all of the stations have good monthly data, so we need to figure out which ones are reporting well, use average numbers, and break this into larger higher-quality groups
"""

# ╔═╡ a0b152a2-6189-459a-9288-514c8b3d14f5
begin
	weather[!, "date"] = Date.(Month.(weather.DATE), Year.(weather.DATE));
	weather_description = describe(weather, :nmissing)
	weather_description_variables = weather_description.variable
	weather_description_nmissing = weather_description.nmissing
	DataFrame(vars=weather_description_variables, percent_missing=weather_description_nmissing / nrow(weather))
end

# ╔═╡ e38363af-1059-4c7e-bc2b-94924f34b01b
md"""
The amount of missing data accross the board is pretty horrendous, so I'm going to just take a smattering of the best data we have and use that as average values across the board. It lools like temperature and precipitation are kind of reliable
"""

# ╔═╡ 9d65dabb-d4d2-49c1-96f2-1f5a02bdbecf
md"""
#### Narrowed by regularly reliable reporting
"""

# ╔═╡ 4608787a-e817-4ae4-bfd7-018445586632
begin
	# maybe going to try and keep these relatively constant... to avoid variable length systems. Something to think about!
	reliable_terms = [
		:date, 
		:STATION,
		:TMAX,
		:TMIN,
		:PRCP,
		:LATITUDE,
		:LONGITUDE,
		:ELEVATION
	]
	weather_reliable = @chain weather begin
		# dropmissing(reliable_terms)
		select(reliable_terms)
	end
	weather_select_stations = @chain weather_reliable begin
		# select(Not([:simple_date]))
		DataFrames.groupby([:date, :STATION])
		DataFrames.combine(_, valuecols(_) .=> mean, renamecols=false)
		dropmissing
	end
end

# ╔═╡ 8dc094fc-f1be-4640-b97e-94ed3e87e6a6
md"""
So this is essentially how we narrow the pool into weather stations which seem reliable enough. If they can report good data throughout the year then they make the cut
"""

# ╔═╡ f05d157c-f2a8-421f-af61-64461b82d863
begin
	unique_stations_select = DataFrames.combine(DataFrames.groupby(weather_select_stations, :STATION), first)
	locations_select = ArchGDAL.createpoint.(unique_stations_select.LONGITUDE,unique_stations_select.LATITUDE)
	reproject_points!(locations_select, source, target)
	locations_tuple_select = [(ArchGDAL.getx(location, 0), ArchGDAL.gety(location, 0)) for location in locations_select];
end;

# ╔═╡ 58e457b8-ade1-43d2-8717-ea750e84a3a1
md"""
The difference between all of the available stations and the stations with good accuracy in reporting
"""

# ╔═╡ 143beaa8-65cb-48fd-a981-92e1543c123b
begin
Plots.plot(council_plot, locations, color=color_scheme[2], alpha=0.3, markersize=3)
Plots.plot!(locations_select, color=color_scheme[end-2], alpha=0.5)
end

# ╔═╡ 6eb80555-636d-4072-b8a2-984a09f7f833
md"""
now to break the region again into weather regions
"""

# ╔═╡ c281ed6a-7adb-489c-afde-292b0cf76485
begin
	geom_locations_select = GeometryBasics.Point2.(locations_tuple_select)
	lons_select, lats_select = unzip(locations_tuple_select)
	lat_min_select = minimum(lats_select) - buffer_radius
	lon_min_select = minimum(lats_select) - buffer_radius
	lat_max_select = maximum(lats_select) + buffer_radius
	lon_max_select = maximum(lats_select) + buffer_radius
	geom_rect_select = Rectangle(
		GeometryBasics.Point2(lon_min_select, lat_min_select),
		GeometryBasics.Point2(lon_max_select, lat_max_select)
	)
	tess_select = voronoicells(geom_locations_select, geom_rect)
end;

# ╔═╡ 9de45182-910c-4451-824a-0c08568c5fd0
begin
# Plots.plot(council.geom, color="transparent", dpi=400)
# Plots.plot!(locations, color=color_scheme[2], alpha=0.3, markersize=3)
Plots.plot(council_points_plot, tess_select, color=color_scheme[end-3], alpha=0.9, markersize=3)
Plots.plot!(locations_select, color=color_scheme[end-2], alpha=0.5)
end

# ╔═╡ 25a32915-7c5e-4e7b-bb94-d1571120dbaa


# ╔═╡ a96bb753-6b33-4bba-9591-aaf007432c0a
md"""
##### Here we have the _cells_ which define the weather zones
"""

# ╔═╡ b485d1e4-c249-47ad-8e45-707756675fdb
begin
function weather_cells(cells)
	regional_polygons::Vector{Vector{Tuple{Float64, Float64}}} = []
	for cell in cells
		cell_coords::Vector{Tuple{Float64, Float64}} = []
		for el in cell
			data_tuple::Tuple{Float64, Float64} = el.data
			push!(cell_coords, data_tuple)
		end
		push!(cell_coords, cell[1].data)
		push!(regional_polygons, cell_coords)
	end
	regional_polygon_geom = ArchGDAL.createpolygon.(regional_polygons)
end
regional_polygon_geom = weather_cells(tess_select.Cells)
unique_stations_select[!,"geom"] = regional_polygon_geom
end;

# ╔═╡ e013cb41-819d-4b8a-8381-0784167b401a
md"""
now this needs to link up with the locations of the weather stations so we're looking at the right things. So going to check for each weather station which region it's in. As a bit of a sanity check after that nonsense, now I'm going to make sure each of the weather stations fall within the geometry of the region.
"""

# ╔═╡ ffaa5a0d-d4eb-42a6-b817-f3571300257c
begin
	found_in = []
	for (index, location) in enumerate(locations_select)
		found_bool = GeoDataFrames.contains(unique_stations_select.geom[index], location)
		push!(found_in, found_bool)
	end
	found_in
end

# ╔═╡ 6b123eec-f8e0-46df-86d1-06c7b4c3cc89
md"""
The result: a dataframe with cleaned monthly average weather for each of the most reliable zones in New York City, as well as the region of coverage for the entire city.
"""

# ╔═╡ 90638da7-7b62-43ec-89d9-770acf945c71
begin
service_areas = select(unique_stations_select, [:STATION, :geom])
geoweather = leftjoin(
	weather_select_stations, 
	service_areas, 
	on="STATION", 
	matchmissing=:notequal
)
end

# ╔═╡ bfef8e4e-0335-4a32-b23b-801a79e0c269


# ╔═╡ d117da6c-bd22-45ed-aff2-baff497858a0
md"""
## 2. Building Locations (Footprints) Data
"""

# ╔═╡ e9627745-e40d-41e4-b569-4ed79f5686c4
md"""
Now that the weather cleaning is more or less finished, time to move to the building data.
1. Building Footprints Data
2. Load Energy Data
3. Combine Energy and Footprints
"""

# ╔═╡ ee5a17b7-d9d0-410f-8196-9a3d9d2010d1
n_sample = 500

# ╔═╡ 666fdd7b-8062-45d9-822b-bedeca682576
# we might want some feature in here which limits the age of the nyc footprint

# ╔═╡ 80339671-3cc9-42cc-858a-64ae9bd9b5a8
raw_footprints = GeoDataFrames.read(building_footprints_path)

# ╔═╡ c14e276e-4b3c-4e67-b5a0-5b19d7a21180
unique(raw_footprints.geomsource)

# ╔═╡ b49eb729-7d16-4165-8c80-e901cbbf3b01
md"""
##### Building Footprints
"""

# ╔═╡ e36d5b6c-9a95-4570-9b4a-d3bf4f48137f
begin
	footprints = GeoDataFrames.read(building_footprints_path)
	rename!(footprints, "base_bbl" => "bbl")
	filter!(x -> tryparse(Float64, x.heightroof) !== nothing, footprints)
	filter!(x -> tryparse(Float64, x.groundelev) !== nothing, footprints)
	footprints[!,:heightroof] = Base.parse.(Float64, footprints.heightroof)
	footprints[!,:groundelev] = Base.parse.(Float64, footprints.groundelev)
	select!(footprints,
	[
		:geom,
		:bbl,
		:heightroof,
		:cnstrct_yr,
		:groundelev,
		:bin
	])
	dropmissing!(footprints, [:bbl,:bin])
	reproject_points!(footprints.geom, source, target)
	footprints[!,:geom_point] = [ArchGDAL.centroid(x) for x in footprints.geom]
	footprints
end

# ╔═╡ 97efb80c-0915-42df-8199-6df2639d2d41
begin
Plots.plot(council_plot)
Plots.plot!(
	unique(footprints.geom_point[1:n_sample]), 
	color=color_scheme[4],
	alpha=1,
	markersize=3,
	dpi=400
)
end

# ╔═╡ 223a20c9-6a8e-4014-a909-1e4e8525d282


# ╔═╡ da770866-0a26-4722-98a8-cf7965e35d15
md"""
## 3. Energy Data
"""

# ╔═╡ ebd20a55-ae35-463f-ad81-fb5dc35c8b41
md"""
##### Monthly Energy Data - LL84
"""

# ╔═╡ a5be3813-81c3-4a08-85ad-cb43f317bf60
begin
	monthly_file = CSV.File(monthly_energy_path, missingstring="Not Available")
	nyc_monthly_energy_2020 = DataFrame(monthly_file)
	filter!(x -> x["Parent Property Name"] == "Not Applicable: Standalone Property", 	nyc_monthly_energy_2020)
	dropmissing!(nyc_monthly_energy_2020)
	select!(nyc_monthly_energy_2020,
	[
		"Property Id",
		"Month",
		"Electricity Use  (kBtu)",
		"Natural Gas Use  (kBtu)"
	])
	rename!(nyc_monthly_energy_2020, "Electricity Use  (kBtu)" => 	"electricity_kbtu")
	rename!(nyc_monthly_energy_2020, "Natural Gas Use  (kBtu)" => "gas_kbtu")
	rename!(nyc_monthly_energy_2020, "Month" => "month")
end

# ╔═╡ 13046794-2b45-43f2-a4dd-484173181109
md"""
##### Combine Monthly Energy Data and Footprints
"""

# ╔═╡ 3a6fc1ba-95fa-446c-97be-36346ff562a6
md"""
The annualized energy consumption has information for the **BBL**, **BIN**, and **PropertyID** of each building, which we can use to link between the energy data and the footprints. Below, I'm loading the full energy consumption and just keeping the linking information
"""

# ╔═╡ 9f8807ca-0ec6-43f2-83b5-8a76f8feebd5
begin
	annual_file = CSV.File(annual_energy_path, missingstring="Not Available")
	nyc_annual_energy_2020 = DataFrame(annual_file)
	# Only use single buildings
	filter!(x -> x["Number of Buildings"] < 2, nyc_annual_energy_2020)
	# Only use entries where the property Id is unique to each building
	unique!(nyc_annual_energy_2020, "Property Id")
	rename!(nyc_annual_energy_2020, "NYC Borough, Block and Lot (BBL)" => "bbl")
	rename!(nyc_annual_energy_2020, "NYC Building Identification Number (BIN)" => "bin")
	select!(nyc_annual_energy_2020, 
	[
		"Property Id",
		"bbl",
		"bin"
	])
	first(nyc_annual_energy_2020, 5)
end

# ╔═╡ a28efb16-2f08-4c69-9851-153ba4c6bf1a
md"""
Now to link the monthly energy consumption data with the footprints using the annual energy as a intermediate
"""

# ╔═╡ ccfc42a6-a122-4ddb-b5a2-f7cd7101789e
begin
	# This first one is to get a BBL from the annual data which we can match to footprints
	nyc_monthly_energy_2020_joined = leftjoin(nyc_monthly_energy_2020,
		nyc_annual_energy_2020, on="Property Id", makeunique=true)
	dropmissing!(nyc_monthly_energy_2020_joined, [:bbl,:bin])
	# This second join is to actually match with the footprints
	nyc_geomonthly_2020 = leftjoin(nyc_monthly_energy_2020_joined,
		footprints, on="bbl", makeunique=true)
	dropmissing!(nyc_geomonthly_2020, [:geom_point])
	select(nyc_geomonthly_2020, :geom_point, :)
end

# ╔═╡ cc6f26f6-3db6-41b4-9e71-b6d48592c410
md"""
##### The buildings which are reporting monthly statistics
"""

# ╔═╡ 6b205e8b-f212-4668-a31e-c5af409c6e93
begin
Plots.plot(
	council_plot,
	unique(nyc_geomonthly_2020.geom_point)[1:n_sample],
	color=color_scheme[end-2],
	alpha=0.6,
	dpi=500,
	markersize=2.5,
)
end

# ╔═╡ caec5908-a4c6-4bc0-8af3-6e098892e24b
md"""
Finally, cleaning up the weather data a bit so that we have daily average statistics for each month. Energy data:
"""

# ╔═╡ c3d1cadb-c3d3-4452-aa8a-1f50cdee1448
begin
	month_lookup = Dict{String, Integer}(
		"Jan" => 1,
		"Feb" => 2,
		"Mar" => 3,
		"Apr" => 4,
		"May" => 5,
		"Jun" => 6,
		"Jul" => 7,
		"Aug" => 8,
		"Sep" => 9,
		"Oct" => 10,
		"Nov" => 11,
		"Dec" => 12
	);
	energy = @chain nyc_geomonthly_2020 begin
		subset!(:cnstrct_yr => ByRow(!=("")))
		transform!(:cnstrct_yr => (x -> Base.parse.(Int64, x)) => :cnstrct_yr)
		rename!(:month => :date_string)
		select!(Not(:bin_1))
		transform!(:date_string => (x -> split.(x,"-")) => [:month,:year])
		transform!(:year => ByRow(x->*("20",x)) => :year)
		transform!(:year => (x -> Base.parse.(Int64, x)) => :year)
		transform!(:month => ByRow(x->month_lookup[x]) => :month)
		transform!([:month, :year] => ByRow((x,y)->Date(Dates.Month.(x),Dates.Year.(y))) => :date)
		transform!(:date => ByRow(Dates.daysinmonth) => :month_days)
		transform!([:electricity_kbtu,:month_days] => ByRow((x,y)->x/y) => :daily_electric)
		transform!([:gas_kbtu,:month_days] => ByRow((x,y)->x/y) => :daily_gas)
		select!([:daily_electric,:daily_gas,:date],:)
		rename!("Property Id" => :id)
	end
end

# ╔═╡ dc647053-fdef-49c1-98e7-6292b9ac55c7
md"""
We can also view some of the summary statistics about the monthly energy data
"""

# ╔═╡ 1f34e819-9eef-4157-8175-c90c8f92882c
Gadfly.plot(
	energy[1:n_sample,:],
	x=:daily_gas,
	y=:daily_electric,
	Geom.point,
	color=:date,
	# Geom.abline(color="red", style=:dash),
	Theme(
		# default_color="white",
		# discrete_highlight_color=c->"black",
		point_size=0.6mm,
		alphas=[0.5]
	),
	Gadfly.Scale.x_log(),
	Gadfly.Scale.y_log(),
	Guide.title("Log Relationship between Electricity and Gas"),
	Guide.xlabel("Daily Gas"),
	Guide.ylabel("Daily Electricity")
)

# ╔═╡ b7cd9153-1353-4051-9197-d8137602d3fe
md"""
## 4. Streetview Data
"""

# ╔═╡ f7b688df-5589-4bdd-b431-94e770bc8a62
md"""
So we have monthly energy consumption for thousands of buildings in New York City, and we want to now capture high resolution satellite imagery for each of these buildings. We have streetview imagery for a large region of the city as well, it would be nice to put that in perspective
"""

# ╔═╡ a3efc796-7604-4260-96d1-9b1c3cd60c0d
md"""
##### Metadata for each of the panorama:
"""

# ╔═╡ 7f8897ad-e30f-48f0-94e2-aa8fbac7954e
begin
	original_streetview_set = CSV.read(streetview_metadata_path, DataFrame);
	streetview = @chain original_streetview_set begin
	dropmissing(["pano_id", "coords.lat", "coords.lng"])
	end
	
	streetview_points = [ 
		ArchGDAL.createpoint(
			streetview[i, "coords.lng"],
			streetview[i, "coords.lat"]
		) for i in 1:nrow(streetview)
	];
	
	reproject_points!(streetview_points, source, target)
	first(streetview, 3)
end

# ╔═╡ 7eb37472-8ee6-4c9b-bf27-eff4905b4e2e
md"""
In case the same dataframe above doesn't show enough information, this is a list of the columns provided:
"""

# ╔═╡ ec822073-7e2a-44b7-8739-0e094a673066
names(streetview[1,:])

# ╔═╡ 501e8ada-010a-4fae-bb2d-855baa9cf923
md"""
Now I want to show where the streetview images are distributed throughout Manhattan:
"""

# ╔═╡ 65ff833e-ca70-408a-992a-128108401dc8
manhattan_councils = council[council.coun_dist .< 11, :];

# ╔═╡ 568fe19d-4a33-4e4f-b3eb-d50aaeef6eed
begin
manhattan_plot = Plots.plot(manhattan_councils.geom, color="transparent", dpi=500)
streetview_coverage = Plots.plot!(manhattan_plot, streetview_points[1:n_sample], markersize=3, color=color_scheme[5], alpha=0.4)
end

# ╔═╡ d915b138-e81d-4bca-941e-a9f15f56914d
md"""
Just for fun, we can also visualize different metadata associated with each of the street level images, like the hight of the roll of the vehicle when taking the photo
	"""

# ╔═╡ 1d2b44be-6ece-4ec0-9142-ea480095b5ac
begin
set_default_plot_size(18cm, 9.5cm)
alt_plot = Gadfly.plot(
	streetview[1:n_sample, :],
	x=Symbol("coords.lng"),
	y=Symbol("coords.lat"),
	color=:alt,
	Guide.xlabel("Longitude"), 
	Guide.ylabel("Latitude"), 
	Guide.title("Altitude Map"),
	size=[1.7pt],
	Theme(alphas=[0.8])
)
roll_plot = Gadfly.plot(
	streetview[1:n_sample, :],
	x=Symbol("coords.lng"),
	y=Symbol("coords.lat"),
	color=:roll_deg,
	Guide.xlabel("Longitude"), 
	Guide.ylabel("Latitude"), 
	Guide.title("Roll Degree Map"),
	size=[1.7pt],
	Theme(alphas=[0.8])
)

hstack(alt_plot, roll_plot)
end

# ╔═╡ f7128f8a-77f1-4584-9cd6-2660ebfb084b
md"""
## 5. Satellite Region of Interest
"""

# ╔═╡ 85716a1c-626a-4ac7-a0d9-37cf0f658f79
md"""
This will be curated by building a box around all of the points we have in the building energy dataset
"""

# ╔═╡ 3e737332-78f9-4a93-931c-3b4bd2a1c195
begin
	unique_energy = energy[.!nonunique(energy, :id), :]
	hull_list = [ [ArchGDAL.getx(p, 0), ArchGDAL.gety(p,0)] for p in unique(unique_energy.geom_point) ];
	nyc_convex_hull = convex_hull(hull_list);
	insert!(nyc_convex_hull, length(nyc_convex_hull)+1, nyc_convex_hull[1]);
	hull_polygon = ArchGDAL.createpolygon(nyc_convex_hull)
	pointslist = [ ArchGDAL.createpoint(v...) for v in hull_list ]
end

# ╔═╡ 3216b2cb-976e-40cd-aee4-69af2ce5cb31
begin
	hp = Plots.plot(hull_polygon, color=color_scheme[1], alpha=0.2)
	Plots.plot(hp, pointslist[1:n_sample], color=color_scheme[4], alpha=0.6)
end

# ╔═╡ 4bc5a61d-12e2-4c55-a9fa-3af21cf55f7e
md"""
The satellite region will need to cover all of the building data, but streetview is less significant and disregarded in curating the region
"""

# ╔═╡ 0c400d13-dc44-4bdd-b627-2ad551109cf9
begin
	hull_buffer = ArchGDAL.buffer(hull_polygon, 3e3)	
	# a = Plots.plot(council_footprints.geom, color="transparent")
	d = Plots.plot(hull_buffer, color=color_scheme[1], alpha=0.2, dpi=800)
	f = Plots.plot!(hull_polygon, color=color_scheme[2], alpha=0.4)
	a = Plots.plot!(f, council_footprints.geom, color="transparent")
	g = Plots.plot!(a, pointslist[1:n_sample], markersize=3, alpha=0.7, color=color_scheme[3])
	Plots.plot!(g, streetview_points[1:100], markersize=3, alpha=0.7, color=color_scheme[5])
	# Plots.plot!(g, pointslist[1:400], markersize=3, alpha=0.7, color="darkblue")
end

# ╔═╡ c4aa4c0a-6ad0-418d-8c76-56d2a840905c


# ╔═╡ 73c36e85-ad6f-42e2-a5c0-5fc10c8db7ce
md"""
## 6. Assignment - builidings to weather zones
"""

# ╔═╡ 54754fac-ceab-47e3-ac6b-d8f837078072
md"""
So we have a huge number of buildings, we now need to check which weather zone each building is within.
"""

# ╔═╡ 75bbca6f-6410-4a93-854c-3ea030f94217
begin
	unique_energyid_df = DataFrames.combine(DataFrames.groupby(energy, :id), first)
	building_service_mapping = Dict()
	for energy_id in eachrow(unique_energyid_df)
		for service_area in eachrow(service_areas)
			if GeoDataFrames.contains(service_area.geom, energy_id.geom_point)
				building_service_mapping[energy_id.id] = service_area.STATION
				@goto next_energy_id
			end
		end
		@label next_energy_id
	end
	energy[!,:STATION] = [ building_service_mapping[id] for id in energy[!,:id]]
	first(select(energy, [:STATION, :id], :), 3)
end

# ╔═╡ 85e138e8-fd2f-4239-9b41-2aa6cb9d8ed2
begin
	unique_small_energy = unique(energy, :id)
	small_energy = unique_small_energy[StatsBase.shuffle(1:nrow(unique_small_energy))[1:2400], :]
	
	unique_colors = cgrad(:matter, 5, categorical = true);
	service_color_map = Dict()
	for (index,zone) in enumerate(unique(small_energy.STATION))
		service_color_map[zone] = unique_colors[index]
	end

	small_energy[!,:service_zone_color] = [ service_color_map[id] for id in small_energy[!,:STATION]];
	first(small_energy, 3)
end;

# ╔═╡ bacb917d-8d9a-4081-8039-966849ade9d6
begin
	weather_membership_plot = Plots.plot(
		small_energy.geom_point,
		palette=small_energy.service_zone_color,
		markersize=3.4,
		markerstrokewidth=0.3,
		alpha=0.7,
		dpi=400
	)
	Plots.plot!(
		weather_membership_plot,
		council.geom,
		color="transparent",
		markerstrokealpha=0.1
	)
end

# ╔═╡ e1bc6d5a-99c8-4d83-aad6-491b5ededfd0
md"""
Finally, we have a cleaned dataset where each building has daily statistics for energy consumption and is linked to the average monthly weather statistics for the region
	"""

# ╔═╡ 1cc473f5-6ac6-4876-af22-987768f2ad16
begin
energy_climate = @chain energy begin
	DataFrames.leftjoin(
		_, 
		geoweather, 
		on = [:STATION, :date],
		makeunique=true 
	)
	select([
		:geom_point,
		:id,
		:daily_electric,
		:daily_gas,
		:heightroof,
		:cnstrct_yr,
		:groundelev,
		:month,
		:year,
		:month_days,
		:TMAX,
		:TMIN,
		:PRCP
	])
end
end

# ╔═╡ d8512b16-84f5-4b0a-a7de-3bffb3367242


# ╔═╡ 05d62c11-9bbf-4b4d-bdf0-b43685e33f1c
md"""
## 7. Assigning the Microsoft building footprints to each building
"""

# ╔═╡ 56ac5f27-682a-42f7-99ea-7882f29506fb
md"""
Microsoft foorprints overview - **TODO:** find a way to first sort this by the start of the date range so that we're using the most up to date geometries first - *note* I think that might have been accomplished enough with the "sort" by release date
"""

# ╔═╡ 18cbefa1-d451-47a7-9e75-b00bbfa2f78a
# begin
# 	microsoft_footprints = GeoDataFrames.read(microsoft_building_footprints_path)
# 	microsoft_new_york_city = @chain microsoft_footprints begin
# 		sort(:release, rev=true)
# 		filter(x -> GeoDataFrames.contains(hull_buffer, x.geom), _)
# 	end
# 	GeoDataFrames.write("./data/nyc/footprints/NewYorkCity.geojson", microsoft_new_york_city)
# end

# ╔═╡ 56683a3f-cab1-478a-a4fa-25c7a5df3ce6
md"""
Checking out what some of the metadata looks like for the microsoft set
"""

# ╔═╡ ee1f546d-2755-48e2-b18c-17e7d1c55298
begin
	microsoft_footprints = GeoDataFrames.read(microsoft_building_footprints_path)
	microsoft_footprints = @chain microsoft_footprints begin
		sort(:release, rev=true)
	end
	reproject_points!(microsoft_footprints.geom, source, target)
	unique(microsoft_footprints, :capture_dates_range)
end

# ╔═╡ 4718a5e0-c28e-4ea2-8e3c-d651740ea344
md"""
We can also just look at the select region of interest for our testing set to see what those buildings might look like
"""

# ╔═╡ 596d95ef-8c84-4e3e-a5a5-bde5c50518b7
test_district = 2

# ╔═╡ 98eb52ed-d4df-4e20-8ca2-3ae7b50e1999
begin
	test_district
	selected_council_footprint = council_footprints[council_footprints.coun_dist .== test_district, :].geom[1]

	if toyset == true
		@chain microsoft_footprints begin
			filter!(x -> GeoDataFrames.contains(selected_council_footprint, x.geom), _)
		end
	end
end;

# ╔═╡ 3566b173-ae10-4b65-83a0-d040e802d181
begin
test_district
Plots.plot(
	microsoft_footprints.geom, 
	color="transparent",
	size=(700,700),
	dpi=500
)
end

# ╔═╡ a87ee29b-a9bb-472d-a9a3-51224e0a1bdd
md"""
Next step is to parse through every building in the dataset to see where it falls in the footprints dataset. Some assumptions I'm making:
1. If the building lat, lng doesn't fall in a footprint, then it will be dropped from the dataset
2. If multiple building IDs collide into the same building footprint, then all of their data will be dropped from the dataset

First step though is to get a unique list of buildings in our energy/climate dataset which is reporting
"""

# ╔═╡ 990b6d78-4650-4665-af61-22fc53fb50b7
begin
	test_district
	buildings_with_data = @chain energy_climate begin
		DataFrames.groupby(:id)
		DataFrames.combine(first)
		select([:id, :geom_point])
	end
	first(buildings_with_data, 3)
end

# ╔═╡ d61157fc-b565-4e61-9d55-4199745c5b05
microsoft_footprints

# ╔═╡ f62813e2-fed3-4b53-a201-bfe9e7f0bfc0
begin
	test_district # this is just to trigger this cell when district is changed
	building_footprint_match = []
	for (building_index, building_point) = enumerate(eachrow(buildings_with_data))
		for (footprint_index,footprint) in enumerate(eachrow(microsoft_footprints))
			if GeoDataFrames.contains(footprint.geom, building_point.geom_point)
				push!(building_footprint_match, footprint_index)
				@goto end_building_logic
			end
		end
		push!(building_footprint_match, missing)
		@label end_building_logic
	end
end	

# ╔═╡ 084c08df-f535-496d-b578-2a83f7b9082b
md"""
This building footprint takes the buildings with data and matches them to a footprint from the microsoft dataset. If no match, we get "missing"
"""

# ╔═╡ 66b45f66-285c-47eb-971b-5593156d3809
buildings_footprint_fullindex = hcat(buildings_with_data, building_footprint_match)

# ╔═╡ 0c6264c1-9dad-4b27-8a52-02123880c4cb
md"""
If we drop missing buildings and buildings which match to two microsoft building footprints, then we're left with only the buildings which have data, linked to a microsoft footprints dataset
"""

# ╔═╡ 1dd01920-0da2-406c-9bd2-10409b38321c
buildings_footprint = @chain buildings_footprint_fullindex begin
	dropmissing(_, :x1)
	unique(_, :x1)
end

# ╔═╡ 813feb02-4c08-482d-a033-56531f0e1d1b
begin
	# this should work because we just dropped the missing terms, so all that's left should be index values from the for loop
	test_district
	buildings_footprint[!,:footprint] = microsoft_footprints.geom[convert.(Int64, buildings_footprint.x1)];
end;

# ╔═╡ afeba502-4a7e-4f7a-b1bb-d4f617763ca5
md"""
We can also plot them to make sure it's all working as intended
"""

# ╔═╡ cc045caa-5293-47a8-bbc5-f174ab86cb2b
begin
	test_district
	Plots.plot(buildings_footprint.footprint[1:50], color="transparent", size=(500,500), dpi=500)
	Plots.plot!(buildings_footprint.geom_point[1:50], color="black", markersize=2.5)
end

# ╔═╡ cd5ccc7d-3113-4db3-bf64-1f0cb8a1e249


# ╔═╡ 11330b51-1bbe-4a7f-b398-1dda6848a669
md"""
## 8. Look for streetview images
"""

# ╔═╡ 23b2ae8e-0149-4df8-b23f-6d48460e779a
md"""
As a first pass, going to just draw a bubble around each building geometry. Now, we want to be more accurate with how far we project bubbles around each building in terms of meters. To do this accurately, I'm going to first convert all of the data into UTM, which lets us evenly project bubbles in terms of meters
"""

# ╔═╡ 3498d6ee-56f8-402d-8917-2fa2aace76f9
m_samples = 100

# ╔═╡ c1f54fd5-404d-4fef-b140-130baccd1d57
border_radius = 55 #in meters

# ╔═╡ 8c07ba91-7f04-46dd-9f30-41df91ee36e2
md"""
We project a little radius around each building which will be our *net*
"""

# ╔═╡ 0d42b241-e7f6-488a-9524-4d3612ba4f2a
begin
border_radius # should be in meters
buildings_footprint[!,:footprint_bubble] = 
	GeoDataFrames.buffer.(
		buildings_footprint.footprint,
		border_radius
	)

# Now we can visualize the results a bit
Plots.plot(
	buildings_footprint.footprint_bubble[1:m_samples], 
	color=color_scheme[1], 
	alpha=0.2,
	size=(500,500),
	dpi=500
)
Plots.plot!(
	buildings_footprint.footprint[1:m_samples], 
	color="transparent"
)
Plots.plot!(
	buildings_footprint.geom_point[1:m_samples], 
	color="indianred", 
	markersize=2.1
)
end

# ╔═╡ 46adf87b-7d4c-4c0f-897d-631fec46cab5
md"""
And we now have this new column in the dataset to play with when looking for streetview photos
	"""

# ╔═╡ b42f75aa-06fb-449e-b9e2-01146d37b218
buildings_footprint.footprint_bubble

# ╔═╡ acd39a29-e8b9-405a-b37d-fe6b17d210dd
md"""
Recall we have two data pieces we can use for the streetview
1. streetview: A DataFrame with the streetview metadata
2. streetview_points: A list of the streetview points, encoded as ArchGDAL points
"""

# ╔═╡ 502364d2-456a-4140-bbf9-70b9dc7363d8
length(streetview_points)

# ╔═╡ d94975d9-bb0b-4363-ad73-18f40f64fef2
nrow(streetview)

# ╔═╡ 5c76d3ce-913d-4dd4-875c-c8f693b0d337
md"""
So for every building, I'm going to check which points the bubble contains
"""

# ╔═╡ 2b7cc18f-8f6c-4163-8da0-3774bf853a4f
begin
	test_district
	border_radius
	function streetview_map(streetview_points, footprint_bubbles)
		building_streetview_map = []
		building_streetview_panos::Vector{Vector{String}} = []
		for building_bubble in footprint_bubbles
			building_streetview_points_index = []
			for (index, streetview_point) in enumerate(streetview_points)
				if GeoDataFrames.contains(building_bubble, streetview_point)
					push!(building_streetview_points_index, index)
				end
			end
			push!(
				building_streetview_map,
				building_streetview_points_index
			)
			push!(
				building_streetview_panos,
				streetview.pano_id[building_streetview_points_index]
			)
		end
	return (building_streetview_panos, building_streetview_map)
	end
	building_streetview_panos, building_streetview_map = streetview_map(streetview_points, buildings_footprint.footprint_bubble)
end

# ╔═╡ f857d45f-767a-4c91-96fd-21d944070fbd
md"""
So now we have a mapping from each building to streetview images nearby the building, by the Pano ID of the image
"""

# ╔═╡ 88fcd33a-a574-45af-8677-26511b5bdf7e
buildings_footprint.streetview_mapping = building_streetview_map

# ╔═╡ 8f94298d-533d-4252-b0e8-180808719a25
jpoint = 15

# ╔═╡ bab915c5-61c1-47f3-9211-87ecb552734d
# Now we can visualize the results a bit
begin
	border_radius
	Plots.plot(
		buildings_footprint.footprint_bubble[jpoint], 
		color=color_scheme[1], 
		alpha=0.12,
		dpi=500
	)
	Plots.plot!(
		buildings_footprint.footprint[jpoint],
		color="transparent",
		linewidth=2
	)
	Plots.plot!(
		buildings_footprint.geom_point[jpoint],
		color=color_scheme[4], 
		markersize=10,
		alpha=0.5
	)
	Plots.plot!(
		streetview_points[buildings_footprint.streetview_mapping[jpoint]], 
		color=color_scheme[end-3], 
		markersize=3,
		alpha=0.7
	)
end

# ╔═╡ 4c31b1a5-adf8-42ae-8f34-f371274c61e1
md"""
But I'm also going to take the opportunity to add the pano IDs as a column which will be more useful for indexing later
"""

# ╔═╡ 8452a50c-1f3d-4407-bfd1-b3e9e50bb5dc
buildings_footprint.streetview_panos = building_streetview_panos

# ╔═╡ 9e8476f5-f39b-4961-86a0-7efd9e8bc9e8
md"""
## 9. Splitting for training and testing
"""

# ╔═╡ e9f57d8a-e13a-472d-a567-790cab0e7c1b
begin
	council_colormap = cgrad(:matter, 2, categorical = true);
	bool_colors = Dict(
		true => "indianred",
		false => "transparent"
	)
	council_colors = map(x -> bool_colors[x], council.coun_dist .== test_district)

	Plots.plot(
		council.geom,
		palette=council_colors,
		title="Test Region",
		dpi=700,
		size=(700,700)
	)
end

# ╔═╡ 4d21be1f-a5e8-405d-9d54-5a0147a202da


# ╔═╡ cafa6163-7b7f-456b-ba28-816ad9fbefc9
md"""
Here's what the full, processed data looks like
"""

# ╔═╡ 88839489-e6ec-4ff4-9d79-cd9da526e35c
begin
buildings_footprint[:,"floorarea"] = [ 
	ArchGDAL.geomarea(x) for x in buildings_footprint.footprint 
]
buildings_footprint
end

# ╔═╡ 1ed35dc3-c604-450a-9218-a49a9b6df84b
begin
buildings_metadata = @chain buildings_footprint begin
	select([:id, :footprint, :floorarea, :streetview_panos])
end
first(buildings_metadata, 3)
end

# ╔═╡ 6837ceea-9d9e-45c5-9f18-f42784227463
md"""
And when we add in all of the energy data
"""

# ╔═╡ 68b2baf3-cc88-45ef-a346-0e731449b00b
begin
	data_full = leftjoin(
		energy_climate,
		buildings_metadata,
		on=:id
	)
	@chain data_full begin
		dropmissing!
	end
	first(data_full, 3)
end

# ╔═╡ 5bac64d2-36ed-493c-b4a5-527fa5c336be
begin
	# builds the training and test datasets
	if toyset == false
		test = filter(
			x -> GeoDataFrames.contains(selected_council_footprint, x.footprint), data_full)
		select!(test, Not(:geom_point))
		train = filter(
			x -> !GeoDataFrames.contains(selected_council_footprint, x.footprint), data_full)
		select!(train, Not(:geom_point))
	
		GeoDataFrames.write("test.geojson", test; layer_name="data", geom_column=:footprint)
		# GeoDataFrames.write("train.geojson", train; layer_name="data", geom_column=:footprint)
	end
end;

# ╔═╡ 7f427ca1-9311-4a52-936b-0f0e23fd8be7
md"""
## 10. Satellite Data!
"""

# ╔═╡ bcddd86e-28a0-4df9-836d-ba985248e889
# test_tif = joinpath(
# 	photos_directory,
# 	"planet_3m_raw",
# 	"lowres.tif"
# )

# ╔═╡ 2ca23b00-4c41-4cb6-a6af-331e12c814de
# satellite_tif = ArchGDAL.read(test_tif);

# ╔═╡ cf8d2ff2-0bac-44ef-9c2c-1e4fe118ed0b
# ArchGDAL.filelist(satellite_tif)

# ╔═╡ a6bc01d8-2dbb-453c-8dce-92cbc8c015a7
# begin
# satellite_box = ArchGDAL.boundingbox(hull_buffer)
# ArchGDAL.reproject(
# 	satellite_box,
# 	GeoFormatTypes.EPSG(source),
# 	GeoFormatTypes.EPSG(target)
# )
# end

# ╔═╡ 30c53681-703e-4c0e-9c18-01b04d232ac3
# testtif_arch = ArchGDAL.read(test_tif)

# ╔═╡ e0716bcb-45ed-4b9b-bd3a-1e48d06af32b
# ArchGDAL.getgeotransform(testtif_arch)

# ╔═╡ 44751276-26d9-421c-8096-a1bf422d6c04
# lowres_path = joinpath(
# 	photos_directory,
# 	"planet_3m_raw",
# 	"lowres.tif"
# )

# ╔═╡ 70294579-9025-4709-accd-3d0279df91c7
# lowres_tif = ArchGDAL.read(lowres_path)

# ╔═╡ 84351e0e-c118-4567-a4d7-86af12b0dfb2
# number_rasters = (ArchGDAL.nraster(lowres_tif))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ArchGDAL = "c9ce4bd3-c3d5-55b8-8973-c0e20141b8c3"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
GMT = "5752ebe1-31b9-557e-87aa-f909b540aa54"
Gadfly = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
GeoDataFrames = "62cb38b5-d8d2-4862-a48e-6a340996859f"
GeoFormatTypes = "68eda718-8dee-11e9-39e7-89f7f65f511f"
GeoJSON = "61d90e0f-e114-555e-ac52-39dfb47a3ef9"
GeoStats = "dcc97b0b-8ce5-5539-9008-bb190f959ef6"
GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
Missings = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"
Rasters = "a3a2b9e3-a471-40c9-b274-f788e487c689"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
VoronoiCells = "e3e34ffb-84e9-5012-9490-92c94d0c60a4"

[compat]
ArchGDAL = "~0.7.4"
CSV = "~0.10.3"
Chain = "~0.4.10"
ColorSchemes = "~3.17.1"
DataFrames = "~1.3.2"
GMT = "~0.41.1"
Gadfly = "~1.3.4"
GeoDataFrames = "~0.2.0"
GeoFormatTypes = "~0.3.0"
GeoJSON = "~0.5.1"
GeoStats = "~0.31.2"
GeometryBasics = "~0.4.2"
LazySets = "~1.56.1"
Missings = "~1.0.2"
Plots = "~1.27.1"
PlutoUI = "~0.7.37"
ProgressMeter = "~1.7.1"
Rasters = "~0.1.1"
StatsBase = "~0.33.16"
VoronoiCells = "~0.3.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[ArchGDAL]]
deps = ["ColorTypes", "Dates", "DiskArrays", "GDAL", "GeoFormatTypes", "GeoInterface", "ImageCore", "Tables"]
git-tree-sha1 = "245d68fd5749c0aee757da162c2956b595b274bb"
uuid = "c9ce4bd3-c3d5-55b8-8973-c0e20141b8c3"
version = "0.7.4"

[[ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "1ee88c4c76caa995a885dc2f22a5d548dfbbc0ba"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.2"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "310b77648d38c223d947ff3f50f511d08690b8d5"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.3"

[[Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "91d6baa911283650df649d0aea7c28639273ae7b"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.1+0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CFTime]]
deps = ["Dates", "Printf"]
git-tree-sha1 = "bca6cb6ee746e6485ca4535f6cc29cf3579a0f20"
uuid = "179af706-886a-5703-950a-314cd64e0468"
version = "0.1.1"

[[CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "32abd86e3c2025db5172aa182b982debed519834"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.1"

[[CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "9310d9495c1eb2e4fa1955dd478660e2ecab1fbb"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.3"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "109664d3a6f2202b1225478335ea8fea3cd8706b"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.5"

[[Chain]]
git-tree-sha1 = "339237319ef4712e6e5df7758d0bccddf5c237d9"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.10"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CircularArrays]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0598a9ea22c65bfde7f07f21485ebf60deee3302"
uuid = "7a955b69-7140-5f4e-a0ed-f168c5e2e749"
version = "1.3.0"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[CoDa]]
deps = ["AxisArrays", "Distances", "Distributions", "FillArrays", "LinearAlgebra", "Printf", "Random", "RecipesBase", "ScientificTypes", "StaticArrays", "Statistics", "StatsBase", "TableOperations", "TableTransforms", "Tables", "UnicodePlots"]
git-tree-sha1 = "3878095fcb8cc1e245ecbbea2023fed518ad55bf"
uuid = "5900dafe-f573-5c72-b367-76665857777b"
version = "0.9.2"

[[CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "9a2695195199f4f20b94898c8a8ac72609e165a4"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.3"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CoupledFields]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "6c9671364c68c1158ac2524ac881536195b7e7bc"
uuid = "7ad07ef1-bdf2-5661-9d2b-286fd4296dac"
version = "0.2.0"

[[CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "32d125af0fb8ec3f8935896122c5e345709909e5"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.0"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[DensityRatioEstimation]]
deps = ["LinearAlgebra", "Parameters", "Random", "Requires", "Statistics", "StatsBase"]
git-tree-sha1 = "12bcfc6c7599069588bf00b83ecb564d5a5eee68"
uuid = "ab46fb84-d57c-11e9-2f65-6f72e4a7229f"
version = "0.5.0"

[[Dictionaries]]
deps = ["Indexing", "Random"]
git-tree-sha1 = "7e73a524c6c282e341de2b046e481abedbabd073"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.19"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[DimensionalData]]
deps = ["Adapt", "ArrayInterface", "ConstructionBase", "Dates", "LinearAlgebra", "Random", "RecipesBase", "SparseArrays", "Statistics", "Tables"]
git-tree-sha1 = "d2777cfa0ff3da5a67544c06c84da415baa99225"
uuid = "0703355e-b756-11e9-17c0-8b28908087d0"
version = "0.19.8"

[[DiskArrays]]
git-tree-sha1 = "cfca3b5d0df57f6315b5187482ab8eae4a5beb0e"
uuid = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
version = "0.2.13"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "5a4168170ede913a2cd679e53c2123cb4b889795"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.53"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "be03cce55664d1e13aa572faee753159abe06578"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.4.0"

[[ErrorfreeArithmetic]]
git-tree-sha1 = "d6863c556f1142a061532e79f611aa46be201686"
uuid = "90fa49ef-747e-5e6f-a989-263ba693cf1a"
version = "0.5.2"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FastRounding]]
deps = ["ErrorfreeArithmetic", "LinearAlgebra"]
git-tree-sha1 = "6344aa18f654196be82e62816935225b3b9abe44"
uuid = "fa42c844-2597-5d31-933b-ebd51ab2693f"
version = "0.3.1"

[[FieldMetadata]]
git-tree-sha1 = "c279c6eab9767a3f62685e5276c850512e0a1afd"
uuid = "bf96fef3-21d2-5d20-8afa-0e7d4c32a885"
version = "0.3.1"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "80ced645013a5dbdc52cf70329399c35ce007fae"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.13.0"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "deed294cde3de20ae0b2e0355a6c4e1c6a5ceffc"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.8"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "56956d1e4c1221000b7781104c58c34019792951"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.11.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Flatten]]
deps = ["ConstructionBase", "FieldMetadata"]
git-tree-sha1 = "d3541c658c7e452fefba6c933c43842282cdfd3e"
uuid = "4c728ea3-d9ee-5c9a-9642-b6f7d7dc04fa"
version = "0.4.3"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "StaticArrays"]
git-tree-sha1 = "8e76bcd47f98ee25c8f8be4b9a1c60f48efa4f9e"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.7"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GDAL]]
deps = ["CEnum", "GDAL_jll", "NetworkOptions", "PROJ_jll"]
git-tree-sha1 = "ac36f6aabae2ab70449af913f8ed60a0aecf06a8"
uuid = "add2ef01-049f-52c4-9ee2-e494f65e021a"
version = "1.3.0"

[[GDAL_jll]]
deps = ["Artifacts", "Expat_jll", "GEOS_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "PROJ_jll", "Pkg", "SQLite_jll", "Zlib_jll", "Zstd_jll", "libgeotiff_jll"]
git-tree-sha1 = "5272b856ca84a20871858b46a1f3b529e12c38a7"
uuid = "a7073274-a066-55f0-b90d-d619367d196c"
version = "300.400.100+0"

[[GEOS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "07f6426d716d0d110cdede90ebd3bbb2f3be0d8c"
uuid = "d604d12d-fa86-5845-992e-78dc15976526"
version = "3.10.0+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "c3cc0a7a4e021620f1c0e67679acdbf1be311eb0"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.0.1"

[[GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "fe68622f32828aa92275895fdb324a85894a5b1b"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+0"

[[GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"

[[GMT]]
deps = ["Conda", "Dates", "Pkg", "Printf", "Statistics"]
git-tree-sha1 = "24fe909d5b5c454bed672c8e1bda2827e169cd08"
uuid = "5752ebe1-31b9-557e-87aa-f909b540aa54"
version = "0.41.2"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "df5f5b0450c489fe6ed59a6c0a9804159c22684d"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "83578392343a7885147726712523c39edc714956"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.1+0"

[[Gadfly]]
deps = ["Base64", "CategoricalArrays", "Colors", "Compose", "Contour", "CoupledFields", "DataAPI", "DataStructures", "Dates", "Distributions", "DocStringExtensions", "Hexagons", "IndirectArrays", "IterTools", "JSON", "Juno", "KernelDensity", "LinearAlgebra", "Loess", "Measures", "Printf", "REPL", "Random", "Requires", "Showoff", "Statistics"]
git-tree-sha1 = "13b402ae74c0558a83c02daa2f3314ddb2d515d3"
uuid = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
version = "1.3.4"

[[GeoClustering]]
deps = ["CategoricalArrays", "Clustering", "Distances", "GeoStatsBase", "LinearAlgebra", "MLJModelInterface", "Meshes", "SparseArrays", "Statistics", "TableDistances", "TableOperations", "Tables"]
git-tree-sha1 = "627e49f0243c6b112d700630096402da593499df"
uuid = "7472b188-6dde-460e-bd07-96c4bc049f7e"
version = "0.2.10"

[[GeoDataFrames]]
deps = ["ArchGDAL", "DataFrames", "GeoFormatTypes", "Pkg", "Tables"]
git-tree-sha1 = "5e7002befc1d6c134ddc66c2d74cb19a085c3bac"
uuid = "62cb38b5-d8d2-4862-a48e-6a340996859f"
version = "0.2.0"

[[GeoEstimation]]
deps = ["Distances", "GeoStatsBase", "KrigingEstimators", "LinearAlgebra", "Meshes", "NearestNeighbors", "Variography"]
git-tree-sha1 = "6b03126f84eac5192a5ea2de5401a7dc24e69341"
uuid = "a4aa24f8-9f24-4d1a-b848-66d123bfa54d"
version = "0.9.4"

[[GeoFormatTypes]]
git-tree-sha1 = "bb75ce99c9d6fb2edd8ef8ee474991cdacf12221"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.3.0"

[[GeoInterface]]
deps = ["RecipesBase"]
git-tree-sha1 = "6b1a29c757f56e0ae01a35918a2c39260e2c4b98"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "0.5.7"

[[GeoJSON]]
deps = ["GeoInterface", "JSON3"]
git-tree-sha1 = "4764da92d333658552b2bedc9f6b379f017c727b"
uuid = "61d90e0f-e114-555e-ac52-39dfb47a3ef9"
version = "0.5.1"

[[GeoLearning]]
deps = ["Distributions", "GeoStatsBase", "MLJModelInterface", "Meshes", "TableOperations", "Tables"]
git-tree-sha1 = "1899ee2d7004b259bd958aaeb39d8f586435c472"
uuid = "90c4468e-a93e-43b4-8fb5-87d804bc629f"
version = "0.1.10"

[[GeoSimulation]]
deps = ["CpuId", "Distributions", "FFTW", "GeoStatsBase", "KrigingEstimators", "LinearAlgebra", "Meshes", "Random", "SpecialFunctions", "Statistics", "Tables", "Variography"]
git-tree-sha1 = "af72ecf00106ed570a633d2293921d4ec1e09bce"
uuid = "220efe8a-9139-4e14-a4fa-f683d572f4c5"
version = "0.7.0"

[[GeoStats]]
deps = ["DensityRatioEstimation", "Distances", "GeoClustering", "GeoEstimation", "GeoLearning", "GeoSimulation", "GeoStatsBase", "KrigingEstimators", "LossFunctions", "Meshes", "PointPatterns", "Reexport", "ScientificTypes", "TableTransforms", "Variography"]
git-tree-sha1 = "0ebdf5c7e525f64aabb712e02b3166fd585258c4"
uuid = "dcc97b0b-8ce5-5539-9008-bb190f959ef6"
version = "0.31.2"

[[GeoStatsBase]]
deps = ["Combinatorics", "DensityRatioEstimation", "Distances", "Distributed", "Distributions", "LinearAlgebra", "LossFunctions", "MLJModelInterface", "Meshes", "Optim", "Parameters", "RecipesBase", "ReferenceFrameRotations", "ScientificTypes", "StaticArrays", "Statistics", "StatsBase", "TableOperations", "Tables", "Transducers", "TypedTables"]
git-tree-sha1 = "f11859648cd7694fdc2a2af929a92af85306a182"
uuid = "323cb8eb-fbf6-51c0-afd0-f8fba70507b2"
version = "0.25.1"

[[GeometricalPredicates]]
git-tree-sha1 = "527d55e28ff359029d8f72d77c0bdcaf28793079"
uuid = "fd0ad045-b25c-564e-8f9c-8ef5c5f21267"
version = "0.4.1"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HDF5]]
deps = ["Blosc", "Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "698c099c6613d7b7f151832868728f426abe698b"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.15.7"

[[HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "bab67c0d1c4662d2c4be8c6007751b0b6111de5c"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.1+0"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hexagons]]
deps = ["Test"]
git-tree-sha1 = "de4a6f9e7c4710ced6838ca906f81905f7385fd6"
uuid = "a1b4810d-1bce-5fbd-ac56-80944d57a21f"
version = "0.2.0"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b15fc0a95c564ca2e0a7ae12c1f095ca848ceb31"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.5"

[[IntervalArithmetic]]
deps = ["CRlibm", "FastRounding", "LinearAlgebra", "Markdown", "Random", "RecipesBase", "RoundingEmulator", "SetRounding", "StaticArrays"]
git-tree-sha1 = "1fa3ba0893ea5611830feedac46b7f95872cbd01"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.20.5"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "bcf640979ee55b652f3b01650444eb7bbe3ea837"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.4"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "8c1f668b24d999fb47baf80436194fdccec65ad2"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.4"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "936e7ebf6c84f0c0202b83bb22461f4ebc5c9969"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.0.0"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[KrigingEstimators]]
deps = ["Combinatorics", "GeoStatsBase", "LinearAlgebra", "Meshes", "Statistics", "Unitful", "Variography"]
git-tree-sha1 = "4ceec05cc372d6e6a8a7b0b03707020df6c08546"
uuid = "d293930c-a38c-56c5-8ebb-12008647b47a"
version = "0.8.9"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "6f14549f7760d84b2db7a9b10b88cd3cc3025730"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.14"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LazySets]]
deps = ["Distributed", "ExprTools", "GLPK", "InteractiveUtils", "IntervalArithmetic", "JuMP", "LinearAlgebra", "Pkg", "Random", "RecipesBase", "Reexport", "Requires", "SharedArrays", "SparseArrays"]
git-tree-sha1 = "ff895ba7a262692870656d93ded246cab67230ec"
uuid = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
version = "1.56.2"

[[LearnBase]]
git-tree-sha1 = "a0d90569edd490b82fdc4dc078ea54a5a800d30a"
uuid = "7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6"
version = "0.4.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "46efcea75c890e5d820e670516dc156689851722"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.4"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LossFunctions]]
deps = ["InteractiveUtils", "LearnBase", "Markdown", "RecipesBase", "StatsBase"]
git-tree-sha1 = "0f057f6ea90a84e73a8ef6eebb4dc7b5c330020f"
uuid = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
version = "0.7.2"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "74d7fb54c306af241c5f9d4816b735cb4051e125"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.4.2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[MarchingCubes]]
deps = ["StaticArrays"]
git-tree-sha1 = "5f768e0a0c3875df386be4c036f78c8bd4b1a9b6"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.2"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "09be99195f42c601f55317bd89f3c6bbaec227dc"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.1.1"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[Meshes]]
deps = ["CategoricalArrays", "CircularArrays", "Distances", "IterTools", "IteratorInterfaceExtensions", "LinearAlgebra", "NearestNeighbors", "Random", "RecipesBase", "ReferenceFrameRotations", "SimpleTraits", "SparseArrays", "SpecialFunctions", "StaticArrays", "StatsBase", "TableTraits", "Tables"]
git-tree-sha1 = "103aabf2208c62de7136ce3bc9211a99405c87a7"
uuid = "eacbb407-ea5a-433e-ab97-5258b1ca43fa"
version = "0.21.1"

[[MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "ba8c0f8732a24facba709388c74ba99dcbfdda1e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.0"

[[NCDatasets]]
deps = ["CFTime", "DataStructures", "Dates", "NetCDF_jll", "Printf"]
git-tree-sha1 = "17e39eb5bbe564f48bdbefbd103bd3f49fcfcb9b"
uuid = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
version = "0.11.9"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded92de95031d4a8c61dfb6ba9adb6f1d8016ddd"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.10"

[[NetCDF_jll]]
deps = ["Artifacts", "HDF5_jll", "JLLWrappers", "LibCURL_jll", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Pkg", "Zlib_jll", "nghttp2_jll"]
git-tree-sha1 = "598f1a5e9829b3e57f233f98b34a22b376dff373"
uuid = "7243133f-43d8-5620-bbf4-c2c921802cf3"
version = "400.702.402+0"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "bc0a748740e8bc5eeb9ea6031e6f050de1fc0ba2"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.6.2"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e8185b83b9fc56eb6456200e873ce598ebc7f262"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.7"

[[PROJ_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Libtiff_jll", "Pkg", "SQLite_jll"]
git-tree-sha1 = "59c43648bf081f732eae1e44b8883f713d2ca1b6"
uuid = "58948b4f-47e0-5654-a9ad-f609743f8632"
version = "800.200.100+0"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "5f6e1309595e95db24342e56cd4dabd2159e0b79"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.3"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[PointPatterns]]
deps = ["Distributions", "GeoStatsBase", "Meshes"]
git-tree-sha1 = "263d83efd53ec54a32f568550cea951be4942bd4"
uuid = "e61b41b6-3414-4803-863f-2b69057479eb"
version = "0.3.14"

[[PolygonInbounds]]
git-tree-sha1 = "8d50c96f4ba5e1e2fd524116b4ef97b29d5f77da"
uuid = "e4521ec6-8c1d-418e-9da2-b3bc4ae105d6"
version = "0.2.0"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "28ef6c7ce353f0b35d0df0d5930e0d072c1f5b9b"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.1"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[RasterDataSources]]
deps = ["Dates", "HTTP", "URIs", "ZipFile"]
git-tree-sha1 = "77831569cb82f3f7a5307fa3a30bcb0246f4a779"
uuid = "3cb90ccd-e1b6-4867-9617-4276c8b2ca36"
version = "0.5.3"

[[Rasters]]
deps = ["Adapt", "ArchGDAL", "ColorTypes", "ConstructionBase", "Dates", "DimensionalData", "DiskArrays", "FillArrays", "Flatten", "GeoFormatTypes", "GeoInterface", "HDF5", "Missings", "Mmap", "NCDatasets", "PolygonInbounds", "ProgressMeter", "RasterDataSources", "RecipesBase", "Reexport", "Requires", "Setfield"]
git-tree-sha1 = "67e6687a89a4f2017be7a865472b5d212f167e96"
uuid = "a3a2b9e3-a471-40c9-b274-f788e487c689"
version = "0.1.1"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[ReferenceFrameRotations]]
deps = ["Crayons", "LinearAlgebra", "Printf", "Random", "StaticArrays"]
git-tree-sha1 = "ec9bde2e30bc221e05e20fcec9a36a9c315e04a6"
uuid = "74f56ac7-18b3-5285-802d-d4bd4f104033"
version = "3.0.0"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SQLite_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "f79c1c58951ea4f5bb63bb96b99bf7f440a3f774"
uuid = "76ed43ae-9a5d-5a62-8c75-30186b810ce8"
version = "3.38.0+0"

[[ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "ba70c9a6e4c81cc3634e3e80bb8163ab5ef57eb8"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.0"

[[ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SetRounding]]
git-tree-sha1 = "d7a25e439d07a17b7cdf97eecee504c50fedf5f6"
uuid = "3cc68bcd-71a2-5612-b932-767ffbe40ab0"
version = "0.2.1"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "35efd62f6f8d9142052d9c7a84e35cd1f9d2db29"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.1"

[[SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "7f5a513baec6f122401abfc8e9c074fdac54f6c1"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "4f6ec5d99a28e1a749559ef7dd518663c5eca3d5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.3"

[[StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "271a7fea12d319f23d55b785c51f6876aadb9ac0"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.0.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "25405d7016a47cf2bd6cd91e66f4de437fd54a07"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.16"

[[StringDistances]]
deps = ["Distances", "StatsAPI"]
git-tree-sha1 = "ceeef74797d961aee825aabf71446d6aba898acb"
uuid = "88034a9c-02f8-509d-84a9-84ec65e18404"
version = "0.11.2"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d24a825a95a6d98c385001212dc9020d609f2d4f"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.8.1"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableDistances]]
deps = ["CategoricalArrays", "CoDa", "Distances", "ScientificTypes", "Statistics", "StringDistances", "TableOperations", "Tables"]
git-tree-sha1 = "0d70fd4545f63f3a4288b3301a2803128eea0f47"
uuid = "e5d66e97-8c70-46bb-8b66-04a2d73ad782"
version = "0.1.4"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[TableTransforms]]
deps = ["Distributions", "LinearAlgebra", "ScientificTypes", "Statistics", "Tables", "Transducers"]
git-tree-sha1 = "30b03c6826cbf30750945d20dbc094983fa86821"
uuid = "0d432bfd-3ee1-4ac1-886a-39f05cc69a3e"
version = "0.1.14"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[TypedTables]]
deps = ["Adapt", "Dictionaries", "Indexing", "SplitApplyCombine", "Tables", "Unicode"]
git-tree-sha1 = "f91a10d0132310a31bc4f8d0d29ce052536bd7d7"
uuid = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"
version = "1.4.0"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[UnicodePlots]]
deps = ["ColorTypes", "Contour", "Crayons", "Dates", "FileIO", "FreeTypeAbstraction", "LinearAlgebra", "MarchingCubes", "NaNMath", "SparseArrays", "StaticArrays", "StatsBase", "Unitful"]
git-tree-sha1 = "e7b68f6d25a79dff79acbd3bcf324db4385c2c6f"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "2.10.1"

[[Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "b649200e887a487468b71821e2644382699f1b0f"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.11.0"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[Variography]]
deps = ["Distances", "GeoStatsBase", "InteractiveUtils", "LinearAlgebra", "Meshes", "NearestNeighbors", "Optim", "Printf", "Random", "RecipesBase", "Setfield", "SpecialFunctions", "Tables", "Transducers", "Unitful"]
git-tree-sha1 = "9c64543903f0c7baebfdefa0ad3d494722aef094"
uuid = "04a0146e-e6df-5636-8d7f-62fa9eb0b20c"
version = "0.14.3"

[[VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[VoronoiCells]]
deps = ["GeometryBasics", "RecipesBase", "VoronoiDelaunay"]
git-tree-sha1 = "56366c6d0ce6c27aa0ea8a5a4909ac434f0fd33f"
uuid = "e3e34ffb-84e9-5012-9490-92c94d0c60a4"
version = "0.3.0"

[[VoronoiDelaunay]]
deps = ["Colors", "GeometricalPredicates", "Random"]
git-tree-sha1 = "ed19f55808fb99951d36e8616a95fc9d94045466"
uuid = "72f80fcb-8c52-57d9-aff0-40c1a3526986"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libgeotiff_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Libtiff_jll", "PROJ_jll", "Pkg"]
git-tree-sha1 = "91197a1c90fc19ce66e5151c92d41679a52ad4b5"
uuid = "06c338fa-64ff-565b-ac2f-249532af990e"
version = "1.7.0+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═49d1ea64-9535-11ec-2eab-119fadb9083e
# ╟─800a75e5-4f56-49e9-8d77-d62cce02ebad
# ╟─8bb0c644-ce7b-49d8-8107-fd473f281199
# ╠═c8ed3ef3-eb49-45c3-933d-6935d8ea14eb
# ╟─e2e418ad-9fcf-4c8b-b20f-383efe07f094
# ╠═e2ed76ac-9b7d-4e9c-9b14-48b4feb9edeb
# ╠═68884f19-bca4-40b0-8a6a-91783e9bc7ac
# ╠═9398797a-82f1-42b2-ae54-5392c3ffb437
# ╠═5e8216f3-8a86-4d6e-8444-11ba63785c1a
# ╟─1d6bd5a3-0b3d-4870-b5bb-7a39f6494394
# ╟─8ef3ef24-6ab4-4e55-a6bc-6ee4ac901a53
# ╠═4d822b90-5b31-4bb3-954b-71675e6c1dd9
# ╟─6d921f17-a3b7-469d-9df2-6d02022fcc33
# ╟─bf92bd9a-eb71-468e-b7c2-1b5980dd0a5c
# ╠═08d4a328-141c-4869-88c7-09ea8ff57590
# ╟─464a6802-c5eb-4a42-b9f4-d0dde12e24e8
# ╟─5157cf2e-5b1c-447c-9d59-9064ecf9a771
# ╟─31f66cb8-afa3-4ebc-9d7d-7d220997ba1f
# ╟─8594d3ad-0e04-49d3-86e1-cdc0c8f8951c
# ╠═0d9ba991-3d5a-4a15-84fd-a9641323e0d6
# ╠═1b9654c4-ea9a-4dcd-a610-6295732d96b3
# ╠═3a92bcdc-9c40-4fea-8955-f011b41a66c0
# ╟─9d4de2c5-c0d2-468d-b694-c678c6961857
# ╠═20886cf8-b48d-4293-b756-a894279d11f8
# ╟─e9df25ae-f0e5-4634-8b44-f40576c7fc87
# ╠═a0b152a2-6189-459a-9288-514c8b3d14f5
# ╟─e38363af-1059-4c7e-bc2b-94924f34b01b
# ╟─9d65dabb-d4d2-49c1-96f2-1f5a02bdbecf
# ╠═4608787a-e817-4ae4-bfd7-018445586632
# ╟─8dc094fc-f1be-4640-b97e-94ed3e87e6a6
# ╟─f05d157c-f2a8-421f-af61-64461b82d863
# ╟─58e457b8-ade1-43d2-8717-ea750e84a3a1
# ╠═143beaa8-65cb-48fd-a981-92e1543c123b
# ╟─6eb80555-636d-4072-b8a2-984a09f7f833
# ╟─c281ed6a-7adb-489c-afde-292b0cf76485
# ╟─9de45182-910c-4451-824a-0c08568c5fd0
# ╟─25a32915-7c5e-4e7b-bb94-d1571120dbaa
# ╟─a96bb753-6b33-4bba-9591-aaf007432c0a
# ╠═b485d1e4-c249-47ad-8e45-707756675fdb
# ╟─e013cb41-819d-4b8a-8381-0784167b401a
# ╠═ffaa5a0d-d4eb-42a6-b817-f3571300257c
# ╟─6b123eec-f8e0-46df-86d1-06c7b4c3cc89
# ╟─90638da7-7b62-43ec-89d9-770acf945c71
# ╟─bfef8e4e-0335-4a32-b23b-801a79e0c269
# ╟─d117da6c-bd22-45ed-aff2-baff497858a0
# ╟─e9627745-e40d-41e4-b569-4ed79f5686c4
# ╠═ee5a17b7-d9d0-410f-8196-9a3d9d2010d1
# ╠═666fdd7b-8062-45d9-822b-bedeca682576
# ╠═80339671-3cc9-42cc-858a-64ae9bd9b5a8
# ╠═c14e276e-4b3c-4e67-b5a0-5b19d7a21180
# ╟─b49eb729-7d16-4165-8c80-e901cbbf3b01
# ╠═e36d5b6c-9a95-4570-9b4a-d3bf4f48137f
# ╠═97efb80c-0915-42df-8199-6df2639d2d41
# ╟─223a20c9-6a8e-4014-a909-1e4e8525d282
# ╟─da770866-0a26-4722-98a8-cf7965e35d15
# ╟─ebd20a55-ae35-463f-ad81-fb5dc35c8b41
# ╠═a5be3813-81c3-4a08-85ad-cb43f317bf60
# ╟─13046794-2b45-43f2-a4dd-484173181109
# ╟─3a6fc1ba-95fa-446c-97be-36346ff562a6
# ╠═9f8807ca-0ec6-43f2-83b5-8a76f8feebd5
# ╟─a28efb16-2f08-4c69-9851-153ba4c6bf1a
# ╠═ccfc42a6-a122-4ddb-b5a2-f7cd7101789e
# ╟─cc6f26f6-3db6-41b4-9e71-b6d48592c410
# ╠═6b205e8b-f212-4668-a31e-c5af409c6e93
# ╟─caec5908-a4c6-4bc0-8af3-6e098892e24b
# ╠═c3d1cadb-c3d3-4452-aa8a-1f50cdee1448
# ╟─dc647053-fdef-49c1-98e7-6292b9ac55c7
# ╟─1f34e819-9eef-4157-8175-c90c8f92882c
# ╟─b7cd9153-1353-4051-9197-d8137602d3fe
# ╟─f7b688df-5589-4bdd-b431-94e770bc8a62
# ╟─a3efc796-7604-4260-96d1-9b1c3cd60c0d
# ╟─7f8897ad-e30f-48f0-94e2-aa8fbac7954e
# ╟─7eb37472-8ee6-4c9b-bf27-eff4905b4e2e
# ╟─ec822073-7e2a-44b7-8739-0e094a673066
# ╟─501e8ada-010a-4fae-bb2d-855baa9cf923
# ╠═65ff833e-ca70-408a-992a-128108401dc8
# ╟─568fe19d-4a33-4e4f-b3eb-d50aaeef6eed
# ╟─d915b138-e81d-4bca-941e-a9f15f56914d
# ╟─1d2b44be-6ece-4ec0-9142-ea480095b5ac
# ╟─f7128f8a-77f1-4584-9cd6-2660ebfb084b
# ╟─85716a1c-626a-4ac7-a0d9-37cf0f658f79
# ╠═3e737332-78f9-4a93-931c-3b4bd2a1c195
# ╟─3216b2cb-976e-40cd-aee4-69af2ce5cb31
# ╟─4bc5a61d-12e2-4c55-a9fa-3af21cf55f7e
# ╟─0c400d13-dc44-4bdd-b627-2ad551109cf9
# ╟─c4aa4c0a-6ad0-418d-8c76-56d2a840905c
# ╟─73c36e85-ad6f-42e2-a5c0-5fc10c8db7ce
# ╟─54754fac-ceab-47e3-ac6b-d8f837078072
# ╟─75bbca6f-6410-4a93-854c-3ea030f94217
# ╟─85e138e8-fd2f-4239-9b41-2aa6cb9d8ed2
# ╟─bacb917d-8d9a-4081-8039-966849ade9d6
# ╟─e1bc6d5a-99c8-4d83-aad6-491b5ededfd0
# ╟─1cc473f5-6ac6-4876-af22-987768f2ad16
# ╟─d8512b16-84f5-4b0a-a7de-3bffb3367242
# ╟─05d62c11-9bbf-4b4d-bdf0-b43685e33f1c
# ╟─56ac5f27-682a-42f7-99ea-7882f29506fb
# ╟─18cbefa1-d451-47a7-9e75-b00bbfa2f78a
# ╟─56683a3f-cab1-478a-a4fa-25c7a5df3ce6
# ╠═ee1f546d-2755-48e2-b18c-17e7d1c55298
# ╟─4718a5e0-c28e-4ea2-8e3c-d651740ea344
# ╠═596d95ef-8c84-4e3e-a5a5-bde5c50518b7
# ╠═98eb52ed-d4df-4e20-8ca2-3ae7b50e1999
# ╠═3566b173-ae10-4b65-83a0-d040e802d181
# ╟─a87ee29b-a9bb-472d-a9a3-51224e0a1bdd
# ╠═990b6d78-4650-4665-af61-22fc53fb50b7
# ╠═d61157fc-b565-4e61-9d55-4199745c5b05
# ╠═f62813e2-fed3-4b53-a201-bfe9e7f0bfc0
# ╟─084c08df-f535-496d-b578-2a83f7b9082b
# ╠═66b45f66-285c-47eb-971b-5593156d3809
# ╟─0c6264c1-9dad-4b27-8a52-02123880c4cb
# ╟─1dd01920-0da2-406c-9bd2-10409b38321c
# ╟─813feb02-4c08-482d-a033-56531f0e1d1b
# ╟─afeba502-4a7e-4f7a-b1bb-d4f617763ca5
# ╟─cc045caa-5293-47a8-bbc5-f174ab86cb2b
# ╟─cd5ccc7d-3113-4db3-bf64-1f0cb8a1e249
# ╟─11330b51-1bbe-4a7f-b398-1dda6848a669
# ╟─23b2ae8e-0149-4df8-b23f-6d48460e779a
# ╠═3498d6ee-56f8-402d-8917-2fa2aace76f9
# ╠═c1f54fd5-404d-4fef-b140-130baccd1d57
# ╟─8c07ba91-7f04-46dd-9f30-41df91ee36e2
# ╟─0d42b241-e7f6-488a-9524-4d3612ba4f2a
# ╟─46adf87b-7d4c-4c0f-897d-631fec46cab5
# ╠═b42f75aa-06fb-449e-b9e2-01146d37b218
# ╟─acd39a29-e8b9-405a-b37d-fe6b17d210dd
# ╠═502364d2-456a-4140-bbf9-70b9dc7363d8
# ╠═d94975d9-bb0b-4363-ad73-18f40f64fef2
# ╟─5c76d3ce-913d-4dd4-875c-c8f693b0d337
# ╠═2b7cc18f-8f6c-4163-8da0-3774bf853a4f
# ╟─f857d45f-767a-4c91-96fd-21d944070fbd
# ╠═88fcd33a-a574-45af-8677-26511b5bdf7e
# ╠═8f94298d-533d-4252-b0e8-180808719a25
# ╟─bab915c5-61c1-47f3-9211-87ecb552734d
# ╟─4c31b1a5-adf8-42ae-8f34-f371274c61e1
# ╠═8452a50c-1f3d-4407-bfd1-b3e9e50bb5dc
# ╟─9e8476f5-f39b-4961-86a0-7efd9e8bc9e8
# ╟─e9f57d8a-e13a-472d-a567-790cab0e7c1b
# ╟─4d21be1f-a5e8-405d-9d54-5a0147a202da
# ╟─cafa6163-7b7f-456b-ba28-816ad9fbefc9
# ╠═88839489-e6ec-4ff4-9d79-cd9da526e35c
# ╠═1ed35dc3-c604-450a-9218-a49a9b6df84b
# ╟─6837ceea-9d9e-45c5-9f18-f42784227463
# ╠═68b2baf3-cc88-45ef-a346-0e731449b00b
# ╠═5bac64d2-36ed-493c-b4a5-527fa5c336be
# ╟─7f427ca1-9311-4a52-936b-0f0e23fd8be7
# ╠═bcddd86e-28a0-4df9-836d-ba985248e889
# ╠═2ca23b00-4c41-4cb6-a6af-331e12c814de
# ╠═cf8d2ff2-0bac-44ef-9c2c-1e4fe118ed0b
# ╠═a6bc01d8-2dbb-453c-8dce-92cbc8c015a7
# ╠═30c53681-703e-4c0e-9c18-01b04d232ac3
# ╠═e0716bcb-45ed-4b9b-bd3a-1e48d06af32b
# ╠═44751276-26d9-421c-8096-a1bf422d6c04
# ╠═70294579-9025-4709-accd-3d0279df91c7
# ╠═84351e0e-c118-4567-a4d7-86af12b0dfb2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
