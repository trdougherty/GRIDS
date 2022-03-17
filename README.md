# dbr
Creation and Data pipeline for developing dbrs

### Data Preprocessing
All of the data processing is done in a single Julia notebook (Pluto) called data_prep.jl. Reference that for all of the data prep.
It should have all of the annotations needed to go step by step through the data cleaning and preparation.

### Data Cleaning Pipeline
1. Initial Weather Data is loaded from NOAA
2. Weather Data is cleaned and anomalies are detected and dropped
3. Building Data is loaded - in our case this comes from the list of buildings in New York City
4. A region of interest is defined as a bounding box which surrounds our buildings
5. This bounding box is then "broken" into weather zones using a voronoi paritioning scheme with the good weather stations
6. Each building is then assigned to one of the weather zones, based on the containing region
7. Satellite photos - a high resolution image is comissioned based on the bounding box region
8. Each "sub" image is then captured by simply indexing the high resolution image with a box around each building location

