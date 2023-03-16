# dbr
Creation and Data pipeline for developing dense building representations (dbrs). References to all of the necessary files for preprocessing the data can be found in _sources.yml_. The root of each dataset will be distinguished by the following parameter:

**city:** _path/to/city/directory/sources.yml_

### Energy and Footprints
Data should be processed using the format defined by the macro_climate utility, with two files for each region.

**energy:** _path/to/city/directory/energy.csv_

_Energy Dataset:_
| Footprint ID | Year | Energy (MWh) | Area (m<sup>2</sup>) |
| :---         |     ---:      |          ---: | ---: |
| 15842106277983404298   | 2016     | 3521.06    | 6503.21 |
| 5206367234345948710   | 2018     | 2277.87    | 10611.11 |

**footprints:** _path/to/city/directory/footprints.geojson_

_Footprints Data:_
| ID | Geometry |
| :---         |     :---:      |
| 15842106277983404298 | POLYGON ((...)) |
| 5206367234345948710 | POLYGON ((...)) |

### Streetlevel Imagery
We additionally need two more files with information about the adjacent streetlevel data. We utilize the architecture of OneFormer to run panoptic segmentation, which outputs two files for each streetview image. These are saved as a json format, with identifiers mapping the same classes used in the cityscapes training set. Compressing these json files into a dataframe will provide a structure which looks like this:

**panoptic-data:** _path/to/city/directory/images/data.csv_

_Streetview Data - postprocessed:_
| ID | Sky | Vegetation | Road |
| :---         |     ---:      | ---: | ---: |
| 2110718573383811636 | 0.24 | 0.11 | 0.22 |
| 9501152238164954080 | 0.32 | 0.07 | 0.15 |


We additionally need a secondary datasource with geographic information for each street level image, to then identify which streetview images will be immediately associated with the building of interest:

**panoptic-locations:** _path/to/city/directory/images/locations.geojson_

_Streetview Data - geolocated:_
| ID | Geometry |
| :---         |     :---:      |
| 2110718573383811636 | POINT ((...)) |
| 9501152238164954080 | POINT ((...)) |

### Data Processing Steps
1. Collection of Energy Data, Footprints, and Streetview Imagery
2. Download and unpack the panoptic segmentation machine provided by OneFormer `unzip data.zip`
<!-- 2. Preprocessing of images:  `python src/preprocess.py city_sources.yml` -->
3. Geospatial joins and links definitions between images: `python src/links.py --config sources.yml --city [city(s)]`
4. Training: `python src/train.py --config sources.yml --city [city(s)]`
5. Inference: `python src/inference.py --config sources.yml --city [city(s)]`

### Custom Training Setup
You need at least four files for each city to curate the computational graph and undergo inference and training.

1. footprints.geojson - building footprint, which will be used to calculate the floor area
2. nodes.geojson - definition of properties on the streetview imagery. at a minimum this needs the image id and the location
3. panopticdata.csv - output of oneformer compression, this has information about the contents of each image
4. (training) energy.csv - ground truth data for the building's energy consumption

## Validation of accuracy
Validation used is RMSE, as the validation metrics found in [the ASHRAE handbook](http://www.eeperformance.org/uploads/8/6/5/0/8650231/ashrae_guideline_14-2002_measurement_of_energy_and_demand_saving.pdf) are defined for monthly and hourly predictions.
<!-- The error terms were created using the guidance of [this ASHRAE handbook](http://www.eeperformance.org/uploads/8/6/5/0/8650231/ashrae_guideline_14-2002_measurement_of_energy_and_demand_saving.pdf) -->
