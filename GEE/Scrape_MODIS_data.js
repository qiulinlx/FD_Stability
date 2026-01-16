// --------------------------------------------------
// Region: Define bounding box and Year of interest
// --------------------------------------------------
var region = ee.Geometry.Rectangle([-124.48, 32.53, -114.13, 42.01]);
var year = 2015;

// --------------------------------------------------
// 1. Land Surface Phenology (MCD12Q2) (cast to float)
// --------------------------------------------------
var phenology = ee.ImageCollection('MODIS/006/MCD12Q2')
  .filter(ee.Filter.calendarRange(year, year, 'year'))
  .first()
  .select([
    'Greenup_1',
    'Peak_1',
    'Senescence_1',
    'Dormancy_1'
  ])
  .rename([
    'greenup',
    'peak',
    'senescence',
    'dormancy'
  ])
  .toFloat()   
  .clip(region);

// --------------------------------------------------
// 2. Albedo (MCD43A3) — robust
// --------------------------------------------------
var albedoCol = ee.ImageCollection('MODIS/006/MCD43A3')
  .filterBounds(region)
  .filter(ee.Filter.calendarRange(year, year, 'year'))
  .select('Albedo_WSA_shortwave');

var albedo = albedoCol
  .mean()
  .multiply(0.001)     // scale factor
  .rename('albedo')
  .clip(region);

// --------------------------------------------------
// 3. FAPAR (MOD15A2H)
// --------------------------------------------------
var faparCol = ee.ImageCollection('MODIS/006/MOD15A2H')
  .filterBounds(region)
  .filter(ee.Filter.calendarRange(year, year, 'year'))
  .select('Fpar_500m');

var fapar = faparCol
  .mean()
  .multiply(0.01)      // scale factor
  .rename('fapar')
  .clip(region);


// --------------------------------------------------
// 4. NDVI (MOD13A1, 500 m, 16-day)
// --------------------------------------------------
var ndvi = ee.ImageCollection('MODIS/006/MOD13A1')
  .filterBounds(region)
  .filter(ee.Filter.calendarRange(year, year, 'year'))
  .select('NDVI')
  .mean()
  .multiply(0.0001)
  .rename('ndvi')
  .clip(region);
  
  
// --------------------------------------------------
// 5. Stack into single multiband image
// --------------------------------------------------
var modis_stack = phenology
  .addBands(albedo)
  .addBands(fapar)
  .addBands(ndvi);

// --------------------------------------------------
// 6. Export single multiband raster
// --------------------------------------------------

modis_stack = modis_stack.toFloat();

Export.image.toDrive({
  image: modis_stack,
  description: 'MODIS_Phenology_Albedo_FAPAR_NDVI_CA_2015',
  folder: 'MODIS',
  region: region,
  scale: 500,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});