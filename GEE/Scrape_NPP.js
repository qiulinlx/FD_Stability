// ==================================================
// Mean + SD Annual MOD17 NPP per PID Location per Year
// ==================================================

// --------------------------------------------------
// USER INPUT: PID asset path
// --------------------------------------------------
var pidLocations = table; // table = PID_locations

// OPTIONAL (recommended for point plots):
// Uncomment to buffer points by 250 m
// pidLocations = pidLocations.map(function(f) {
//   return f.buffer(250);
// });

// --------------------------------------------------
// Parameters
// --------------------------------------------------
var startYear   = 2011;
var endYear     = 2020;
var scaleFactor = 0.0001; // MOD17 scale factor (kg C m-2 yr-1)

// --------------------------------------------------
// Load MOD17 Annual NPP
// --------------------------------------------------
var mod17 = ee.ImageCollection('MODIS/061/MOD17A3HGF')
  .filter(ee.Filter.calendarRange(startYear, endYear, 'year'))
  .select('Npp');

// --------------------------------------------------
// Function: extract mean + SD per location for one year
// --------------------------------------------------
var extractYear = function(img) {
  var year = ee.Date(img.get('system:time_start')).get('year');
  var nppScaled = img.multiply(scaleFactor);

  var stats = nppScaled.reduceRegions({
    collection: pidLocations,
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.stdDev(), '', true),
    scale: 500
  });

  return stats.map(function(f) {
    return f.set('year', year);
  });
};

// --------------------------------------------------
// Apply across years and flatten
// --------------------------------------------------
var results = mod17.map(extractYear).flatten();

// --------------------------------------------------
// Inspect output
// --------------------------------------------------
print('Annual mean + SD NPP per PID location', results.limit(10));

// --------------------------------------------------
// Export to CSV
// --------------------------------------------------
Export.table.toDrive({
  collection: results,
  description: 'MOD17_Annual_NPP_Mean_SD_per_PID_2011_2020',
  folder: 'GEE_MOD17',
  fileNamePrefix: 'MOD17_Annual_NPP_Mean_SD_per_PID',
  fileFormat: 'CSV'
});
