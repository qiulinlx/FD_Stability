// ============================================================
// MODIS Annual NPP extraction for PID points, 2010–2020
// MODIS product: MOD17A3HGF v061 — Annual Net Primary Production
// Units: kg C/m²/yr (scale factor 0.0001 applied)
// Input table columns: PID, lat, lon, BHAGE, managed, ownership, biome
// ============================================================

// --- 1. Load your uploaded CSV as a GEE Asset ---
// Upload PID_sample.csv via Assets panel → New → CSV upload
// Then replace the path below with your asset path:
var table = ee.FeatureCollection('projects/unique-rarity-475113-h0/assets/PID_location_part2');

// --- 2. Build point geometries from lat/lon columns ---
// GEE CSV imports don't auto-create geometry, so we do it manually.
// Note: your CSV has lat first, then lon — handled correctly below.
table = table.map(function(f) {
  var geom = ee.Geometry.Point([
    ee.Number.parse(f.get('lon')),
    ee.Number.parse(f.get('lat'))
  ]);
  return f.setGeometry(geom);
});

// --- 3. Load MODIS Annual NPP 2010–2020 ---
var nppCollection = ee.ImageCollection('MODIS/061/MOD17A3HGF')
  .filterDate('2005-01-01', '2024-12-31')
  .select('Npp');

// --- 4. Extract NPP per point per year ---
var years = ee.List.sequence(2010, 2020);

var results = years.map(function(year) {
  year = ee.Number(year);

  var img = nppCollection
    .filterDate(
      ee.Date.fromYMD(year, 1, 1),
      ee.Date.fromYMD(year, 12, 31)
    )
    .first()
    .multiply(0.0001)       // scale factor → kg C/m²/yr
    .set('year', year)
    .rename('NPP_kgC_m2_yr');

  var sampled = img.reduceRegions({
    collection: table,
    reducer: ee.Reducer.first().setOutputs(['NPP_kgC_m2_yr']), 
    scale: 500              // MOD17A3HGF native resolution (500 m)
  });

  // Tag with year
  return sampled.map(function(f) {
    return f.set('year', year);
  });
});

var output = ee.FeatureCollection(results).flatten();

// // --- 5. Preview in console ---
// print('Total features:', output.size());
// print('Sample (first 10):', output.limit(10));

// Optional: check one PID across all years
// var pid = '2_56_19_181_1';
// print('NPP for ' + pid + ':', output.filter(ee.Filter.eq('PID', pid)));

// --- 6. Export to Google Drive as CSV ---
Export.table.toDrive({
  collection: output,
  description: 'MODIS_NPP_2010_2020_PIDs',
  folder: 'GEE_exports',           // optional: specify a Drive folder
  fileFormat: 'CSV',
  selectors: ['PID', 'lat', 'lon', 'year', 'NPP_kgC_m2_yr']
}); 