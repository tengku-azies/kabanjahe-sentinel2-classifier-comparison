# Machine learning versus conventional classifiers for Sentinel-2 land cover mapping: effect of spatially grouped validation

Code, data, and results for the manuscript:

> M. Miftahurridlo, H. Tjahjono, and T. A. Azis, "Spatially grouped validation narrows the gap between machine learning and conventional classifiers for Sentinel-2 land cover mapping," *(under review)*.

Seven classifiers (RF, CART, SVM, Gaussian NB, MLC, MD, SAM) are compared on one Sentinel-2 Level-2A median composite of Kabanjahe District, Karo Regency, North Sumatra, Indonesia, using the same 15,432 sample pixels from 60 polygons in four classes. Each classifier is evaluated by five-fold cross-validation under two designs: pixel-level random folds and polygon-grouped folds.

## Repository structure

```
scripts/
  01_gee_classification.js   Google Earth Engine: composite, samples, 7 classifiers, cross-validation, maps, areas
  02_reanalysis.py           Python: nested hyperparameter tuning (sensitivity analysis)
  03_stats_from_gee.py       Python: accuracy metrics, bootstrap CIs, McNemar tests from the GEE predictions
data/
  samples.csv                15,432 sample pixels: 10 band values, class, polygon ID, longitude, latitude
  oof_pixel.csv              out-of-fold predictions, pixel-level folds (GEE)
  oof_polygon.csv            out-of-fold predictions, polygon-grouped folds (GEE)
  sample_polygons.*          60 sample polygons (shapefile)
  kabanjahe_boundary.*       district boundary (shapefile)
results/
  gee/                       main results (Tables 2-4 of the manuscript)
  tuning_python/             sensitivity analysis with nested tuning
  maps/                      classified maps (LULC_*.tif) and the 10-band composite
figures/
  figure1_workflow.png, figure2_maps.png
```

Class codes: 0 = built-up (BU), 1 = forest (F), 2 = vegetated cropland (VC), 3 = non-vegetated cropland (NVC).

## Data

- Imagery: Copernicus Sentinel-2 Level-2A (`COPERNICUS/S2_SR_HARMONIZED`), scenes of 22 Oct 2024, 23 Aug 2025, and 22 May 2026, cloud-masked with Cloud Score+ (cs_cdf ≥ 0.60). Contains modified Copernicus Sentinel data (2024–2026).
- Boundary: 2024 administrative boundary of the Geospatial Information Agency of Indonesia (BIG).
- CRS: WGS 84 / UTM Zone 47N (EPSG:32647), 10 m.

## How to reproduce

### 1. Google Earth Engine

1. Upload `data/kabanjahe_boundary.*` and `data/sample_polygons.*` as table assets (Assets → New → Shape files).
2. Open `scripts/01_gee_classification.js` in the Code Editor and define the imports at the top of the script, replacing the asset paths with your own:

   ```javascript
   var geometry = ee.FeatureCollection('users/<your_user>/kabanjahe_boundary').geometry();
   var polys0 = ee.FeatureCollection('users/<your_user>/sample_polygons');
   var areaterbangun = polys0.filter(ee.Filter.eq('class', 0));
   var Hutan = polys0.filter(ee.Filter.eq('class', 1));
   var lahanpertanianbervegetasi = polys0.filter(ee.Filter.eq('class', 2));
   var lahanpertaniantidakbervegetasi = polys0.filter(ee.Filter.eq('class', 3));
   ```

3. Run the script and start the export tasks in the Tasks tab. The outputs are written to the `GEE_Revisi` folder in Google Drive.

### 2. Statistics from the GEE predictions

```bash
pip install -r requirements.txt
cd scripts
python 03_stats_from_gee.py --pixel ../data/oof_pixel.csv --polygon ../data/oof_polygon.csv --out ../results/gee
```

### 3. Sensitivity analysis with nested tuning (optional, about 40 minutes on 2 CPU cores)

```bash
cd scripts
python 02_reanalysis.py cv --samples ../data/samples.csv --out ../results/tuning_python
```

## Methods summary

| Classifier | Implementation | Settings |
|---|---|---|
| RF | `ee.Classifier.smileRandomForest` | 500 trees, 3 variables per split, minimum leaf 1 |
| CART | `ee.Classifier.smileCart` | minimum leaf population 5 |
| SVM | `ee.Classifier.libsvm` | C-SVC, RBF, C = 100, γ = 0.1, standardized bands |
| NB | Gaussian classifier (array operations) | diagonal covariance, equal priors |
| MLC | Gaussian classifier (array operations) | full covariance, equal priors |
| MD | `ee.Classifier.minimumDistance` | Euclidean |
| SAM | `ee.Classifier.minimumDistance` | cosine (spectral angle) |

Confidence intervals use a polygon-level bootstrap (2,000 replicates). Pairwise comparisons use McNemar's test with Holm correction and bootstrap CIs of the OA difference.

## License

Code: MIT License (see `LICENSE`). Data and results: CC BY 4.0.

## Citation

If you use this code or data, please cite the article above and this repository (DOI to be added after archiving on Zenodo).
