
from utils.model_utils import quick_eval
import utils.cross_validation as cval
import random
import pandas as pd
from utils.model_utils import train_test_split_v2
from sklearn.ensemble import RandomForestRegressor


test_size=0.2
random_key=42

biome_mapping = {
    'Temperate broadleaf forests': 'Temperate broadleaf forests',
    'Temperate conifer forests': 'Temperate conifer forests',
    'Temperate grasslands': 'Temperate grasslands',
    'Xeric shrublands': 'Xeric shrublands',
    'Mediterranean woodlands': 'Mediterranean woodlands',
    'Flooded grasslands': 'Flooded grasslands',
    'Mangroves': 'Mangroves',
    'Tundra': 'Boreal and Tundra forests',
    'Boreal forests or taiga': 'Boreal and Tundra forests',
    'Tropical grasslands': 'Tropical',
    'Tropical coniferous forests': 'Tropical',
    'Tropical moist broadleaf forests': 'Tropical',
    'Tropical dry broadleaf forests': 'Tropical',
    np.nan: np.nan,  # Will be dropped
}

fd_df = pd.read_csv('data/final/final_dataset_with_aridity.csv')
# fd_df.drop(columns=[ 'managed', 'ownership','Functional_Divergences', 'Shannon Equitabiltiy Index', 'DIA'], inplace=True)
fd_df.dropna(subset=['Raos_Q', 'Functional_Evenness',
                     'Species Richness', 'Shannon Diversity', "Simpson's Index", 'biome'], inplace=True)

ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

ecoregions=ecoregions[['ECO_NAME', 'geometry']]

#Preprocessing the data for Random forest regression
# fbiome_dfs= data_processing_v2(fd_df, biome_mapping, ecoregions)

fd_df['biome'] = fd_df['biome'].map(biome_mapping) # Only run this once ! 

biome_dfs = {k: v for k, v in fd_df.groupby('biome')}


biome_list=["Temperate broadleaf forests", "Temperate conifer forests", "Temperate grasslands",
             "Xeric shrublands", "Boreal and Tundra forests", "Tropical",
               "Mediterranean woodlands"]

rf_params = {
    'n_estimators': 200,          # Fewer trees for simpler model
    'max_depth': 6,               # Shallower = more interpretable
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'n_jobs': -1,
    'random_state': 42
    }
for biome in biome_list:
    print(f"Biome: {biome}")

    random.seed(random_key)

    n_points= round(len(biome_dfs[biome])*test_size)
    fX_train, fy_train, fX_test, fy_test = train_test_split_v2(biome_dfs[biome], random_key=random_key, n_points_to_select=n_points, buffer_km=50)
    
    model = RandomForestRegressor(**rf_params)
    model.fit(fX_train, fy_train)
    fig, metrics = quick_eval(
        model=model,
        X_test=fX_test,
        y_test=fy_test,
        biome_name=biome,
        feature_names=fX_test.columns.tolist(),
        show_shap=True
    )