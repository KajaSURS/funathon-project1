# %%
import pandas as pd

# %%
dict_fr_en = {
  'idmutation':['trans_id', 'Unique identifier code of the transaction'],
  'datemut':['trans_date', 'Date of the official certified transaction'],
  'anneemut':['trans_year', 'Year of the official certified transaction'],
  'moismut':['trans_month','Month of the official certified transaction'], 
  'idnatmut':['trans_type_code','Type of transaction','There are several types of transaction in the original data (sale, off-plan sale, sale of building land, tender, compulsory purchase). Original data have been filtered to keep only sale.'], 
  'libnatmut':['trans_type_label','Type of transaction'],
  'valeurfonc':['price','Price of the transaction', 'Price of the transaction is in EUR'],  
  'dteloc':['prop_type','Type of property',  '1 represents a flat and 2 a house'],
  'jannath':['prop_year','Year of contruction of the property', 'This variable has been harmonized to correct for typing mistakes. More details is available in the introduction.'],
  'ccodep':['prop_loc_dep','Department code where the property is located', "Data doesn't cover the whole French territory - overseas territory are included but Alsace-Moselle (Eastern part of France) isn't"],  # list of departments in France  - overseas yes but Alsace Moselle nope
  'depcom':['prop_loc_',"Official city's code where the property is located", 'see remarks above'],  # COG : official geographical code https://www.insee.fr/en/metadonnees/source/serie/s2084
  'x':['prop_loc_x','Longitude where the property is located', 'see remarks above'],  # Type of projection ?
  'y':['prop_loc_y','Latitude where the property is located', 'see remarks above'],  # Type of projection ?
  'distance_ltm':['dist_tosea','Distance of the property to the nearest seashore - capped at 10km'],
  'distance_ltm_corr':['dist_tosea_corr','Corrected distance of the property to the nearest seashore - capped at 10km'],
  'dnbniv':['n_floors','Number of floor in the property (building or house)', 'This variable is more reliable with houses than with buildings. Underground floors encoding is not fully harmonized and is often equal to 81 for minus 1, 82 for minus 2 ... It can also be encoded as 99, 98. A flat at the 2nd floor of a seven-floors building should be encoded with nth_floor=1 and n_floors=8 (ground floor and seven floors above ground level)'], # 0 to 90
  'dnbbai':['n_bath','Number of bathtubs reported in the property'],  # 0 to 41
  'dnbdou':['n_show','Number of showers reported in the property'],  # 0 to 44
  'dnblav':['n_sink','Number of sinks reported in the property'],  # 0 to 90
  'dnbwc':['n_wc','Number of toilets reported in the property'], # 0 to 51
  'dnbppr':['n_mrooms','Number of main rooms reported in the property', 'n_mrooms = n_eatr + n_slr + n_kit8 + n_kit9 + n_washr'], # 0 to 99 
  'dnbsam':['n_eatr','Number of eating rooms reported in the property'], # 0 to 99
  'dnbcha':['n_slr','Number of sleeping rooms reported in the property'], # 0 to 99
  'dnbcu8':['n_kit8','Number of kitchens reported in the property with an area of less than 8 square meters'],  # 0 to 90
  'dnbcu9':['n_kit9','Number of kitchens reported in the property with an area of larger than 9 square meters'],# 0 to 90
  'dnbsea':['n_washr','Number of washing rooms reported in the property'],  # 0 to 99
  'dnbann':['n_ancrooms','Number of ancillary rooms reported in the property', 'Ancillary rooms include hallways, attics. They differ from n_otherannex.'], # 0 to 99
  'dnbpdc':['n_rooms','Number of rooms reported in the property', 'n_rooms = n_mrooms + n_annex'], # 0 to 99
  'dsupdc':['farea','Reported floor area of the property'], # 15 to 1915
  'geaulc':['has_water', 'If the property has access to water'], # 0 to 2 ## 2 and more ?  pas une indicatrice ?
  'gelelc':['has_elec', 'If the property has access to electricity'], # 0 to 2 ## 2 and more ?  pas une indicatrice ?
  'gesclc':['stair', 'If the property has stairs'], # 0 to 2 ## 2 and more ?  
  'ggazlc':['has_gas', 'If the building is connected to the gas mains'], # 0 to 2 ## 2 and more ?  pas une indicatrice ?
  'gasclc':['has_elevator', 'If the building of the flat has an elevator'], # 0 to 2 ## 2 and more ? 
  'gchclc':['gteglc'], # 0 to 2 ## 2 and more ? 
  'gvorlc':['gteglc'], # 0 to 2 ## 2 and more ? 
  'gteglc':['gteglc'], # 0 to 2 ## 2 and more ? 
  'dniv':['nth_floor','Reported floor of the property', 'It represents the floor of the flat (in France, the second floor is the first floor above ground level). This variable is set to 00 for houses. Underground floors encoding is not fully harmonized and is often equal to 81 for minus 1, 82 for minus 2 ... It can also be encoded as 99, 98. A flat at the 2nd floor of a seven-floors building should be encoded with nth_floor=1 and n_floors=8 (ground floor and seven floors above ground level). '], # 0 to 99 ## 99 and more ? 
  'dcntsol':['s_land_artif', 'Artificial land area (square meters)', 'Artificial land includes recreational areas, land, building plots and gardens. Artificial land refers to land that has been altered by humans.'], # 0 to 1684404  
  'dcntagri':['s_land_agri', 'Agricultural land area (square meters)', 'Agricultural land is used for farming. It includes fields, meadows, orchards and vineyards. '], # 0 to 4486113  
  'dcntnat':['s_land_nat', 'Natural land area (square meters)', 'Natural land is land that has not been altered. This includes, for example, forests. '], # 0 to 4894480 
  'nb_garages/ffnbpgarag':['n_garage','Number of garages reported in the property'], # 0 to 215
  'nb_piscines/ffnbppisci':['n_pool','Number of pools reported in the property'], # 0 to 3 ## 3 and more ? 
  'nb_terrasses/ffnbpterra':['n_terrace','Number of terraces reported in the property'], # 0 to 5 ## 5 and more ? 
  'nb_greniers/ffnbpaut':['n_attic','Number of attics reported in the property'], # 0 to 13 ## 13 and more ? 
  'nb_caves/ffnbpaut':['n_basmt','Number of basements reported in the property'], # 0 to 22 ## 22 and more ? 
  'nb_autresdep':['n_otherannex','Number of other annexes reported in the property'],  # 0 to 91 ## 91 and more ? 
  'price_sqm':['price_sqm','Price per square meter of the transaction'],
  'dnivrel':['nth_floor_rel', 'Relative floor of the property (from 0, ground floor, to 1, last floor)']
 }

col_label_fr = "Original label of the variable (French)"
col_label = "Label of the variable"
col_full_name = "Full name of the variable"
col_expl = "Explanation and remarks"

res = pd.DataFrame.from_dict(dict_fr_en, orient="index", columns=[col_label, col_full_name, col_expl]
    ).reset_index(names=col_label_fr).sort_values(col_label).fillna('')

res[' '] = range(1, res.shape[0]+1)

with open('table_dict.Qmd', 'w') as f: 
    f.write(res[[" ", col_label, col_full_name, col_expl, col_label_fr]].to_markdown(index=False))

# %%
