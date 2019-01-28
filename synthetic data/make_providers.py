import json
import pickle

import pandas as pd


def lowerize(strings_list):
    return list(map(lambda item: item.lower(), strings_list))


#######################################################################################################################
diag_df = pd.read_csv("resources/CMS32_DESC_LONG_SHORT_DX.csv")
df_icd_codes, df_diag_desc = diag_df[list(diag_df)[0]].values.squeeze().tolist(), diag_df[list(diag_df)[1]].values.squeeze().tolist() + diag_df[list(diag_df)[2]].values.squeeze().tolist()
#######################################################################################################################
sig_df = pd.read_csv("resources/CMS32_DESC_LONG_SHORT_SG.csv")
df_cpt_codes, df_treat_desc = sig_df[list(sig_df)[0]].values.squeeze().tolist(), sig_df[list(sig_df)[1]].values.squeeze().tolist() + sig_df[list(sig_df)[2]].values.squeeze().tolist()
#######################################################################################################################
dsm_df = pd.read_csv("resources/dsmiv-code-table.csv")
dsm_codes_df, dsm_desc_df = dsm_df[list(dsm_df)[0]].values.squeeze().tolist(), dsm_df[list(dsm_df)[1]].values.squeeze().tolist()
#######################################################################################################################
hospitals_df = pd.read_csv("resources/Hospitals.csv")
hospital_name_df, hospital_address_df = hospitals_df[["NAME"]].values.squeeze().tolist(), hospitals_df[["ADDRESS"]].values.squeeze().tolist()
#######################################################################################################################
population_df = pd.read_csv("resources/US City Populations.csv")
city_df = population_df[["City"]].values.squeeze().tolist()
#######################################################################################################################
zips_df = pd.read_csv("resources/zipcode.csv")
zip_df = zips_df[["zip"]].values.squeeze().tolist()
#######################################################################################################################
cities_kaggle = json.load(open("resources/result.json"))
zip_code_kaggle, _ = list(cities_kaggle.keys()), list(cities_kaggle.values())
#######################################################################################################################
icd_codes_github_df = pd.read_csv("resources/icd codes.csv")
diag_code_github = icd_codes_github_df.iloc[:, 2].values.squeeze().tolist()
diag_desc_github = icd_codes_github_df.iloc[:, 3].values.squeeze().tolist() + icd_codes_github_df.iloc[:, 4].values.squeeze().tolist()
#######################################################################################################################
final_addresses = json.load(open("resources/my_addresses.json"))
final_diagnosis = list(set(diag_desc_github + df_diag_desc))
final_icd_codes = list(set(diag_code_github + df_icd_codes))
final_cpt_codes = df_cpt_codes
final_treatments = df_treat_desc
final_dsmv_description = dsm_desc_df
final_dsmv_codes = dsm_codes_df
final_hospitals_names = hospital_name_df
final_hospital_address = hospital_address_df
final_cities = lowerize(city_df)
final_zip_codes = zip_code_kaggle + zip_df

# with open("./providers.pkl", 'rb') as f:
#     cities, addresses, hospitals_names, hospital_address, icd_codes, diagnosis, cpt_codes, treatments, dsmv_codes, dsmv_description, zip_codes = pickle.load(f)
#
# assert zip_code_kaggle + zip_df == zip_codes
# assert lowerize(city_df) == cities
# assert hospital_name_df == hospitals_names
# assert hospital_address_df == hospital_address
# assert df_cpt_codes == cpt_codes
# assert df_treat_desc == treatments
# assert dsm_desc_df == dsmv_description
# assert dsm_codes_df == dsmv_codes

with open("providers.pkl", 'wb') as f:
    pickle.dump((final_cities, final_addresses, final_hospitals_names, final_hospital_address, final_icd_codes, final_diagnosis, final_cpt_codes, final_treatments, final_dsmv_codes, final_dsmv_description, final_zip_codes), f)  # dumps get the string
