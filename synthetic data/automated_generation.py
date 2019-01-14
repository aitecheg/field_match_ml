import pickle
import random
import time

import pandas as pd

import generators

visualize_samples = False  # view each generated sample pair
demo_data = False  # a small testing dataset for humans

with open("./buckets_of_fields.pkl", 'rb') as f:
    fields_containing_dates, fields_containing_compound_names, fields_containing_names, fields_containing_SSN, fields_containing_gender, fields_containing_mail, fields_containing_fax, fields_containing_tele, fields_containing_addresses, fields_containing_icd, fields_containing_MI, fields_containing_zip, fields_containing_yes_no, fields_containing_signature, fields_containing_mailing_address, fields_containing_state, fields_containing_city, fields_containing_tax_id, fields_containing_hospitals, fields_containing_num_hours, fields_containing_length_height_weight, fields_containing_relation, fields_containing_burn_deg, fields_containing_when_to_back, fields_containing_numbers, fields_containing_diag = pickle.load(f)

generator_mapping = [(fields_containing_dates, generators.date_generator),

                     (fields_containing_names, generators.name_generator),
                     (fields_containing_SSN, generators.ssn_generator),
                     (fields_containing_gender, generators.gender_generator),
                     (fields_containing_mail, generators.mail_generator),
                     (fields_containing_fax, generators.fax_generator),
                     (fields_containing_tele, generators.tel_generator),
                     (fields_containing_icd, generators.icd_generator),
                     (fields_containing_MI, generators.mi_generator),
                     (fields_containing_zip, generators.zip_generator),
                     (fields_containing_yes_no, generators.yes_no_generator),
                     (fields_containing_state, generators.state_generator),
                     (fields_containing_city, generators.city_generator),
                     (fields_containing_tax_id, generators.tax_id_generator),
                     (fields_containing_diag, generators.diagnose_generator),
                     (fields_containing_signature, generators.placeholder_generator),
                     (fields_containing_mailing_address, generators.placeholder_generator),
                     (fields_containing_compound_names, generators.placeholder_generator),
                     (fields_containing_num_hours, generators.placeholder_generator),
                     (fields_containing_length_height_weight, generators.placeholder_generator),
                     (fields_containing_relation, generators.placeholder_generator),
                     (fields_containing_burn_deg, generators.placeholder_generator),
                     (fields_containing_when_to_back, generators.placeholder_generator),
                     (fields_containing_numbers, generators.placeholder_generator),

                     ]

generator_mapping = list(map(lambda item: (list(item[0]), item[1]), generator_mapping))
print(sum(list(map(lambda item: 0 if item[1] == generators.placeholder_generator else len(item[0]), generator_mapping))))

if demo_data:
    demo_field = []
    demo_value = []
    demo_label = []
    for i in range(25):
        selection = random.choice(generator_mapping)
        while selection[1] == generators.placeholder_generator:
            selection = random.choice(generator_mapping)

        field = random.choice(selection[0])
        value = selection[1](field)
        demo_field.append(field)
        demo_value.append(value)
        demo_label.append(1)

        for _ in range(4):
            value = generators.generate_another_value(selection[1], field)
            demo_field.append(field)
            demo_value.append(value)
            demo_label.append(0)

    synthesized_demo_data = pd.DataFrame({"id": list(range(25 * 5)), "question1": demo_field, "question2": demo_value, "is_duplicate": demo_label})
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(synthesized_demo_data)
        print(synthesized_demo_data[["is_duplicate"]].describe())
    synthesized_demo_data.to_csv("/home/mohammed-alaa/Downloads/synthesized demo data.csv")

else:
    fields = []
    values = []
    labels = []
    for i in range(450000):

        selection = random.choice(generator_mapping)
        while selection[1] == generators.placeholder_generator:
            selection = random.choice(generator_mapping)

        field = random.choice(selection[0])
        label = random.random() > .5
        if label:
            value = selection[1](field)
        else:
            value = generators.generate_another_value(selection[1], field)
        if visualize_samples:
            print(field, ":", value)
            time.sleep(1)

        fields.append(field)
        values.append(value)
        labels.append(int(label))

    if not visualize_samples:
        valid_split = int(len(fields) * .05)
        synthesized_data = pd.DataFrame({"question1": fields[:valid_split], "question2": values[:valid_split], "is_duplicate": labels[:valid_split]})
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(synthesized_data.head())
            print(synthesized_data[["is_duplicate"]].describe())
        synthesized_data.to_csv("/home/mohammed-alaa/Downloads/synthesized test data.csv")

        # synthesized_data = pd.DataFrame({"question1": fields, "question2": values, "is_duplicate": labels})
        synthesized_data = pd.DataFrame({"question1": fields[valid_split:], "question2": values[valid_split:], "is_duplicate": labels[valid_split:]})
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(synthesized_data.head())
            print(synthesized_data[["is_duplicate"]].describe())
        synthesized_data.to_csv("/home/mohammed-alaa/Downloads/synthesized train data.csv")
