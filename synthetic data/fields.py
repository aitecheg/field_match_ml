import re

fields = open("original_fields.txt").readlines()

print("original fields", len(fields))
fields = list(map(lambda item: item.strip().lower(), fields))
fields = set(fields)
print("unique fields", len(fields))

fields_containing_dates = set()
fields_containing_compound_names = set()  # perfect
fields_containing_names = set()
fields_containing_SSN = set()
fields_containing_gender = set()
fields_containing_mail = set()
fields_containing_fax = set()
fields_containing_tele = set()
fields_containing_addresses = set()  # perfect
fields_containing_icd = set()  # perfect
fields_containing_dsm = set()  # perfect
fields_containing_MI = set()
fields_containing_zip = set()
fields_containing_yes_no = set()
fields_containing_signature = set()  # leave me
fields_containing_mailing_address = set()
fields_containing_state = set()
fields_containing_city = set()
fields_containing_tax_id = set()
fields_containing_hospitals = set()
fields_containing_num_hours = set()  # perfect
fields_containing_length_height_weight = set()  # perfect
fields_containing_relation = set()  # perfect
fields_containing_burn_deg = set()  # perfect
fields_containing_when_to_back = set()  # perfect
fields_containing_numbers = set()  # perfect
fields_containing_diag = set()
fields_with_no_association = set()
fields_containing_cpt = set()
fields_containing_selection = set()
fields_containing_bank_info = set()
fields_containing_bank_account = set()
fields_containing_policy_numbers = set()
fields_containing_treatments = set()
fields_containing_restrictions_limitations = set()
fields_containing_procedure_codes = set()
fields_containing_symptoms = set()

fields_needs_description = set()  # perfect
fields_containing_name_word = set()
fields_containing_date_word = set()

tryme = set()
for field in fields.copy():
    withins = re.findall(r'(?<=\()(.+?)(?=$)', field)
    score = 0
    if len(withins) > 0:
        for within in withins:
            score = max(score, int("last" in within) + int("first" in within) + int("mi" in within) + int("suffix" in within) + int("middle" in within))
    if "(mm/dd/yy" in field or "(mm/dd/yyyy)" in field:
        fields_containing_dates.add(field)
        fields.remove(field)
    elif score > 1:
        fields_containing_compound_names.add(field)
        fields.remove(field)
    elif "date" in field:
        fields_containing_date_word.add(field)
        fields.remove(field)
    elif "name" in field:
        fields_containing_name_word.add(field)
        fields.remove(field)
    elif "social security" in field or "ssn" in field:
        fields_containing_SSN.add(field)
        fields.remove(field)
    elif "gender" in field:
        fields_containing_gender.add(field)
        fields.remove(field)
    elif "icd" in field:
        fields_containing_icd.add(field)
        fields.remove(field)

    # elif "mi" in field:
    #     fields_containing_MI.add(field)
    #     fields.remove(field)

# print("\n".join(fields_containing_SSN))
# exit()
for field in fields_containing_dates.copy():
    normalized = re.sub("\(mm/dd/yy\)|\(mm/dd/yy|\"\(mm/dd/yyyy\)", "", field)
    for candidate_date in fields_containing_date_word.copy():
        if normalized.strip() in candidate_date and len(candidate_date) < len(normalized.strip()) + 20:
            fields_containing_dates.add(candidate_date)
            fields_containing_date_word.remove(candidate_date)


def transfer_list(source_set, dist_set, list_of_fields):
    for manual_field in list_of_fields:
        manual_field = manual_field.lower().strip()
        # assert manual_field in source_set
        try:
            assert manual_field in source_set
        except Exception as e:
            print()
        if manual_field in source_set:
            source_set.remove(manual_field)
            dist_set.add(manual_field)


transfer_list(fields_containing_date_word, fields_containing_dates, ["inception date", "signed date", "(date signed)", "date signed", "if yes, indicate date below", "effective date", "if yes, please provide dates of treatment:", "final date of treatment", "date of follow up visit following confinement or outpatient surgery", "dates of confinement in intensive care, including coronary care unit: from", "b) date of first visit regarding current condition?", "c) date patient caused work beacuse of condition?", "submission date", "date", "treatment dates", "dates of inpatient hospital confinement: from", "dates of confinement in intensive care, including coronary care unit:", "date of accident", "date of office visit following confinement or outpatient surgery", "b) date of last examination"])
fields = set.union(fields, fields_containing_date_word)

transfer_list(fields, fields_containing_mail, ["preferred email address (for confirmation purposes only)", "email address", "email address:", "preferred e-mail address (for confirmation purposes only)", ])
transfer_list(fields, fields_containing_names, ["my spouse:", "my spouse", "suffix"])
transfer_list(fields, fields_containing_fax, ["fax:", "fax number", "fax", "fax no."])
transfer_list(fields, fields_containing_dates, ["to"])
transfer_list(fields, fields_containing_MI, ["mi"])
transfer_list(fields, fields_containing_signature, ["c. signatue of attending physician", "e. signature of attending physician or provider of service", "signature of physician", "employee (applicant) signature", "insured's signature", "h. signature of insured", "d. signature of individual", "patient signature", "insured/patient signature", "signature of individual", "claimant signature", "physician signature", "(claimant signature)", "signature", "d. signature of attending physician", "signature of doctor", "policyholder/employee signature", "c. signature of attending physician", ])
transfer_list(fields, fields_containing_tele, ["cellular no:", "work phone #", "home phone #", "home phone number", "phone number", "work phone number", "tel (w):", "tel (h):", "(telephone number)", "home telephone number", "telephone number", "telephone number where we can reach you", "telephone no.", "patient telephone number", "cellular telephone number", "employer telephone number", "work telephone number", "treating physician telephone number"])
transfer_list(fields, fields_containing_zip, ["zip", "zip code"])
transfer_list(fields, fields_containing_yes_no,
              ["has the patient been hospitalized", "e) has the patient been treated for the same/similar condition in the past?", "has the patient received any chiropractic, physical, occupational and/or speech therapy?", "has the patient been treated for the same or a similar condition by any physician in the past?", "i authorize unum to leave message about my claim on my voicemail/answering machine.", "has the patient been treated for the same or a similar condition by another physician in the past?", "has the patient been treated for the same/similar condition in the past?", "has the patient been hospitalizes?", "if yes, have you received workers' compensation benefits for your occupational injury?", "i authorize unom to leave messages about my claim on my voicemail / answering machine.", "i authorize unum to leave message about my claim on my voicemail / answering machine.", "c) has patient been hospitalized?",
               "f) is the patient's condition due to injury or sickness involving the patient's employment?", "if \"no, \" is your spouse a u.s. citizen?",
               "3. is your condition work related?",
               "6. have you returned to work?", "has the patient been hospitalized?", "if yes, have you filed a workers' compensation claim?", "were these any complications causing your patient to stop working prior to her expected delivery date?",
               "were there any complications causing your patient to stop working prior to her expected delivery date?", "were you at work at the time of your accident?",
               "were you at work (working for wage or profit) at the time of your accident?",
               "were there any complications causing you to stop work prior to your expected delivery date?", "d) did you advise patient to cease work?", "did you advice your patient to stop working?", "do you work for another employer?",
               "do you support your patient's return to work within the restrictions and limitations you provided?", "does the spouse live in the u.s.?", "does the patient have permanent restrictions and limitations?", "have you advised the patient to return to work?",
               "have you advise the patient to return to work?", "have you advised the patient to work?", "e) is the patient still under your care?", "4. have you been hospitalized?", "have you stopped working?", "was this a motor vehicle accident?", "was surgery performed?", "was surgery performed", "did you advice the patient to stop working?", "did you advise your patient to stop working?", "is the patient's condition due to injury or sickness involving the patient's employment?", "is the patient permanently disabled?", "is this condition to result of an accidental injury?", "is your claim pending a workers' compensation decision?", "is this condition the result of an accident injury", "is the patient still under your care?", "is the patient's condition related to his/her employment?", "is the patient's condition work related?", "is this condition the result of his/her employment", "are you currently self-employed?",
               "are you, the physician, related to this patient?", "are you related to this patient?", "are you actively at work?", "are you related to the patient?", "are there any cognitive deficits or psychiatric conditions that impact function?", ])

transfer_list(fields, fields_containing_mailing_address, ["mailing address", "mailing address (street, city, state, zip)", ])
transfer_list(fields, fields_containing_state, ["the state in which you work", "state", ])
transfer_list(fields, fields_containing_tax_id, ["tax id", "physician's tax id number:", "physician tax id number", "physician's tax id number", ])
transfer_list(fields, fields_containing_city, ["city", ])
transfer_list(fields, fields_containing_num_hours, ["part-time hours per week", "hours per day", "scheduled number of work hours/week", "part-time hours per day", "number of hours worked on date last worked", ])
transfer_list(fields, fields_containing_dates, ["from", ])
transfer_list(fields, fields_containing_length_height_weight, ["patient's weight", "weight", "if related to a laceration. please indicate the length:", "a) height", "patient's height", "patient height", "height", ])
transfer_list(fields, fields_containing_relation, ["i signed on behalf of the claimant as", "i signed on behalf of claimant as", "if yes, what is the relation?", "i signed on behalf of the insured, as", "patient relationship to insured:", "i signed on behalf of the insured as", "if yes, what is the relationship?", "if yes, what is the relationship", "if claim is for a child, please state your relationship to the child", "relationship", "relationship to insured/policyholder (check one)", ])
transfer_list(fields, fields_containing_burn_deg, ["if related to a burn. please indicate the degree:", ])
transfer_list(fields, fields_containing_when_to_back, ["a) when do you expect improvement in the patient's capabilities?", "if you have not returned to work, when do you expect to return?", "when do you expect the patient to return to work?", "if yes, when?", ])
transfer_list(fields, fields_containing_numbers, ["employee id/payroll #", "customer number", "id number:"])

transfer_list(fields, fields_containing_policy_numbers, ["policy number", "policy number:", "policy #", "accident policy number"])
transfer_list(fields, fields_containing_bank_account, ["account number:", "personal account number"])

transfer_list(fields, fields_containing_diag, ["what is the name of your medical condition?",
                                               "diagnosis and treatment", "what is the primary diagnosis preventing the patient from working?", "secondary diagnosis", "what diagnostic or clinical findings support your diagnosis?", "a) what is the primary diagnosis preventing your patient from working?", "diagnosis:", "secondary diagnosis:", "diagnosis remarks", "diagnosis", "what diagnostic or clinical findings support your patient's work restrictions and limitations?", "primary diagnosis", "diagnostic procedure codes/description", ])

transfer_list(fields_containing_name_word, fields_containing_names, ["name of employee/policyholder", "(name)", "patient name", "full name of primary doctor", "name of doctor", "name of patient", "first name", "print or type name", "treating physician name", "provider name", "if yes, employer name", "(print name)", "Printed name", "name of patient (if not self)", "full name of treating doctor", "employer name", "full name of referring doctor/hospital", "claimant name", "name", "policyholder/employer's name", "employee name", "last name", "primary care physician name"])
fields = set.union(fields, fields_containing_name_word)

transfer_list(fields, fields_containing_addresses, ["st/po box", "home address (street/po box)", "street address", "hospital address", "address details:", "address", "home address", "address:", "name/address of facility", ])

transfer_list(fields, fields_containing_hospitals, ["information about your doctor(s) and/or hospital (please print)", "hospital", "hospital name", ])
transfer_list(fields, fields_containing_dsm, ["dsm-iv i", ])
transfer_list(fields, fields_containing_procedure_codes, ["procedure/procedure code", ])

transfer_list(fields, fields_containing_cpt, ["cpt code", "cpt code:", "d) was surgery performed? cpt 4 code(s)", ])
transfer_list(fields, fields_containing_selection, ["d) patient's ability to perform (please check)", "b) patient's ability to: (please check)", "c) patients ability to lift/carry (please check)", "language preference", "please check all types of coverage you have with unum", "a) patient's ability to: (please check number of hours per workday and how often)", "if known, please check all types of coverage you have with unum.", "please check all types of coverage you have with unum.", "c. information about the patient (if different from insured/policyholder) check one", "please check the type of claim you are filing", ])
transfer_list(fields, fields_containing_bank_info, ["bank transit/routing number", "banking details:", "bank name:", "banking details", ])
transfer_list(fields, fields_containing_symptoms, ["c) describe reported symptoms", "what symptoms is your patient reporting about his/her condition?", "b) describe reported symptoms", ])

transfer_list(fields, fields_with_no_association, ["v", "ii", "iv", "iii", "employee/policyholder information", "b. information about how to set-up or change your direct deposit", "please verify the transit routing number with your bank.", "if this claim is related to normal pregnancy, please provide the following", "d. information about your condition", "e. complete this section for accidental injury claims", "attending physician or provider of service statement (continued)", "attending physician statement", "b. complete this section for disability claims only.", "if yes, please provide the following", "a routing number beginning with the number 5 is not valid", "part i: to be completed by patient", "a. information about you", "c. information about your medical providers", "this claim is for", "b. information about your disability", "insured/patient statement (please print)", "insured/patient statement (continued)", "employee statement (continued)",
                                                   "please provide copies of all test results, operative reports, pathology reports, and/or your detailed medical statement related to the service provided to the patient.", "employee statement (please print)", "if your hospital bill does not contain this information, please ask your doctor to complete the attending physician statement (pages 8-10 of this form.)", "attending physician or provider of service statement (please print)", "a. attending physician's statement (please print)", "attending physician statement (continued)", "attending physician statement (please print)",
                                                   "a. complete this section for accident claims only.", "section 4: employee (applicant) statements", "a. complete this section for pregnancy, then go to section c", "d. complete this section for hospital confinement/intensive care claims.", "section 1: employee (applicant) information - always complete", "b. information about the insured/policyholder",
                                                   "b. complete this section for all conditions except pregnancy, then go to section c", "section 3: coverage information", "to be completed by physician or treating provider", "d. complete this section for inpatient/outpatient surgery claims (please refer to place of service codes above)", "section a. general information", "b. complete this section for diagnostic testing claims", "a. complete this section for all medical conditions", "part i: to be completed by insured/patient", "please complete this section if you are canceling your direct deposit agreement", "c. complete this section for hospital confinement/intensive care benefit claims.", "f. information about physicians and hospitals", "part ii: to be completed by physician or treating provider", "b. information about the insured", "e. information about physician", "c. information about the patient", "bank/financial institution information", "c. fuctional capacity",
                                                   "to be completed by attending physician or treating provider", "voluntary benefits cancer/critical illness insurance", "if this claim is related to normal pregnancy, please provide the following:", "c. complete this section for emergency room and/or hospital/icu confinement claims (please refer to place of service codes above)", "a. complete this section for normal pregnancy, then go to section c", "c. direct deposit cancellation request", "patient information", "section c. hospital confinement, intensive care benefit", "b. complete this section for all conditions except pregnancy, then go to sextion c", "part ii: to be completed by attending physician or treating provider", "section b. accidental injury", "b. complete this section for all conditions except normal pregnancy", "c. functional capacity", "section 2: spouse information - complete only if applying for spouse coverage", ])
transfer_list(fields, fields_containing_treatments, ["procedure", "if yes, what procedure was performed?", "surgical procedure", "b) medications (please list all medications including dosage and frequency)", "medications (please attach medication log)", "a) describe the patient's current treatment program (include facilities name/address if applicable)", "what is your treatment plan? please include all medications", "what is your treatment plan? please include all medications.", "treatment", "what is your treatment plan?", ])
transfer_list(fields, fields_containing_restrictions_limitations, ["restrictions and/or limitations", "b) restrictions (activities patient should not do)", "if your patient has current restrictions (activities patient should not do) and/or limitations (activities patient cannot do) list below", "(activities patient cannot do), please initial here", "if yes, please provide restrictions and limitations:", "please provide the duration of these restrictions and limitations.", "current restrictions (activities patient should not do)", "if yes, please list the permanent restrictions and limitations.", "if no, please indicate the restrictions and limitations that prevent the patient from returning to work in the space provided below.", "current restrictions (activities patient should not do) and current limitations (activities patient cannot do). please be specific and understand tht a reply of \"no work\" or \"totally incapacitated\" will not enable us to evaluate the claim for benefits",
                                                                   "if your patient has current restrictions (activities patient should not do) and/or limitations (activities patient cannot do) list below.", "c) limitations (activities patient cannot do)", "current limitations (activities patient cannot do)", ])
transfer_list(fields, fields_needs_description,
              [
                  "medical specialty", "degree/specialty", "specialty",
                  "if yes what is the recommended frequency of treatment?",
                  "electronically signed indicator",

                  "name, specialty, address, phone #, fax #, treatment from, to", "address (street, apt. #, city, state, zip)",
                  "life insurance",

                  "delivery type:", "what type of delivery?", "c) delivery type", "delivery type",
                  "part time", "full time",
                  "all other conditions",
                  "risk",
                  "place of service codes",
                  "av order / other",
                  "time of accident",
                  "(name/relationship)",
                  "[optional employee selected benefit]", "employer selected benefit[s]", "cancel my direct deposit agreement",
                  "contact details", "ain mgmt w/", "name, specialty, address, phone #", "through", "other details:", "c) describe physical findings (mris, x-rays, emg/ncv studies, lab test, clinical findings, gaf etc.)", "other conditions (please attach additional information as necessary)", "return to work assessment",
                  "if related to an injury, when, where and how did the injury occur?", "change direct deposit account", "normal pregnancy", "sos services:", "2. for other than pregnancy, is your disability caused by",

                  "individual disability", "physical capabilities", "ient referral",
                  "if no, when do you expect improvement in the patient's functional capacity?",
                  "primary beneficiary", "contingent beneficiary",
                  "unknown", "policy endorsements:",

                  "choose type of account - note: we are only able to deposit benefit payments into one account.", "facility name",
                  "short term disability", "long term disability",
                  "agent/insp:", "the above statements are true and complete to the best of my knowledge and belief.",
                  "voluntary benefits disability", "voluntary benefits medsupport insurance", "dates of inpatient hospital confinement", "claim event identifier",
                  "(name / relationship)",
                  "supplies/casting order", "group accident",
                  "next oppointment",
                  "occupation",
                  "set-up direct deposit", "mri", "certification",
                  "if related to a fracture or dislocation, please indicate:", "branch:", "degree",

                  "accident details", "(if the patient received multiple tests, please provide dates and locations in an attached doument)", "d) describe physical findings (mris, x-rays, emg/ncv studies, lab test, clinical findings, gaf etc.)", "tell us how your accident happened: (if you need more space, you may attach oa separate place pf paper.)", "other providers: are you aware or have you referred your patient to other treating providers? if yes, please provide complete name, contact information and specialty of any other treating physicians", "if yes, please explain how the work related injury/illness occured", "please explain how your accident happened. (if you need more space, please attach a separate sheet of paper).", "other providers: please supply complete name, contact information and specialty of any other treating physicians or hospital.", "if yes, please describe",
                  "please attach an itemized copy of your hospital bill that includes the following information. diagnosis, admission and discharge dates, name of facility and address.", "if yes, please indicate any ongoing restrictions and limitations in the space provided below.",
                  "if your patient has current restrictions (activities patient should not do) and/or limitation (activities patient cannot do) list below. please be specific and understand that a reply of \"no work\" or \"totally disabled\" will not enable us to evaluate your patient's claim for benefits and may result in us having to contact you for clarification.",
                  "other providers: are you aware of or have you referred your patient to other treating providers? if yes, please provide complete name, contact information and specialty of any other treating physicians.", "what are the other conditions that prevent the patient from working?", "are there other conditions that prevent your patient from working? if so, please list with information as follows:", "if yes, please explain",
              ])
# print("\n".join(fields_containing_name_word))


print("remaining unrevised", len(fields))
print("\n\n[\"", "\",\n\"".join(fields), "\"]\n\n", sep="")

total = [fields_containing_dates,
         fields_containing_compound_names,  #
         fields_containing_names,
         fields_containing_SSN,
         fields_containing_gender,
         fields_containing_mail,
         fields_containing_fax,
         fields_containing_tele,
         fields_containing_addresses,
         fields_containing_icd,
         fields_containing_MI,
         fields_containing_zip,
         fields_containing_yes_no,
         fields_containing_signature,
         fields_containing_mailing_address,  #
         fields_containing_state,
         fields_containing_city,
         fields_containing_tax_id,
         fields_containing_hospitals,
         fields_containing_num_hours,  #
         fields_containing_length_height_weight,
         fields_containing_relation,
         fields_containing_burn_deg,
         fields_containing_when_to_back,
         fields_containing_numbers,  #
         fields_containing_diag,
         fields_with_no_association,
         fields_containing_cpt,
         fields_containing_selection,
         fields_containing_bank_info,
         fields_containing_bank_account,
         fields_containing_policy_numbers,
         fields_containing_dsm,
         fields_containing_treatments,
         fields_containing_restrictions_limitations,
         fields_containing_procedure_codes,
         fields_containing_symptoms
         ]


print("total revised fields", sum(list(map(lambda item: len(item), total))))

# print(len(fields))
# import Levenshtein # https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html

import pickle

with open("./buckets_of_fields.pkl", 'wb') as f:
    pickle.dump(total, f)  # dumps get the string
    # cities,addresses,hospitals,codes,description
