import pickle
import random
import string

from faker import Faker

fake = Faker()
generators = []
others = {}

with open("./providers.pkl", 'rb') as f:
    cities, addresses, hospitals_names, hospital_address, icd_codes, diagnosis, cpt_codes, treatments, dsmv_codes, dsmv_description, zip_codes = pickle.load(f)


def generate_another_value(generator, field):
    selection = random.choice(others[generator.__name__])
    return selection(field)


##############################################################
def address_generator(field):
    # return (fake.secondary_address())
    # return(fake.address())
    if (random.random()) > .8:
        return fake.street_address()
    else:
        return random.choice(addresses)


generators.append(address_generator)


##############################################################

def city_generator(field):
    return random.choice(cities)


generators.append(city_generator)


##############################################################

def signature_generator(field):
    return "'S'"


generators.append(signature_generator)


##############################################################

def none_generator(field):
    return "ø"


generators.append(none_generator)


##############################################################

def procedure_desc_generator(field):
    return random.choice(treatments)


generators.append(procedure_desc_generator)


##############################################################

def cpt_generator(field):
    return random.choice(cpt_codes)


generators.append(cpt_generator)


##############################################################

def dsm_generator(field):
    return random.choice(dsmv_codes)


generators.append(dsm_generator)


##############################################################
def state_generator(field):
    if (random.random()) > .5:
        return fake.state_abbr()
    return fake.state()


generators.append(state_generator)


##############################################################
def zip_generator(field):  # 15025 60644 35203
    if (random.random()) > .5:
        return str(random.choice(zip_codes))
    return fake.zipcode()


generators.append(zip_generator)


##############################################################

def fax_generator(field):
    num10 = str(fake.random_number(digits=10, fix_len=True))
    num10 = list(num10[0:3] + " " + num10[3:6] + " " + num10[6:])

    if (random.random()) > .5:
        num10[3] = "-"
    if (random.random()) > .5:
        num10[7] = "-"

    num10 = "".join(num10)
    paren = (random.random()) > .5
    return "(" + num10[0:3] + ")" + num10[3:] if paren else num10


generators.append(fax_generator)


##############################################################

def ssn_generator(field):
    replace = (random.random()) > .5
    ssn = fake.ssn().replace("-", " ") if replace else fake.ssn()
    replace = (random.random()) > .6
    ssn = ssn.replace(" ", "") if replace else ssn
    replace = (random.random()) > .9
    return ssn[:4] + str(random.randint(0, 9)) + ssn[4:] if replace else ssn


generators.append(ssn_generator)


##############################################################
def tel_generator(field):  # home telephone number cellular telephone number work phone #
    return fax_generator(field)


generators.append(tel_generator)


##############################################################
def tax_id_generator(field):
    num9 = str(fake.random_number(digits=9, fix_len=True))
    if (random.random()) > .5:
        num9 = num9[:2] + "-" + num9[2:]
    return num9


generators.append(tax_id_generator)


##############################################################

def hospitals_name_generator(field):
    return random.choice(hospitals_names)


generators.append(hospitals_name_generator)


##############################################################
def mi_generator(field):
    return random.choice(string.ascii_lowercase)


generators.append(mi_generator)


##############################################################

def icd_generator(field):
    icd = random.choice(icd_codes)

    return icd[:3] + "." + icd[3:] if len(icd) > 3 else icd


generators.append(icd_generator)


##############################################################
def date_generator(field):
    YYYY = ["%m-%d-%Y", "%m %d %Y", "%m/%d/%Y"]
    YY = ["%m/%d/%y", "%m-%d-%y", "%m %d %y"]
    if "yyyy" in field:
        date = fake.date(random.choice(YYYY))
    elif "yy" in field:
        date = fake.date(random.choice(YY))
    else:
        date = fake.date(random.choice(YYYY + YY))

    return date


generators.append(date_generator)


##############################################################

def name_generator(field):
    if "first" in field:
        value = fake.first_name()
    elif "last" in field or "suffix" in field or "middle" in field:
        value = fake.last_name()
    else:
        value = fake.name()
    return value


generators.append(name_generator)


##############################################################
def gender_generator(field):
    return random.choice(["m", "male", "f", "female"])


generators.append(gender_generator)


##############################################################

def mail_generator(field):
    return random.choice([fake.ascii_company_email, fake.ascii_email, fake.ascii_free_email, fake.ascii_safe_email, fake.company_email, fake.email, fake.free_email, fake.free_email_domain])()


generators.append(mail_generator)


##############################################################
def yes_no_generator(field):
    return random.choice(["yes", "no"])


generators.append(yes_no_generator)


##############################################################
def diagnose_generator(field):
    return random.choice(diagnosis)


generators.append(diagnose_generator)


##############################################################
def hospital_address_generator(field):
    return random.choice(hospital_address)


# i wont add this
##############################################################

def placeholder_generator():
    raise NotImplementedError()


for index, generator in enumerate(generators):
    others[generator.__name__] = generators[:index] + generators[index + 1:]

if __name__ == "__main__":
    print("||".join([hospital_address_generator("") for _ in range(50)]))
