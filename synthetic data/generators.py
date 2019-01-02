import pickle
import random
import string

from faker import Faker

fake = Faker()

with open("./providers.pkl", 'rb') as f:
    cities, addresses, hospitals, codes, description = pickle.load(f)


def other_value(generator, field):
    selection = random.choice(others[generator.__name__])
    return selection(field)


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


def name_generator(field):
    if "first" in field:
        value = fake.first_name()
    elif "last" in field or "suffix" in field or "middle" in field:
        value = fake.last_name()
    else:
        value = fake.name()
    return value


def ssn_generator(field):
    replace = (random.random()) > .5
    return fake.ssn().replace("-", " ") if replace else fake.ssn()


def gender_generator(field):
    return random.choice(["m", "male", "f", "female"])


def mail_generator(field):
    return random.choice([fake.ascii_company_email, fake.ascii_email, fake.ascii_free_email, fake.ascii_safe_email, fake.company_email, fake.email, fake.free_email, fake.free_email_domain])()


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


def tel_generator(field):
    return fax_generator(field)


def mi_generator(field):
    return random.choice(string.ascii_lowercase)


def zip_generator(field):  # 15025 60644 35203
    return fake.zipcode()


def state_generator(field):
    return fake.state_abbr()


def city_generator(field):
    return random.choice(cities)


def address_generator(field):
    # return (fake.secondary_address())
    # return(fake.address())
    if (random.random()) > .8:
        return fake.street_address()
    else:
        return random.choice(addresses)


def tax_id_generator(field):
    num9 = str(fake.random_number(digits=9, fix_len=True))
    if (random.random()) > .5:
        num9 = num9[:2] + "-" + num9[2:]
    return num9


def yes_no_generator(field):
    return random.choice(["yes", "no"])


def hospitals_generator(field):
    return random.choice(hospitals)


def icd_generator(field):
    icd = random.choice(codes)

    return icd[:3] + "." + icd[3:] if len(icd) > 3 else icd


def diagnose_generator(field):
    return random.choice(description)


def placeholder_generator():
    raise NotImplementedError()


generators = [date_generator, name_generator, ssn_generator, gender_generator, mail_generator, fax_generator, tel_generator, mi_generator, zip_generator, state_generator, city_generator, address_generator, tax_id_generator, yes_no_generator, hospitals_generator, icd_generator, diagnose_generator]
others = {}
for index, generator in enumerate(generators):
    others[generator.__name__] = generators[:index] + generators[index + 1:]

if __name__ == "__main__":
    print(icd_generator(""))
