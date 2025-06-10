
import random
import string

def generate_code(prefix, length=8):
    chars = string.ascii_uppercase + string.digits
    return prefix + ''.join(random.choices(chars, k=length))

prefix = 'TPHCGP-'
total_codes = 2100000
unique_codes = set()

while len(unique_codes) < total_codes:
    unique_codes.add(generate_code(prefix))

codes = list(unique_codes)

# Save to 3 separate files with 7 lakh codes each
for i in range(3):
    with open(f'gpay_coupons_batch_{i+1}.txt', 'w') as file:
        for code in codes[i*700000 : (i+1)*700000]:
            file.write(code + '\n')