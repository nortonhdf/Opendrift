from opendrift.models.openoil.adios import get_oil_names
oils = get_oil_names()
keywords = ['MARLIM','RONCADOR','PEREGRINO','JUBARTE','FRADE','ALBACORA','POLVO','BRAZIL','BRASIL','PETROBRAS','HEAVY','BUNKER','CRUDE']
print(f"Total oils: {len(oils)}")
found = [n for n in oils if any(k in n.upper() for k in keywords)]
print(f"\nMatches for keywords ({len(found)}):")
for n in sorted(found):
    print(" ", repr(n))
print("\nFirst 30 oils (alphabetical):")
for n in sorted(oils)[:30]:
    print(" ", repr(n))
