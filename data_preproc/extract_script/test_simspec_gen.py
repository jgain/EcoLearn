import sqlite3
import numpy as np

def generate_random_simspec(nsims, concurrent_subbiomes_perc, sql_db_filename, out_filename):
    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def normalize_dict(d, dname):
        csum = 0
        for key, value in d.items():
            csum += value
        if csum <= 1e-5:
            raise ValueError("Sum of {} values must not be zero or negative".format(dname))
        for key in d:
            d[key] = d[key] / csum

    def create_cumul_dict(d):
        cumul_d = dict(d)
        keys = [k for k in d]
        csum = 0
        for k in keys:
            csum += d[k]
            cumul_d[k] = csum
        return keys, cumul_d

    def sample_from_probdict(probdict, keys_ordered):
        rnum = np.random.uniform()
        for k in keys_ordered:
            if rnum < sub_csums[k]:
                return k
        return None     # this indicates there is a problem with the probability dictionary

    conn = sqlite3.connect(sql_db_filename)
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    biome_ids = []
    for row in cursor.execute("SELECT Sub_biome_ID FROM subBiomes"):
        biome_ids.append(row["Sub_biome_ID"])
    
    subbiomes = {i: [] for i in biome_ids}
    for row in cursor.execute("SELECT Sub_biome_ID, Tree_ID FROM subBiomesMapping WHERE Canopy = 1"):
        subbiomes[row["Sub_biome_ID"]].append(row["Tree_ID"])
    
    sub_cc = concurrent_subbiomes_perc
    normalize_dict(sub_cc, "concurrent_subbiomes_perc")
    keys, sub_csums = create_cumul_dict(sub_cc)

    nsbiomes = None
    for sim_idx in range(nsims):
        nsbiomes = sample_from_probdict(sub_csums, keys)
        assert(nsbiomes is not None)
        sim_subbiomes = list(np.random.choice(keys, size=nsbiomes, replace=False))
        sub_probs = {}
        for sb in sim_subbiomes:
            sub_probs[sb] = np.random.uniform() * 0.6 + 0.2
        normalize_dict(sub_probs, "sub_probs")
        filestr = ""
        for sb, prob in sub_probs.items():
            filestr += str(sb) + " " + str(prob) + " "
        filestr.strip()
        filestr += "\n"
        for sb in sim_subbiomes:
            species = subbiomes[sb]
            probs = np.array([np.random.uniform() * 0.6 + 0.2 for _ in species])
            probs = probs / np.sum(probs)
            probs = list(probs)
            for sp, p in zip(species, probs):
                filestr += str(sb) + " " + str(sp) + " " + str(p) + "\n"
    filestr = filestr[:-1]
    with open(out_filename, "w") as outfile:
        outfile.write(filestr)

generate_random_simspec(1, {1: 0.5, 2: 0.2, 3:0.3}, "/home/konrad/EcoSynth/data_preproc/common_data/sonoma.db", "/home/konrad/simspec_test.txt")
