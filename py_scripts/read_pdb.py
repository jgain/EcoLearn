import numpy as np
import copy

class plant:
    def __init__(self, x, y, z, h, r):
        self.x = x
        self.y = y
        self.z = z
        self.h = h
        self.r = r

    def __eq__(self, other):
        return self.x == other.x and \
                self.y == other.y and \
                self.z == other.z and \
                self.h == other.h and \
                self.r == other.r
    
    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "xyz: {} {} {}, h: {}, r: {}".format(self.x, self.y, self.z, self.h, self.r)

class pdb_content:
    def __init__(self, pdb_filename, width=None, height=None, normalize_distances=True):
        self.normalize_distances = normalize_distances
        self.meter_per_ft = 0.3048
        if normalize_distances:
            self.meters_per_unit = self.meter_per_ft * 3    # pdb units are in meters, so we divide to get it into the 3ft units we want
        else:
            self.meters_per_unit = 1
        self.infile = open(pdb_filename, "r")
        self.width = width
        self.height = height
        self.species = []
        self.read_all()

    def read_nspecies(self):
        line = next(self.infile)
        self.nspecies = int(line)

    def get_species_ids(self):
        speciesids = set()
        for specdict in self.species:
            speciesids.add(specdict["spec_id"])
        return speciesids

    def mult_heights(self, mult):
        for plnt in self.all_plants():
            plnt.h *= mult

    def mult_radii(self, mult):
        for plnt in self.all_plants():
            plnt.r *= mult

    def get_metadata_on_species(self):
        line = next(self.infile)
        line = line.split(" ")
        spec_idx = int(line[0])
        minh = float(line[1])
        maxh = float(line[2])
        #line = next(self.infile)
        avg_canopy_to_height = float(line[3])
        line = next(self.infile)
        nplants = int(line)
        spec_dict = {"spec_id": spec_idx, "minh": minh, "maxh": maxh, "average_canopy_to_height": avg_canopy_to_height, "numplants": nplants, "plants": []}
        return spec_dict

    def keepplants(self, startx, endx, starty, endy, adapt_pos=False):
        
        def within(x, y):
            return x >= startx and x <= endx and y >= starty and y <= endy
        
        def calc_stats(spdict):
            nplants = len(spdict["plants"])
            minh, maxh = float("inf"), -float("inf")
            avgh, avgr = 0.0, 0.0
            if len(spdict["plants"]) > 0:
                for plnt in spdict["plants"]:
                    if plnt.h > maxh:
                        maxh = plnt.h
                    if plnt.h < minh:
                        minh = plnt.h
                    avgh += plnt.h
                    avgr += plnt.r
                avgrtoh = avgr / avgh
                spdict["minh"] = minh
                spdict["maxh"] = maxh
                spdict["average_canopy_to_height"] = avgrtoh
                spdict["numplants"] = nplants
            else:
                spdict["minh"] = 0.0
                spdict["maxh"] = 0.0
                spdict["average_canopy_to_height"] = 0.0
                spdict["numplants"] = nplants
                

        for spdict in self.species:
            rm_idxes = []
            for pidx in range(len(spdict["plants"])):
                plnt = spdict["plants"][pidx]
                if not within(plnt.x, plnt.y):
                    rm_idxes.append(pidx)
                elif adapt_pos:
                    spdict["plants"][pidx].x -= startx
                    spdict["plants"][pidx].y -= starty
            rm_idxes.reverse()
            for i in rm_idxes:
                del spdict["plants"][i]
            calc_stats(spdict)
            

    def get_plants_for_species(self, spec_dict):
        nplants = spec_dict["numplants"]
        for _ in range(nplants):
            line = next(self.infile)
            line = line.split(" ")
            x, y, z = (float(l) / self.meters_per_unit for l in line[:-2])
            h, r = float(line[-2]) / self.meters_per_unit, float(line[-1]) / self.meters_per_unit
            spec_dict["plants"].append(plant(x, y, z, h * self.meters_per_unit, r))

    def get_next_species(self):
        spec_dict = self.get_metadata_on_species()
        self.get_plants_for_species(spec_dict)
        self.species.append(spec_dict)

    def read_all(self):
        self.read_nspecies()
        for _ in range(self.nspecies):
            self.get_next_species()

    def all_plants(self, add_species=True):
        for sp in self.species:
            for plnt in sp["plants"]:
                plnt.specie = sp["spec_id"]
                yield plnt

    def get_nplants_total(self):
        nplants = 0
        for sp in self.species:
            nplants += sp["numplants"]
        return nplants

    def get_nplants_species(self, spec_id):
        for sp in self.species:
            if sp["spec_id"] == spec_id:
                return sp["numplants"]
        return None

    def all_species_ids(self):
        for sp in self.species:
            yield sp["spec_id"]

    def get_all_species_ids(self):
        return [idx for idx in self.all_species_ids()]

    def write_pdb(self, filename):
        with open(filename, "w+") as outfile:
            nspec_str = "{}\n".format(self.nspecies)
            outfile.write(nspec_str)
            for specdict in self.species:
                specinfo_writestr = "{} {} {} {}\n".format(specdict["spec_id"], specdict["minh"], specdict["maxh"], specdict["average_canopy_to_height"])
                outfile.write(specinfo_writestr)
                nplants_writestr = "{}\n".format(specdict["numplants"])
                outfile.write(nplants_writestr)
                plants = specdict["plants"]
                for p in plants:
                    pstr = "{} {} {} {} {}\n".format(p.x * self.meters_per_unit, \
                            p.y * self.meters_per_unit, \
                            p.z * self.meters_per_unit, \
                            p.h, \
                            p.r * self.meters_per_unit)
                    outfile.write(pstr)

    def test_write_pdb(self, output_fname, delete_after=True):
        self.write_pdb(output_fname)
        pdb = pdb_content(output_fname, normalize_distances=self.normalize_distances)
        success = pdb.species == self.species
        if not success:
            print("nspecies: {} {}".format(pdb.nspecies, self.nspecies))
            for s1, s2 in zip(pdb.species, self.species):
                print({key: val for key, val in s1.items() if key != "plants"})
                print({key: val for key, val in s2.items() if key != "plants"})
                print("-----------------------------------------")
                for p1, p2 in zip(s1["plants"], s2["plants"]):
                    if p1 != p2:
                        print("Plants not equal: {} | {}".format(p1, p2))
        else:
            print("PDB files equal")
        """
        if (not success):
            print("write_pdb function failed")
            print(pdb.species)
            print(self.species)
        """

    def __del__(self):
        self.infile.close()

if __name__ == "__main__":
    testobj = pdb_content("/home/konrad/PhDStuff/data/analyse_specassign0/analyse_specassign0_canopy0.pdb", normalize_distances=False)
    testobj.test_write_pdb("/home/konrad/PhDStuff/data/analyse_specassign0/testpdbwrite.pdb", False)
