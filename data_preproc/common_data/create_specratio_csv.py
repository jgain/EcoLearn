def create_specratios():
    class model:
        def __init__(self, vueid, hmin, hmax, prob, modheight, modname, rotstr, whratio):
            self.vueid = vueid
            self.hmin = hmin
            self.hmax = hmax
            self.prob = prob
            self.modheight = modheight
            self.modname = modname
            self.rotstr = rotstr
            self.whratio = whratio

        def __repr__(self):
            return "vueid: {}, hmin: {}, hmax: {}, prob: {}, modheight: {}, modname: {}, rotstr: {}, whratio: {}".format(self.vueid, \
                    self.hmin, self.hmax, self.prob, self.modheight, self.modname, self.rotstr, self.whratio)

        def __str__(self):
            return self.__repr__()

        def get_aslist(self):
            return [self.vueid, self.hmin, self.hmax, self.prob, self.modheight, self.modname, self.whratio]

    class specie:
        def __init__(self, specid, modlist):
            self.specid = specid
            self.models = []
            for mod in modlist:
                self.models.append(model(*mod[1:]))

        def __repr__(self):
            retstr = "ID: {}\n"
            for mod in self.models:
                retstr += "\t" + str(mod) + "\n"
            retstr = retstr.format(self.specid)
            return retstr

        def __str__(self):
            return self.__repr__()
        
        def get_aslist(self):
            modlist = []
            for m in self.models:
                modlist.append(m.get_aslist())
                modlist[-1].insert(1, self.specid)
            return modlist

    heading = ["vueid PRIMARY", "hmin", "hmax", "prob", "modheight", "modname", "whratio"]
    junction_heading = ["vueid FOREIGN", "Tree ID FOREIGN"]

    tabmodels =  {0:[(0,0,0.000,10.0,1.000,0.210,'Oxalis_oregano_A001','',2.64)],\
     1: [(1,4,0.000,0.500,1.000,0.700,'Artemisia_californica_D001','',1.66),\
     (1,3,0.500,0.750,1.000,0.840,'Artemisia_californica_C001','',1.62),\
     (1,2,0.750,1.000,1.000,0.950,'Artemisia_californica_B001','',1.76),\
     (1,1,1.000,10.000,1.000,1.090,'Artemisia_californica_A001','',1.73)],\
     2:[(2,6,0.000,0.500,0.500,0.560,'Adenostoma_fasciculatum_B001','',3.38),\
     (2,8,0.000,0.500,0.500,0.600,'Adenostoma_fasciculatum_D001','',2.08),\
     (2,5,0.500,3.500,0.500,0.650,'Adenostoma_fasciculatum_A001','',3.88),\
     (2,7,0.500,10.000,0.500,0.750,'Adenostoma_fasciculatum_C001','',2.14)],\
     3:[(3,11,0.000,0.800,1.000,0.710,'Polystichum_munitum_C001','',2.12),\
     (3,10,0.800,1,1.000,0.790,'Polystichum_munitum_B001','',2.44),\
     (3,9,1.000,10.000,1.000,0.820,'Polystichum_munitum_A001','',2.78)],\
     4:[(4,14,0.000,1.000,1.000,1.220,'Arctostaphylos_manzanita_C001','',1.55),\
     (4,13,1.000,2.000,1.000,1.440,'Arctostaphylos_manzanita_B001','',1.67),\
     (4,12,2.000,40.000,1.000,1.640,'Arctostaphylos_manzanita_A001','',1.59)],\
     5:[(5,17,0.000,1.000,1.000,0.910,'Heteromeles_arbutifolia_C001','',1.43),\
     (5,16,1.000,1.500,1.000,1.240,'Heteromeles_arbutifolia_B001','',1.33),\
     (5,15,1.500,90.000,1.000,1.480,'Heteromeles_arbutifolia_A001','',1.34)],\
     6:[(6,18,0.000,2.000,1.000,1.830,'Toxicodendron_diversilobum_A001','',0.35),\
     (6,19,2.000,3.000,1.000,2.350,'Toxicodendron_diversilobum_B001','',0.32),\
     (6,20,3.000,40.000,1.000,2.550,'Toxicodendron_diversilobum_C001','',0.30)],\
     7:[(7,22,0.000,3.000,1.000,2.590,'Notholithocarpus_densiflorus_B001','',1.47),\
     (7,21,3.000,10.000,1.000,3.730,'Notholithocarpus_densiflorus_A001','',1.52),\
     (7,23,10.000,20.000,1.000,7.530,'Notholithocarpus_densiflorus_C001','',0.69),\
     (7,24,20.000,100.000,1.000,8.040,'Notholithocarpus_densiflorus_D001','',1.03)],\
     8:[(8,46,0.000,4.000,1.000,4.5,'Quercus_Agrifolia_Young','',1.66),\
     (8,26,4.000,20.000,1.000,13.400,'Quercus_agrifolia_B001','',1.06),\
     (8,25,20.000,100.000,1.000,16.500,'Quercus_agrifolia_A001','',1.03)],\
     9:[(9,50,0.000,5.000,1.000,5.00,'Sessile_Oak_Young',"<rotate x=\"1\" angle=\"-90\"/>",0.83), \
     (9,28,5.000,20.000,1.000,17.200,'Quercus_garryana_B001','',1.05),\
     (9,27,20.000,100.000,1.000,21.100,'Quercus_garryana_A001','',0.96)],\
     10:[(10,50,0.000,5.000,1.000,5.0,'Sessile_Oak_Young',"<rotate x=\"1\" angle=\"-90\"/>",0.83),\
     (10,30,5.000,25.000,1.000,14.400,'Quercus_kelloggii_B001','',1.33),\
     (10,29,25.000,100.000,1.000,19.400,'Quercus_kelloggii_A001','',1.52)],\
     11:[(11,33,0,5.000,1.000,3.39,'Umbellularia_californica_C001','',1.23),\
     (11,32,5.000,15.000,1.000,6.36,'Umbellularia_californica_B001','',0.96),\
     (11,31,15.000,100.000,1.000,9.14,'Umbellularia_californica_A001','',0.6)],\
     12:[(12,37,0.000,12.000,1.000,8.760,'Arbutus_menziesii_D01','',1.06),\
     (12,36,12.000,16.000,1.000,11.700,'Arbutus_menziesii_C01','',0.94),\
     (12,34,16.000,30.000,0.500,15.100,'Arbutus_menziesii_A01','',1.00),\
     (12,35,16.000,100.000,0.500,14.700,'Arbutus_menziesii_B01','',0.92)],\
     13:[(13,47,0.000,5.000,1.000,4.00,'Bishop_Pine_Young',"<rotate x=\"1\" angle=\"-90\"/>",0.66),\
     (13,39,5.000,25.000,0.500,16.400,'Pinus_sabiniana_A02','',0.60),\
     (13,41,5.000,25.000,0.500,14.600,'Pinus_sabiniana_B02','',0.64),\
     (13,38,25.000,45.000,0.500,21.800,'Pinus_sabiniana_A01','',0.69),\
     (13,40,25.000,100.000,0.500,22.000,'Pinus_sabiniana_B01','',0.52)],\
     14:[(14,49,0.000,5.000,1.000,4.000,'Douglas_Fir_Young',"<rotate x=\"1\" angle=\"-90\"/>",0.53),\
     (14,42,5.000,20.000,1.000,10.000,'Pseudotsuga_menziesii_v2',"<rotate x=\"1\" angle=\"-90\"/>",0.36),\
     (14,43,20.000,100.000,1.000,15.100,'Pseudotsuga_menziesii_v3',"<rotate x=\"1\" angle=\"-90\"/>",0.34)], \
     15:[(15,48,0.000,10.000,1.000,14.500,'Giant_Sequoia_Young',"<rotate x=\"1\" angle=\"-90\"/>",0.60), \
     (15,45,10.000,45.000,1.000,20.400,'Sequoia_sempervirens_B',"<rotate x=\"1\" angle=\"-90\"/>",0.33),\
     (15,44,45.000,110.000,1.000,56.000,'Giant_Sequoia_Adult',"<rotate x=\"1\" angle=\"-90\"/>",0.33)]}

    all_species = []
    all_models = {}
    spec_to_models = {}

    for modid, modlist in tabmodels.items():
        all_species.append(specie(modid, modlist))
        spec_to_models[modid] = []
        for moddetails in modlist:
            m = model(*moddetails[1:])
            mlist = m.get_aslist()
            if moddetails[1] in all_models:
                assert(mlist == all_models[moddetails[1]])
            all_models[moddetails[1]] = m.get_aslist()
            spec_to_models[modid].append(moddetails[1])

    csvstr = ",".join(heading) + "\n"

    for modid, moddetails in all_models.items():
        #print(moddetails)
        assert(len(moddetails) == len(heading))
        csvstr += ",".join(map(str, moddetails)) + "\n"

    jcsvstr = ",".join(junction_heading) + "\n"

    for specid, modlist in spec_to_models.items():
        for m in modlist:
            jcsvstr += "{},{}\n".format(m, specid)

    """
    for spec in all_species:
        modlist = spec.get_aslist()
        for m in modlist:
            assert(len(m) == len(heading))
            csvstr += ",".join(map(str, m)) + "\n"
            print(m)
        print(spec)
    """

    print(csvstr)
    print("---------------------------")
    print(jcsvstr)

    with open("modelDetails.csv", "w+") as outfile:
        outfile.write(csvstr)

    with open("modelMapping.csv", "w+") as outfile:
        outfile.write(jcsvstr)
