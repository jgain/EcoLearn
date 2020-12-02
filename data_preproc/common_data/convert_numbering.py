import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("newfile")
arg_parser.add_argument("reffile")
arg_parser.add_argument("changefile")

a = arg_parser.parse_args()

old = {}
new = {}

otonew = {}

keycol_new = -1
namecol_new = -1

with open(a.newfile, "r") as infile:
    for linen, line in enumerate(infile):
        line = line.split(",")
        line = [s.strip() for s in line]
        #line = [s for s in line if len(s) > 0]
        if linen == 0:
            for coli in range(len(line)):
                if line[coli] == "Tree ID PRIMARY":
                    keycol_new = coli
                elif line[coli] == "common name":
                    namecol_new = coli
        else:
            new[line[namecol_new]] = int(line[keycol_new])

with open(a.reffile, "r") as infile:
    for linen, line in enumerate(infile):
        line = line.split(",")
        line = [s.strip() for s in line]
        #line = [s for s in line if len(s) > 0]
        if linen == 0:
            for coli in range(len(line)):
                if line[coli] == "key":
                    keycol_old = coli
                elif line[coli] == "common name":
                    namecol_old = coli
        else:
            print("name: {} {}, {}".format(line[namecol_old], line[keycol_old], linen))
            try:
                old[line[namecol_old]] = int(line[keycol_old])
            except ValueError:
                pass

for key, val in old.items():
    print("{}, {}".format(key, val))
    if key in new:
        otonew[val] = new[key]
    else:
        otonew[val] = -1

for key, val in new.items():
    print("{}, {}".format(key, val))

for key, val in otonew.items():
    print("{} {}".format(key, val))

writelines = []

with open(a.changefile, "r") as infile:
    for linen, line in enumerate(infile):
        line = line.split(",")
        line = [s.strip() for s in line]
        #line = [s for s in line if len(s) > 0]
        writelines.append(line)
        if linen > 0:
            for i in range(2, len(line)):
                if len(line[i]) > 0:
                    writelines[-1][i] = str(otonew[int(line[i])])


cplisti = writelines[0].index("canopy list")
cosplisti = writelines[0].index("co-species list")

print("{} {}".format(cplisti, cosplisti))

for line in writelines:
    for i in range(2, len(line)):
        if line[i] == "-1" or len(line[i]) == 0:
            if i < cosplisti:
                nextone = -1
                for j in range(i + 1, cosplisti):
                    if len(line[j]) > 0 and int(line[j]) > -1:
                        nextone = j
                        break
                if nextone == -1:
                    line[i] = ""
                    continue
                line[i] = line[nextone]
                line[nextone] = ""
            else:
                nextone = -1
                for j in range(i + 1, len(line)):
                    if len(line[j]) > 0 and int(line[j]) > -1:
                        nextone = j
                        break
                if nextone == -1:
                    line[i] = ""
                    break
                line[i] = line[nextone]
                line[nextone] = ""

for line in writelines:
    outstrline = ",".join(line)
    print(outstrline)
with open(a.changefile + "new.csv", "w+") as outfile:
    for line in writelines:
        outstrline = ",".join(line) + "\n"
        outfile.write(outstrline)

#for line in writelines:
#    print(line)
