#include <iostream>
#include <experimental/filesystem>
#include "validator.h"
#include "data_importer.h"
#include <ctype.h>
#include <cassert>

void print_result(std::string category, int result)
{
    std::cout << category << ": ";
    if (result > 0)
    {
        std::cout << "SUCCESS";
    }
    else if (result == 0)
    {
        std::cout << "FAIL";
    }
    else if (result < 0)
    {
        std::cout << "NOT TESTED";
    }
    std::cout << std::endl;
}

int main(int argc, char * argv [])
{
    std::map<int, std::vector<std::string> > pdb_filenames;
    std::string datadir, dbpathname;
    if (argc > 5 || argc < 4)
    {
        std::cout << "Usage: main <pdb descr> <data directory> <db filename> --pipeout_num" << std::endl;
        return 1;
    }
    else if (argc == 4)
    {
        datadir = argv[2];
        pdb_filenames[-1].push_back(std::string(argv[1]));
        dbpathname = argv[3];
    }
    else if (argc == 5 && std::string(argv[4]) == "--pipeout_num")
    {
        using namespace std::experimental::filesystem;

        datadir = argv[2];
        dbpathname = argv[3];

        path pipepath_base = path(datadir);
        pipepath_base /= path("pipe_out");

        bool validspec = false;
        std::string containsstr;
        if (std::string(argv[1]) == "canopy")
        {
            validspec = true;
        }
        else if (std::string(argv[1]) == "undergrowth_quick")
        {
            validspec = true;
        }
        else if (std::string (argv[1]) == "undergrowth")
        {
            validspec = true;
        }
        else
        {
            std::cout << "Usage: main <pdb descr> <data directory> --pipeout_num" << std::endl;
            throw std::invalid_argument("pdb descr parameter must be either 'canopy' or 'undergrowth' if --pipeout_num is given");
        }
        containsstr = argv[1];
        try
        {
            for (auto &f : directory_iterator(datadir))
            {
                path p = f.path();
                path pname = p.filename();
                std::string pstr = pname.string();
                std::string numstr;
                while (isdigit(pstr.back()))
                {
                    numstr.insert(numstr.begin(), pstr.back());
                    pstr.pop_back();
                }
                if (pstr == "pipe_out")
                {
                    int num = std::stoi(numstr);
                    try
                    {
                        for (auto &fp : directory_iterator(p))
                        {
                            std::string fpname = fp.path().filename().string();
                            if (fpname.find(containsstr) != std::string::npos)
                            {
                                if (fp.path().filename().extension().string() == ".pdb")
                                    pdb_filenames[num].push_back(fp.path().string());
                            }
                        }
                    }
                    catch (v1::__cxx11::filesystem_error &e)
                    {
                        throw v1::__cxx11::filesystem_error(std::string(std::string(e.what()) +  "\nAdditional info: error when opening directory " + p.string()).c_str(), std::make_error_code(std::errc::no_such_file_or_directory));
                    }
                }
            }
        }
        catch (v1::__cxx11::filesystem_error &e)
        {
            throw v1::__cxx11::filesystem_error(std::string(std::string(e.what()) +  "\nAdditional info: error when opening directory " + datadir).c_str(), std::make_error_code(std::errc::no_such_file_or_directory));
        }
    }
    else
    {
        std::cout << "Usage: main <pdb descr> <data directory> <db pathname> --pipeout_num" << std::endl;
    }

    /*
    for (auto &f : pdb_filenames)
    {
        std::cout << f << std::endl;
    }
    */

    //validator v("/home/konrad/pipeline_canopytrees1.pdb", "/home/konrad/PhDStuff/data/analyse_specassign0");

    std::vector<std::string> fail_fnames;

    data_importer::common_data cdata(dbpathname);

    data_importer::modelset &mset = cdata.modelsamplers.at(15);

    int vueid = mset.sample_selection_fast(60.0f);

    printf("vueid selected for species id %d at height %f\n", 15, 60.0f);

    return 0;

    for (auto &p: pdb_filenames)
    {
        if (p.first > -1)
        {
            std::cout << "Validation for pipeline output " << p.first << ": " << std::endl;
        }
        else
        {
            assert(pdb_filenames.size() == 1);
            std::cout << "Validation for pdb file: ";
        }
        for (auto &fname : p.second)
        {
            std::cout << "------------------------------" << std::endl;
            std::cout << fname << std::endl;
            validator v(fname, datadir);

            bool valid = true;

            if (!v.validate_bounds()) valid = false;
            if (!v.validate_intersect()) valid = false;
            if (!v.validate_water(false)) valid = false;

            if (!valid) fail_fnames.push_back(fname);

            print_result("WATER", v.water_isvalid());
            print_result("BOUNDS", v.bounds_isvalid());
            print_result("INTERSECT", v.intersect_isvalid());

            if (v.bounds_isvalid() == 0)
            {
                v.analyse_boundsfailure();
            }
        }
        std::cout << "------------------------------" << std::endl;
    }

    if (fail_fnames.size() > 0)
    {
        std::cout << "Following pdb files had problems: " << std::endl;
        for (auto &fname : fail_fnames)
        {
            std::cout << fname << std::endl;
        }
    }
    else
    {
        std::cout << "All succeeded" << std::endl;
    }

	return 0;
}
