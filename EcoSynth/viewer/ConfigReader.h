#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include "rapidjson/document.h"
#include "glwidget.h"

struct configparams
{
    std::string scene_dirname;
    std::vector<std::string> clusterdata_filenames;
    std::string canopy_filename;
    std::string undergrowth_filename;
    ControlMode ctrlmode;
    bool render_canopy;
    bool render_undergrowth;
};

class ConfigReader
{
public:
        ConfigReader(std::string filename);

        bool read();
        configparams get_params();

private:
        std::string filename;
        configparams params;
        rapidjson::Document jsondoc;
        bool has_read;

};

#endif // CONFIG_READER_H
