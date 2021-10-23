#include "GlobalSettings.h"

#include <sstream>
#include <fstream>

#include <boost/property_tree/json_parser.hpp>

#include "Resources.h"

struct GlobalSettingsImpl
{
    boost::property_tree::ptree _tree;
};


GlobalSettings& GlobalSettings::getInstance()
{
    static GlobalSettings instance;
    return instance;
}

GpuSettings GlobalSettings::getGpuSettings()
{
    GpuSettings result;
    encodeDecodeGpuSettings(result, Parser::Task::Decode);
    return result;
}

void GlobalSettings::setGpuSettings(GpuSettings gpuSettings)
{
    encodeDecodeGpuSettings(gpuSettings, Parser::Task::Encode);
}

void GlobalSettings::encodeDecodeGpuSettings(GpuSettings& gpuSettings, Parser::Task task)
{
    GpuSettings defaultSettings;
    Parser::encodeDecode(
        _impl->_tree,
        gpuSettings.NUM_BLOCKS,
        defaultSettings.NUM_BLOCKS, "GPU settings.num blocks", task);
    Parser::encodeDecode(
        _impl->_tree,
        gpuSettings.NUM_THREADS_PER_BLOCK,
        defaultSettings.NUM_THREADS_PER_BLOCK,
        "GPU settings.num threads per block",
        task);
}

GlobalSettings::GlobalSettings()
{
    try {
        _impl = new GlobalSettingsImpl;
        std::ifstream stream(Const::SettingsFilename, std::ios::binary);
        if (!stream) {
            return;
        }
        auto data = std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
        stream.close();

        std::stringstream ss;
        ss << data;
        boost::property_tree::read_json(ss, _impl->_tree);
    } catch (...) {
        //do nothing
    }
}

GlobalSettings::~GlobalSettings()
{
    try {
        std::stringstream ss;
        boost::property_tree::json_parser::write_json(ss, _impl->_tree);
        auto data = ss.str();

        std::ofstream stream(Const::SettingsFilename, std::ios::binary);
        if (stream) {
            stream << data;
            stream.close();
        }
    } catch (...) {
        //do nothing
    }

    delete _impl;
}
