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
    encodeDecodeGpuSettings(result, ParserTask::Decode);
    return result;
}

void GlobalSettings::setGpuSettings(GpuSettings gpuSettings)
{
    encodeDecodeGpuSettings(gpuSettings, ParserTask::Encode);
}

bool GlobalSettings::getBoolState(std::string const& name, bool defaultValue)
{
    bool result;
    JsonParser::encodeDecode(_impl->_tree, result, defaultValue, name, ParserTask::Decode);
    return result;
}

void GlobalSettings::setBoolState(std::string const& name, bool value)
{
    JsonParser::encodeDecode(_impl->_tree, value, false, name, ParserTask::Encode);
}

int GlobalSettings::getIntState(std::string const& name, int defaultValue)
{
    int result;
    JsonParser::encodeDecode(_impl->_tree, result, defaultValue, name, ParserTask::Decode);
    return result;
}

void GlobalSettings::setIntState(std::string const& name, int value)
{
    JsonParser::encodeDecode(_impl->_tree, value, 0, name, ParserTask::Encode);
}

void GlobalSettings::encodeDecodeGpuSettings(GpuSettings& gpuSettings, ParserTask task)
{
    GpuSettings defaultSettings;
    JsonParser::encodeDecode(
        _impl->_tree,
        gpuSettings.NUM_BLOCKS,
        defaultSettings.NUM_BLOCKS, "settings.gpu.num blocks", task);
    JsonParser::encodeDecode(
        _impl->_tree,
        gpuSettings.NUM_THREADS_PER_BLOCK,
        defaultSettings.NUM_THREADS_PER_BLOCK,
        "settings.gpu.num threads per block",
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
