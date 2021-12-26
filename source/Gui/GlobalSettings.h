#pragma once

#include "Base/JsonParser.h"
#include "EngineInterface/GpuSettings.h"

#include "Definitions.h"

struct GlobalSettingsImpl;

class GlobalSettings
{
public:
    static GlobalSettings& getInstance();

    GlobalSettings(GlobalSettings const&) = delete;
    void operator=(GlobalSettings const&) = delete;

    GpuSettings getGpuSettings();
    void setGpuSettings(GpuSettings gpuSettings);

    bool getBoolState(std::string const& name, bool defaultValue);
    void setBoolState(std::string const& name, bool value);

    int getIntState(std::string const& name, int defaultValue);
    void setIntState(std::string const& name, int value);

    std::string getStringState(std::string const& name, std::string defaultValue);
    void setStringState(std::string const& name, std::string value);

private:
    GlobalSettings();
    ~GlobalSettings();

    void encodeDecodeGpuSettings(GpuSettings& gpuSettings, ParserTask task);

    boost::shared_ptr<GlobalSettingsImpl> _impl;
};