#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/Parser.h"

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

private:
    GlobalSettings();
    ~GlobalSettings();

    void encodeDecodeGpuSettings(GpuSettings& gpuSettings, Parser::Task task);

    GlobalSettingsImpl* _impl;
};