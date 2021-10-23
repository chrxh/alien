#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/Parser.h"

#include "Definitions.h"

struct GlobalSettingsImpl;

class _GlobalSettings
{
public:
    _GlobalSettings();
    ~_GlobalSettings();

    _GlobalSettings(_GlobalSettings const&) = delete;
    void operator=(_GlobalSettings const&) = delete;

    GpuSettings getGpuSettings();
    void setGpuSettings(GpuSettings gpuSettings);

private:
    void encodeDecodeGpuSettings(GpuSettings& gpuSettings, Parser::Task task);

    GlobalSettingsImpl* _impl;
};