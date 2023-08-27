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

    bool getBoolState(std::string const& key, bool defaultValue);
    void setBoolState(std::string const& key, bool value);

    int getIntState(std::string const& key, int defaultValue);
    void setIntState(std::string const& key, int value);

    float getFloatState(std::string const& key, float defaultValue);
    void setFloatState(std::string const& key, float value);

    std::string getStringState(std::string const& key, std::string const& defaultValue);
    void setStringState(std::string const& key, std::string value);

private:
    GlobalSettings();
    ~GlobalSettings();

    std::shared_ptr<GlobalSettingsImpl> _impl;
};