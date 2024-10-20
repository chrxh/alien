#pragma once

#include <memory>
#include <string>
#include <vector>

struct GlobalSettingsImpl;

class GlobalSettings
{
public:
    static GlobalSettings& get();
    GlobalSettings(GlobalSettings const&) = delete;
    void operator=(GlobalSettings const&) = delete;

    bool isDebugMode() const;
    void setDebugMode(bool value) const;

    bool getValue(std::string const& key, bool defaultValue);
    void setValue(std::string const& key, bool value);

    int getValue(std::string const& key, int defaultValue);
    void setValue(std::string const& key, int value);

    float getValue(std::string const& key, float defaultValue);
    void setValue(std::string const& key, float value);

    std::string getValue(std::string const& key, std::string const& defaultValue);
    void setValue(std::string const& key, std::string value);

    std::vector<std::string> getValue(std::string const& key, std::vector<std::string> const& defaultValue);
    void setValue(std::string const& key, std::vector<std::string> value);

private:
    GlobalSettings();
    ~GlobalSettings();

    std::shared_ptr<GlobalSettingsImpl> _impl;
};