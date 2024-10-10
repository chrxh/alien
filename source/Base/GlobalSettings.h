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

    bool getBool(std::string const& key, bool defaultValue);
    void setBool(std::string const& key, bool value);

    int getInt(std::string const& key, int defaultValue);
    void setInt(std::string const& key, int value);

    float getFloat(std::string const& key, float defaultValue);
    void setFloat(std::string const& key, float value);

    std::string getString(std::string const& key, std::string const& defaultValue);
    void setString(std::string const& key, std::string value);

    std::vector<std::string> getStringVector(std::string const& key, std::vector<std::string> const& defaultValue);
    void setStringVector(std::string const& key, std::vector<std::string> value);

private:
    GlobalSettings();
    ~GlobalSettings();

    std::shared_ptr<GlobalSettingsImpl> _impl;
};