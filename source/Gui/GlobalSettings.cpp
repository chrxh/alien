#include "GlobalSettings.h"

#include <sstream>
#include <fstream>

#include <boost/property_tree/json_parser.hpp>

#ifdef _WIN32
#include "WinReg/WinReg.hpp"
#endif

#include "Base/LoggingService.h"
#include "Base/Resources.h"

#ifdef _WIN32
namespace
{
    std::string ConvertWideToUtf8(const std::wstring& wstr)
    {
        int count = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), wstr.length(), NULL, 0, NULL, NULL);
        std::string str(count, 0);
        WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &str[0], count, NULL, NULL);
        return str;
    }

    std::wstring ConvertUtf8ToWide(const std::string& str)
    {
        int count = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), str.length(), NULL, 0);
        std::wstring wstr(count, 0);
        MultiByteToWideChar(CP_UTF8, 0, str.c_str(), str.length(), &wstr[0], count);
        return wstr;
    }

    winreg::RegKey getOrCreateAlienRegKey()
    {
        const std::wstring testSubKey = L"SOFTWARE\\alien";
        winreg::RegKey key{HKEY_CURRENT_USER, testSubKey};
        if (key.TryOpen(HKEY_CURRENT_USER, testSubKey)) {
            return key;
        } else {
            auto result = key.TryCreate(HKEY_CURRENT_USER, testSubKey);
            if (result.IsOk()) {
                if (key.TryOpen(HKEY_CURRENT_USER, testSubKey)) {
                    return key;
                }
            }
        }
        throw std::runtime_error("Could not open a registry key.");
    }

    std::string getStringFromWinReg(std::string const& name, std::string const& defaultValue)
    {
        try {
            auto key = getOrCreateAlienRegKey();
            auto regResult = key.TryGetStringValue(ConvertUtf8ToWide(name));
            if (regResult.IsValid()) {
                return ConvertWideToUtf8(regResult.GetValue());
            } else {
                return defaultValue;
            }

        } catch (std::exception const& exception) {
            log(Priority::Important, exception.what());
            return defaultValue;
        }
    }

    void setStringToWinReg(std::string const& name, std::string const& value)
    {
        try {
            auto key = getOrCreateAlienRegKey();
            key.SetStringValue(ConvertUtf8ToWide(name), ConvertUtf8ToWide(value));

        } catch (std::exception const& exception) {
            log(Priority::Important, exception.what());
        }
    }

    int getIntFromWinReg(std::string const& name, int const& defaultValue)
    {
        try {
            auto key = getOrCreateAlienRegKey();
            auto regResult = key.TryGetDwordValue(ConvertUtf8ToWide(name));
            if (regResult.IsValid()) {
                return regResult.GetValue();
            } else {
                return defaultValue;
            }

        } catch (std::exception const& exception) {
            log(Priority::Important, exception.what());
            return defaultValue;
        }
    }

    void setIntToWinReg(std::string const& name, int const& value)
    {
        try {
            auto key = getOrCreateAlienRegKey();
            key.SetDwordValue(ConvertUtf8ToWide(name), value);

        } catch (std::exception const& exception) {
            log(Priority::Important, exception.what());
        }
    }

    float getFloatFromWinReg(std::string const& name, float const& defaultValue)
    {
        auto resultAsString = getStringFromWinReg(name, std::to_string(defaultValue));
        return std::stof(resultAsString);
    }

    void setFloatToWinReg(std::string const& name, float const& value)
    {
        setStringToWinReg(name, std::to_string(value));
    }

    bool getBoolFromWinReg(std::string const& name, bool const& defaultValue)
    {
        auto resultAsString = getStringFromWinReg(name, defaultValue ? "1" : "0");
        return resultAsString == "1" ? true : false;
    }

    void setBoolToWinReg(std::string const& name, bool const& value)
    {
        setStringToWinReg(name, value ? "1" : "0");
    }
}
#endif

struct GlobalSettingsImpl
{
    boost::property_tree::ptree _tree;
};


GlobalSettings& GlobalSettings::getInstance()
{
    static GlobalSettings instance;
    return instance;
}

bool GlobalSettings::getBoolState(std::string const& key, bool defaultValue)
{
    bool result;
#ifdef _WIN32
    result = getBoolFromWinReg(key, defaultValue);
#else
    JsonParser::encodeDecode(_impl->_tree, result, defaultValue, key, ParserTask::Decode);
#endif
    return result;
}

void GlobalSettings::setBoolState(std::string const& key, bool value)
{
#ifdef _WIN32
    setBoolToWinReg(key, value);
#else
    JsonParser::encodeDecode(_impl->_tree, value, false, key, ParserTask::Encode);
#endif
}

int GlobalSettings::getIntState(std::string const& key, int defaultValue)
{
    int result;
#ifdef _WIN32
    result = getIntFromWinReg(key, defaultValue);
#else
    JsonParser::encodeDecode(_impl->_tree, result, defaultValue, key, ParserTask::Decode);
#endif
    return result;
}

void GlobalSettings::setIntState(std::string const& key, int value)
{
#ifdef _WIN32
    setIntToWinReg(key, value);
#else
    JsonParser::encodeDecode(_impl->_tree, value, 0, key, ParserTask::Encode);
#endif
}

float GlobalSettings::getFloatState(std::string const& key, float defaultValue)
{
    float result;
#ifdef _WIN32
    result = getFloatFromWinReg(key, defaultValue);
#else
    JsonParser::encodeDecode(_impl->_tree, result, defaultValue, key, ParserTask::Decode);
#endif
    return result;
}

void GlobalSettings::setFloatState(std::string const& key, float value)
{
#ifdef _WIN32
    setFloatToWinReg(key, value);
#else
    JsonParser::encodeDecode(_impl->_tree, value, 0.0f, key, ParserTask::Encode);
#endif
}

std::string GlobalSettings::getStringState(std::string const& key, std::string const& defaultValue)
{
    std::string result;
#ifdef _WIN32
    result = getStringFromWinReg(key, defaultValue);
#else
    JsonParser::encodeDecode(_impl->_tree, result, defaultValue, key, ParserTask::Decode);
#endif
    return result;
}

void GlobalSettings::setStringState(std::string const& key, std::string value)
{
#ifdef _WIN32
    setStringToWinReg(key, value);
#else
    JsonParser::encodeDecode(_impl->_tree, value, std::string(), key, ParserTask::Encode);
#endif
}

GlobalSettings::GlobalSettings()
{
#ifndef _WIN32
    try {
        _impl = std::make_shared<GlobalSettingsImpl>();
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
#endif
}

GlobalSettings::~GlobalSettings()
{
#ifndef _WIN32
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
#endif
}
