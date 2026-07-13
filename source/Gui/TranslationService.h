#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

class TranslationService
{
public:
    static TranslationService& get();

    bool load(std::string const& language);
    char const* tr(char const* key) const;
    std::string const& currentLanguage() const;
    std::vector<std::string> availableLanguages() const;

private:
    TranslationService() = default;

    nlohmann::json _translations = nlohmann::json::object();
    std::string _currentLanguage = "en";
};

#define _(key) TranslationService::get().tr(key)
