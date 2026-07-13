#include "TranslationService.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

TranslationService& TranslationService::get()
{
    static TranslationService instance;
    return instance;
}

bool TranslationService::load(std::string const& language)
{
    std::vector<std::string> candidates = {
        "translations/" + language + ".json",
        "../source/Gui/translations/" + language + ".json",
        fs::path(fs::absolute(fs::path(__FILE__)).parent_path().generic_string()) / "translations" / (language + ".json"),
    };

    for (auto const& path : candidates) {
        if (!fs::exists(path)) {
            continue;
        }
        std::ifstream file(path);
        if (!file.is_open()) {
            continue;
        }
        try {
            _translations = nlohmann::json::parse(file);
            _currentLanguage = language;
            return true;
        } catch (...) {
            continue;
        }
    }
    return false;
}

char const* TranslationService::tr(char const* key) const
{
    if (_currentLanguage == "en") {
        return key;
    }
    auto it = _translations.find(key);
    if (it != _translations.end() && it->is_string()) {
        return it->get_ref<std::string const&>().c_str();
    }
    return key;
}

std::string const& TranslationService::currentLanguage() const
{
    return _currentLanguage;
}

std::vector<std::string> TranslationService::availableLanguages() const
{
    std::vector<std::string> languages{"en"};
    for (auto const& dir : {"translations", "../source/Gui/translations"}) {
        if (!fs::exists(dir)) {
            continue;
        }
        for (auto const& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".json") {
                auto lang = entry.path().stem().string();
                if (lang != "en" && std::find(languages.begin(), languages.end(), lang) == languages.end()) {
                    languages.push_back(lang);
                }
            }
        }
    }
    return languages;
}
