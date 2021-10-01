#include "StyleRepository.h"

#include <stdexcept>

#include "imgui.h"
#include "Resources.h"

_StyleRepository::_StyleRepository()
{
    ImGuiIO& io = ImGui::GetIO();

    io = ImGui::GetIO();
    if (io.Fonts->AddFontFromFileTTF(Const::FontFilename, 16.0f)
        == NULL) {
        throw std::runtime_error("Could not load font.");
    };
    _mediumFont = io.Fonts->AddFontFromFileTTF(Const::FontFilename, 24.0f);
    if (_mediumFont == NULL) {
        throw std::runtime_error("Could not load font.");
    }
    _largeFont = io.Fonts->AddFontFromFileTTF(Const::FontFilename, 48.0f);
    if (_largeFont == NULL) {
        throw std::runtime_error("Could not load font.");
    }
}

ImFont* _StyleRepository::getMediumFont() const
{
    return _mediumFont;
}

ImFont* _StyleRepository::getLargeFont() const
{
    return _largeFont;
}
