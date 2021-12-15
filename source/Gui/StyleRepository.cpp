#include "StyleRepository.h"

#include <stdexcept>

#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <imgui.h>
#include <imgui_freetype.h>

#include "Fonts/DroidSans.h"
#include "Fonts/Cousine-Regular.h"
#include "IconFontCppHeaders/FontAwesomeSolid.h"
#include "IconFontCppHeaders/IconsFontAwesome5.h"

#include "Resources.h"

_StyleRepository::_StyleRepository()
{
    float temp;
    glfwGetMonitorContentScale(
        glfwGetPrimaryMonitor(), &_contentScaleFactor, &temp);  //consider only horizontal content scale

    ImGuiIO& io = ImGui::GetIO();

    if (io.Fonts->AddFontFromMemoryCompressedTTF(
            DroidSans_compressed_data, DroidSans_compressed_size, 16.0f * _contentScaleFactor)
        == NULL) {
        throw std::runtime_error("Could not load font.");
    };

    ImFontConfig configMerge;
    configMerge.MergeMode = true;
    configMerge.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LightHinting;
    static const ImWchar rangesIcons[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
    io.Fonts->AddFontFromMemoryCompressedTTF(
        FontAwesomeSolid_compressed_data,
        FontAwesomeSolid_compressed_size,
        16.0f * _contentScaleFactor,
        &configMerge,
        rangesIcons);

    _mediumFont = io.Fonts->AddFontFromMemoryCompressedTTF(
        DroidSans_compressed_data, DroidSans_compressed_size, 24.0f * _contentScaleFactor);
    if (_mediumFont == NULL) {
        throw std::runtime_error("Could not load font.");
    }
    _largeFont = io.Fonts->AddFontFromMemoryCompressedTTF(
        DroidSans_compressed_data, DroidSans_compressed_size, 48.0f * _contentScaleFactor);
    if (_largeFont == NULL) {
        throw std::runtime_error("Could not load font.");
    }
    _monospaceFont = io.Fonts->AddFontFromMemoryCompressedTTF(
        Cousine_Regular_compressed_data, Cousine_Regular_compressed_size, 14.0f * _contentScaleFactor);
    if (_monospaceFont == NULL) {
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

ImFont* _StyleRepository::getMonospaceFont() const
{
    return _monospaceFont;
}

float _StyleRepository::scaleContent(float value) const
{
    return _contentScaleFactor * value;
}
