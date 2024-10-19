#include "StyleRepository.h"

#include <stdexcept>

#include <ImFileDialog.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <imgui.h>
#include <imgui_freetype.h>
#include <implot.h>

#include "Fonts/DroidSans.h"
#include "Fonts/DroidSansBold.h"
#include "Fonts/Cousine-Regular.h"
#include "Fonts/AlienIconFont.h"
#include "Fonts/FontAwesomeSolid.h"
#include "Fonts/IconsFontAwesome5.h"
#include "Fonts/Reef.h"

#include "WindowController.h"

void StyleRepository::init()
{
    auto scaleFactor = WindowController::get().getContentScaleFactor();

    auto& style = ImGui::GetStyle();
    style.ScaleAllSizes(scaleFactor);


    ImFontConfig configMerge;
    configMerge.MergeMode = true;
    configMerge.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LightHinting;

    ImGuiIO& io = ImGui::GetIO();

    //default font (small with icons)
    io.Fonts->AddFontFromMemoryCompressedTTF(DroidSans_compressed_data, DroidSans_compressed_size, 16.0f * scaleFactor);
    {
        static const ImWchar rangesIcons[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
        io.Fonts->AddFontFromMemoryCompressedTTF(
            FontAwesomeSolid_compressed_data, FontAwesomeSolid_compressed_size, 16.0f * scaleFactor, &configMerge, rangesIcons);
    }

    //small bold font
    _smallBoldFont = io.Fonts->AddFontFromMemoryCompressedTTF(DroidSansBold_compressed_data, DroidSansBold_compressed_size, 16.0f * scaleFactor);

    //medium bold font
    _mediumBoldFont = io.Fonts->AddFontFromMemoryCompressedTTF(DroidSansBold_compressed_data, DroidSansBold_compressed_size, 24.0f * scaleFactor);

    //medium font
    _mediumFont = io.Fonts->AddFontFromMemoryCompressedTTF(DroidSans_compressed_data, DroidSans_compressed_size, 24.0f * scaleFactor);

    //large font
    _largeFont = io.Fonts->AddFontFromMemoryCompressedTTF(DroidSans_compressed_data, DroidSans_compressed_size, 48.0f * scaleFactor);

    //icon font
    _iconFont = io.Fonts->AddFontFromMemoryCompressedTTF(AlienIconFont_compressed_data, AlienIconFont_compressed_size, 24.0f * scaleFactor);
    {
        static const ImWchar rangesIcons[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
        io.Fonts->AddFontFromMemoryCompressedTTF(
            FontAwesomeSolid_compressed_data, FontAwesomeSolid_compressed_size, 28.0f * scaleFactor, &configMerge, rangesIcons);
        io.Fonts->Build();
    }

    //monospace medium font
    _monospaceMediumFont = io.Fonts->AddFontFromMemoryCompressedTTF(Cousine_Regular_compressed_data, Cousine_Regular_compressed_size, 14.0f * scaleFactor);

    //monospace large font
    _monospaceLargeFont = io.Fonts->AddFontFromMemoryCompressedTTF(Cousine_Regular_compressed_data, Cousine_Regular_compressed_size, 128.0f * scaleFactor);

    _reefMediumFont = io.Fonts->AddFontFromMemoryCompressedTTF(Reef_compressed_data, Reef_compressed_size, 24.0f * scaleFactor);
    _reefLargeFont = io.Fonts->AddFontFromMemoryCompressedTTF(Reef_compressed_data, Reef_compressed_size, 64.0f * scaleFactor);
}

ImFont* StyleRepository::getIconFont() const
{
    return _iconFont;
}

ImFont* StyleRepository::getSmallBoldFont() const
{
    return _smallBoldFont;
}

ImFont* StyleRepository::getMediumBoldFont() const
{
    return _mediumBoldFont;
}

ImFont* StyleRepository::getMediumFont() const
{
    return _mediumFont;
}

ImFont* StyleRepository::getLargeFont() const
{
    return _largeFont;
}

ImFont* StyleRepository::getMonospaceMediumFont() const
{
    return _monospaceMediumFont;
}

ImFont* StyleRepository::getMonospaceLargeFont() const
{
    return _monospaceLargeFont;
}

ImFont* StyleRepository::getReefMediumFont() const
{
    return _reefMediumFont;
}

ImFont* StyleRepository::getReefLargeFont() const
{
    return _reefLargeFont;
}

float StyleRepository::scale(float value) const
{
    return WindowController::get().getContentScaleFactor() * value;
}

float StyleRepository::scaleInverse(float value) const
{
    return WindowController::get().getContentScaleFactor() / value;
}
