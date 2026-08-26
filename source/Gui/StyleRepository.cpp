#include "StyleRepository.h"

#include <stdexcept>

#include <imgui.h>
#include <imgui_freetype.h>

#include <Fonts/AlienIconFont.h>
#include <Fonts/Cousine-Regular.h>
#include <Fonts/DroidSans.h>
#include <Fonts/DroidSansBold.h>
#include <Fonts/FontAwesomeSolid.h>
#include <Fonts/IconsFontAwesome5.h>
#include <Fonts/Reef.h>

#include <GLFW/glfw3.h>  // Will drag system OpenGL headers
#include <ImFileDialog.h>
#include <implot.h>

#include "WindowController.h"

void StyleRepository::setup()
{
    auto scaleFactor = WindowController::get().getContentScaleFactor();

    auto& style = ImGui::GetStyle();
    style.ScaleAllSizes(scaleFactor);

    setupSizes(style);
    setupColors(style);

    ImFontConfig configMerge;
    configMerge.MergeMode = true;
    configMerge.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LightHinting;

    ImGuiIO& io = ImGui::GetIO();

    // Default font (small with icons)
    io.Fonts->AddFontFromMemoryCompressedTTF(DroidSans_compressed_data, DroidSans_compressed_size, 16.0f * scaleFactor);
    {
        static const ImWchar rangesIcons[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
        io.Fonts->AddFontFromMemoryCompressedTTF(
            FontAwesomeSolid_compressed_data, FontAwesomeSolid_compressed_size, 16.0f * scaleFactor, &configMerge, rangesIcons);
    }

    // Tiny font
    _tinyFont = io.Fonts->AddFontFromMemoryCompressedTTF(DroidSans_compressed_data, DroidSans_compressed_size, 11.0f * scaleFactor);
    {
        static const ImWchar rangesIcons[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
        io.Fonts->AddFontFromMemoryCompressedTTF(
            FontAwesomeSolid_compressed_data, FontAwesomeSolid_compressed_size, 11.0f * scaleFactor, &configMerge, rangesIcons);
    }

    // Small bold font
    _smallBoldFont = io.Fonts->AddFontFromMemoryCompressedTTF(DroidSansBold_compressed_data, DroidSansBold_compressed_size, 16.0f * scaleFactor);

    // Medium bold font
    _mediumBoldFont = io.Fonts->AddFontFromMemoryCompressedTTF(DroidSansBold_compressed_data, DroidSansBold_compressed_size, 24.0f * scaleFactor);

    // Medium font
    _mediumFont = io.Fonts->AddFontFromMemoryCompressedTTF(DroidSans_compressed_data, DroidSans_compressed_size, 24.0f * scaleFactor);

    // Large font
    _largeFont = io.Fonts->AddFontFromMemoryCompressedTTF(DroidSans_compressed_data, DroidSans_compressed_size, 48.0f * scaleFactor);

    // Icon font
    _iconFont = io.Fonts->AddFontFromMemoryCompressedTTF(AlienIconFont_compressed_data, AlienIconFont_compressed_size, 24.0f * scaleFactor);
    {
        static const ImWchar rangesIcons[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
        io.Fonts->AddFontFromMemoryCompressedTTF(
            FontAwesomeSolid_compressed_data, FontAwesomeSolid_compressed_size, 28.0f * scaleFactor, &configMerge, rangesIcons);
        io.Fonts->Build();
    }

    // Monospace medium font
    _monospaceMediumFont = io.Fonts->AddFontFromMemoryCompressedTTF(Cousine_Regular_compressed_data, Cousine_Regular_compressed_size, 14.0f * scaleFactor);

    // Monospace large font
    _monospaceLargeFont = io.Fonts->AddFontFromMemoryCompressedTTF(Cousine_Regular_compressed_data, Cousine_Regular_compressed_size, 128.0f * scaleFactor);

    _reefMediumFont = io.Fonts->AddFontFromMemoryCompressedTTF(Reef_compressed_data, Reef_compressed_size, 24.0f * scaleFactor);
    _reefLargeFont = io.Fonts->AddFontFromMemoryCompressedTTF(Reef_compressed_data, Reef_compressed_size, 64.0f * scaleFactor);
}

void StyleRepository::setupSizes(ImGuiStyle& style) const
{
    style.FrameRounding = scale(6.0f);
    style.ChildRounding = scale(8.0f);
    style.PopupRounding = scale(8.0f);
    style.GrabRounding = scale(6.0f);
    style.ScrollbarRounding = scale(6.0f);
    style.TabRounding = scale(6.0f);
}

void StyleRepository::setupColors(ImGuiStyle& style) const
{
    auto transparent = ImColor::HSV(0.0f, 0.0f, 0.0f, 0.0f);

    style.Colors[ImGuiCol_Text] = Const::TextBaseColor.Value;
    style.Colors[ImGuiCol_TextDisabled] = Const::TextFaintColor.Value;

    style.Colors[ImGuiCol_WindowBg] = Const::BackgroundColor.Value;
    style.Colors[ImGuiCol_ChildBg] = transparent.Value;
    style.Colors[ImGuiCol_PopupBg] = Const::PanelColor.Value;
    style.Colors[ImGuiCol_Border] = Const::LineColor.Value;
    style.Colors[ImGuiCol_BorderShadow] = transparent.Value;

    style.Colors[ImGuiCol_FrameBg] = Const::InputColor.Value;
    style.Colors[ImGuiCol_FrameBgHovered] = Const::RaisedColor.Value;
    style.Colors[ImGuiCol_FrameBgActive] = Const::RaisedColor.Value;

    style.Colors[ImGuiCol_TitleBg] = Const::PanelColor.Value;
    style.Colors[ImGuiCol_TitleBgActive] = Const::PanelColor.Value;
    style.Colors[ImGuiCol_TitleBgCollapsed] = Const::BackgroundColor.Value;
    style.Colors[ImGuiCol_MenuBarBg] = Const::PanelColor.Value;

    style.Colors[ImGuiCol_ScrollbarBg] = transparent.Value;
    style.Colors[ImGuiCol_ScrollbarGrab] = Const::LineColor.Value;
    style.Colors[ImGuiCol_ScrollbarGrabHovered] = Const::TextFaintColor.Value;
    style.Colors[ImGuiCol_ScrollbarGrabActive] = Const::AccentLineColor.Value;

    style.Colors[ImGuiCol_CheckMark] = Const::AccentColor.Value;
    style.Colors[ImGuiCol_SliderGrab] = Const::AccentLineColor.Value;
    style.Colors[ImGuiCol_SliderGrabActive] = Const::AccentColor.Value;

    style.Colors[ImGuiCol_Button] = Const::RaisedColor.Value;
    style.Colors[ImGuiCol_ButtonHovered] = Const::ActionButtonHoveredColor.Value;
    style.Colors[ImGuiCol_ButtonActive] = Const::AccentDeepColor.Value;

    style.Colors[ImGuiCol_Header] = Const::HeaderColor.Value;
    style.Colors[ImGuiCol_HeaderHovered] = Const::HeaderHoveredColor.Value;
    style.Colors[ImGuiCol_HeaderActive] = Const::HeaderActiveColor.Value;

    style.Colors[ImGuiCol_Separator] = Const::LineSoftColor.Value;
    style.Colors[ImGuiCol_SeparatorHovered] = Const::AccentLineColor.Value;
    style.Colors[ImGuiCol_SeparatorActive] = Const::AccentColor.Value;

    style.Colors[ImGuiCol_ResizeGrip] = Const::LineColor.Value;
    style.Colors[ImGuiCol_ResizeGripHovered] = Const::AccentLineColor.Value;
    style.Colors[ImGuiCol_ResizeGripActive] = Const::AccentColor.Value;

    style.Colors[ImGuiCol_Tab] = Const::PanelColor.Value;
    style.Colors[ImGuiCol_TabHovered] = Const::RaisedColor.Value;
    style.Colors[ImGuiCol_TabSelected] = Const::RaisedColor.Value;
    style.Colors[ImGuiCol_TabSelectedOverline] = Const::AccentColor.Value;
    style.Colors[ImGuiCol_TabDimmed] = Const::BackgroundColor.Value;
    style.Colors[ImGuiCol_TabDimmedSelected] = Const::PanelColor.Value;
    style.Colors[ImGuiCol_TabDimmedSelectedOverline] = Const::AccentLineColor.Value;

    style.Colors[ImGuiCol_TableHeaderBg] = Const::TableHeaderColor.Value;
    style.Colors[ImGuiCol_TableBorderStrong] = Const::LineColor.Value;
    style.Colors[ImGuiCol_TableBorderLight] = Const::LineSoftColor.Value;
    style.Colors[ImGuiCol_TableRowBg] = transparent.Value;
    style.Colors[ImGuiCol_TableRowBgAlt] = ImColor::HSV(0.583f, 0.323f, 0.122f, 0.4f).Value;

    style.Colors[ImGuiCol_TextSelectedBg] = Const::AccentDeepColor.Value;
    style.Colors[ImGuiCol_DragDropTarget] = Const::AccentColor.Value;
    style.Colors[ImGuiCol_NavCursor] = Const::AccentLineColor.Value;
}

ImFont* StyleRepository::getIconFont() const
{
    return _iconFont;
}

ImFont* StyleRepository::getDefaultFont() const
{
    return ImGui::GetIO().Fonts->Fonts[0];
}

ImFont* StyleRepository::getTinyFont() const
{
    return _tinyFont;
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
    return value / WindowController::get().getContentScaleFactor();
}
