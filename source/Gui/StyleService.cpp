#include "StyleService.h"

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

namespace Const
{
    float const WindowAlpha = 0.9f;
    float const MaximizedWindowAlpha = 0.95f;
    float const SliderBarWidth = 30.0f;
    float const WindowsRounding = 10.0f;

    // Base palette: neutral dark grays with a slight blue bias, one accent reserved for selection and focus
    ImColor const BackgroundColor = ImColor::HSV(0.583f, 0.316f, 0.075f);
    ImColor const PanelColor = ImColor::HSV(0.583f, 0.323f, 0.122f);
    ImColor const RaisedColor = ImColor::HSV(0.583f, 0.333f, 0.165f);
    ImColor const InputColor = ImColor::HSV(0.583f, 0.350f, 0.157f);
    ImColor const LineColor = ImColor::HSV(0.574f, 0.310f, 0.227f);
    ImColor const LineSoftColor = ImColor::HSV(0.578f, 0.326f, 0.180f);
    ImColor const TextBaseColor = ImColor::HSV(0.564f, 0.055f, 0.925f);
    ImColor const TextDimColor = ImColor::HSV(0.570f, 0.158f, 0.647f);
    ImColor const TextFaintColor = ImColor::HSV(0.570f, 0.222f, 0.459f);
    ImColor const AccentColor = ImColor::HSV(0.484f, 0.587f, 0.816f);
    ImColor const AccentDeepColor = ImColor::HSV(0.490f, 0.556f, 0.247f);
    ImColor const AccentLineColor = ImColor::HSV(0.487f, 0.569f, 0.427f);
    ImColor const WarningColor = ImColor::HSV(0.093f, 0.607f, 0.878f);
    ImColor const DangerColor = ImColor::HSV(0.979f, 0.603f, 0.878f);

    ImColor const ProgramVersionTextColor = ImColor::HSV(0.5f, 0.1f, 1.0f, 1.0f);

    ImColor const RenderingDisabledTextColor = ImColor::HSV(0.5f, 0.1f, 1.0f, 0.2f);

    int64_t const SimulationSliderColor_Base = 0xff4c4c4c;
    int64_t const SimulationSliderColor_Active = 0xff6c6c6c;

    ImColor const TextTooltipColor = ImColor::HSV(0.0f, 0.0f, 1.0f);
    ImColor const TextInfoColor = ImColor::HSV(0.167f, 0.64f, 0.53f);
    ImColor const TextDecentColor = TextFaintColor;
    ImColor const TextConflictColor = WarningColor;

    ImColor const HeaderColor = AccentDeepColor;
    ImColor const HeaderActiveColor = ImColor::HSV(0.489f, 0.548f, 0.329f);
    ImColor const HeaderHoveredColor = RaisedColor;
    ImColor const HeaderSelectedHoveredColor = ImColor::HSV(0.489f, 0.552f, 0.302f);

    ImColor const MenuButtonColor = ImColor::HSV(0.583f, 0.333f, 0.212f);
    ImColor const MenuButtonHoveredColor = AccentLineColor;
    ImColor const MenuButtonActiveColor = AccentDeepColor;

    ImColor const ImportantButtonColor = ImColor::HSV(0.980f, 0.607f, 0.478f);
    ImColor const ImportantButtonHoveredColor = ImColor::HSV(0.980f, 0.620f, 0.639f);
    ImColor const ImportantButtonActiveColor = DangerColor;

    ImColor const TreeNodeHighColor = ImColor::HSV(0.578f, 0.333f, 0.200f);
    ImColor const TreeNodeHighHoveredColor = ImColor::HSV(0.579f, 0.323f, 0.255f);
    ImColor const TreeNodeHighActiveColor = AccentDeepColor;
    ImColor const TreeNodeDefaultColor = PanelColor;
    ImColor const TreeNodeDefaultHoveredColor = RaisedColor;
    ImColor const TreeNodeDefaultActiveColor = AccentDeepColor;
    ImColor const TreeNodeLowColor = ImColor::HSV(0.556f, 0.333f, 0.106f);
    ImColor const TreeNodeLowHoveredColor = RaisedColor;
    ImColor const TreeNodeLowActiveColor = AccentDeepColor;

    ImColor const DisabledOverlayColor1 = ImColor::HSV(0.0f, 0.0f, 0.35f, 0.5f);
    ImColor const DisabledOverlayColor2 = ImColor::HSV(0.0f, 0.0f, 0.06f, 0.2f);

    ImColor const GroupDefaultColor = ImColor::HSV(0.583f, 0.333f, 0.141f);
    ImColor const GroupHighColor = ImColor::HSV(0.583f, 0.348f, 0.180f);
    ImColor const GroupAccentBarColor = AccentColor;
    ImColor const GroupTextColor = TextDimColor;
    ImColor const GroupHighTextColor = TextBaseColor;

    ImColor const MovableSeparatorColor = LineSoftColor;
    ImColor const MovableSeparatorHoveredColor = AccentLineColor;
    ImColor const MovableSeparatorActiveColor = AccentColor;

    ImColor const TableHeaderColor = PanelColor;

    ImColor const MonospaceColor = ImColor::HSV(0.3f, 1.0f, 1.0f);
    ImColor const StatusBarTextColor = ImColor::HSV(0.0f, 0.0f, 1.0f);

    ImColor const HeadlineColor = AccentColor;

    ImColor const UnsavedChangesColor = WarningColor;
    ImColor const UnsavedChangesBackgroundColor = ImColor::HSV(0.094f, 0.545f, 0.216f);

    ImColor const SelectionAreaFillColor = ImColor::HSV(0.33f, 0.0f, 1.0f, 0.6f);
    ImColor const SelectionAreaBorderColor = ImColor::HSV(0.33f, 0.0f, 1.0f, 1.0f);

    ImColor const ConstructionPreviewLineColor = ImColor::HSV(0.54f, 0.3f, 1.0f, 0.9f);
    ImColor const ConstructionPreviewHintLineColor = ImColor::HSV(0.54f, 0.3f, 1.0f, 0.4f);
    ImColor const ConstructionPreviewPointColor = ImColor::HSV(0.54f, 0.0f, 1.0f, 1.0f);
    ImColor const ConstructionPreviewBrushColor = ImColor::HSV(0.54f, 0.3f, 1.0f, 0.6f);

    ImColor const CellTypeOverlayColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 0.5f);
    ImColor const CellTypeOverlayShadowColor = ImColor::HSV(0.0f, 0.0f, 0.0f, 0.7f);
    ImColor const ExecutionNumberOverlayColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 0.8f);
    ImColor const ExecutionNumberOverlayShadowColor = ImColor::HSV(0.0f, 0.0f, 0.0f, 0.7f);

    ImColor const SelectedObjectOverlayColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 0.5f);

    // Icon buttons keep a light blue tint so they stand out against the neutral panels
    ImColor const ToolbarButtonTextColor = ImColor::HSV(0.530f, 0.320f, 0.950f);
    ImColor const ToolbarButtonBackgroundColor = ImColor::HSV(0.0f, 0.0f, 0.0f, 0.0f);
    ImColor const ToolbarButtonHoveredColor = RaisedColor;
    ImColor const ToolbarButtonSelectedColor = AccentDeepColor;
    ImColor const ToolbarButtonSelectedTextColor = AccentColor;
    ImColor const ToolbarButtonDisabledTextColor = TextFaintColor;
    ImColor const ToolbarSelectionBarColor = AccentColor;
    ImColor const ToolbarGroupColor = ImColor::HSV(0.583f, 0.100f, 1.000f, 0.050f);
    ImColor const ToolbarOverflowColor = TextFaintColor;
    ImColor const ToolbarOverflowHoveredColor = AccentColor;
    ImColor const ToolbarMenuHoveredColor = AccentDeepColor;

    // Floats above the simulation, therefore a semi-transparent background
    ImColor const EditToggleColor = ImColor::HSV(0.583f, 0.323f, 0.122f, 0.850f);
    ImColor const EditToggleSelectedColor = ImColor::HSV(0.490f, 0.556f, 0.247f, 0.900f);
    ImColor const EditToggleBorderColor = LineColor;
    ImColor const EditToggleSelectedBorderColor = AccentColor;
    ImColor const EditToggleGlowColor = ImColor::HSV(0.484f, 0.587f, 0.816f, 0.180f);
    ImColor const EditToggleIconColor = TextDimColor;
    ImColor const EditToggleHoveredIconColor = TextBaseColor;
    ImColor const EditToggleSelectedIconColor = AccentColor;
    ImColor const EditToggleLabelColor = TextBaseColor;
    ImColor const EditToggleShortcutColor = TextFaintColor;

    ImColor const ActionButtonTextColor = ImColor::HSV(0.530f, 0.400f, 0.950f);
    ImColor const ActionButtonHighlightedTextColor = AccentColor;
    ImColor const ActionButtonBackgroundColor = RaisedColor;
    ImColor const ActionButtonHoveredColor = ImColor::HSV(0.583f, 0.321f, 0.220f);
    ImColor const ActionButtonActiveColor = AccentDeepColor;

    ImColor const ButtonColor = ImColor::HSV(0.54f, 0.33f, 1.0f, 1.0f);
    ImColor const ToggleOnColor = AccentLineColor;
    ImColor const ToggleOnHoveredColor = AccentColor;
    ImColor const ToggleOffColor = LineSoftColor;
    ImColor const ToggleOffHoveredColor = LineColor;
    ImColor const ToggleKnobColor = TextBaseColor;
    ImColor const ToggleKnobBorderColor = BackgroundColor;
    ImColor const DetailButtonColor = ImColor::HSV(0, 0, 1.0f);

    ImColor const InspectorLineColor = ImColor::HSV(0.54f, 0.0f, 1.0f, 1.0f);
    ImColor const InspectorRectColor = ImColor::HSV(0.54f, 0.0f, 0.5f, 1.0f);

    ImColor const CursorShadowColor = ImColor::HSV(0, 0, 0, 1.0f);
    ImColor const CursorColor = ImColor::HSV(0, 0.0f, 1.0f, 1.0f);

    ImColor const GenomePreviewBackgroundColor = ImColor::HSV(0.667f, 1.0f, 0.106f);
    ImColor const GenomePreviewSeparatorColor = ImColor::HSV(0, 0, 0.25f);
    ImColor const GenomePreviewConnectionColor = ImColor::HSV(0, 0, 0.5f);
    ImColor const GenomePreviewInactiveColor = ImColor::HSV(0, 0, 0.15f);
    ImColor const GenomePreviewDotSymbolColor = ImColor::HSV(0, 0, 0.7f);
    ImColor const GenomePreviewGeneRefBackgroundColor1 = ImColor::HSV(0, 0, 1.0f);
    ImColor const GenomePreviewGeneRefBackgroundColor2 = ImColor::HSV(0, 0, 0.6f);
    ImColor const GenomePreviewLinkToGeneTextColor = ImColor::HSV(0, 0, 0);
    ImColor const GenomePreviewStartColor = ImColor::HSV(0.58f, 0.8f, 1.0f, 1.0f);
    ImColor const GenomePreviewEndColor = ImColor::HSV(0.0f, 0.8f, 1.0f, 1.0f);
    ImColor const GenomePreviewMultipleConstructorColor = ImColor::HSV(0.375f, 0.8f, 1.0f, 1.0f);
    ImColor const GenomePreviewSelfReplicatorColor = ImColor::HSV(0.79f, 0.8f, 1.0f, 1.0f);

    ImColor const FloatingCardBackgroundColor = ImColor::HSV(0.583f, 0.323f, 0.122f, 0.95f);
    ImColor const FloatingCardBorderColor = LineColor;

    ImColor const NeuronEditorConnectionColor = ImColor::HSV(0.0f, 0.0f, 0.1f);
    ImColor const NeuronEditorGridColor = ImColor::HSV(0.0f, 0.0f, 0.2f);
    ImColor const NeuronEditorZeroLinePlotColor = ImColor::HSV(0.6f, 1.0f, 0.7f);
    ImColor const NeuronEditorPlotColor = ImColor::HSV(0.0f, 0.0f, 1.0f);

    ImColor const BrowserAddReactionButtonTextColor = ImColor::HSV(0.375f, 0.6f, 0.7f, 1.0f);
    ImColor const BrowserDownloadButtonTextColor = ImColor::HSV(0.55f, 0.6f, 1.0f, 1.0f);
    ImColor const BrowserDeleteButtonTextColor = ImColor::HSV(0.0f, 0.6f, 0.8f, 1.0f);
    ImColor const BrowserLeafTextColor = ImColor::HSV(0.58f, 0.2f, 1.0f);
    ImColor const BrowserResourceTextColor = ImColor::HSV(0.0f, 0.0f, 1.0f);
    ImColor const BrowserResourceLineColor = ImColor::HSV(0.0f, 0.0f, 0.5f);
    ImColor const BrowserResourceNewTextColor = ImColor::HSV(0.15f, 0.8f, 1.0f);
    ImColor const BrowserResourceSymbolColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 1.0f);

    ImColor const BrowserLoginBannerColor = ImColor::HSV(0.490f, 0.556f, 0.247f, 0.500f);
    ImColor const BrowserLoginBannerBarColor = AccentColor;

    ImColor const BrowserLoginHintCardColor = ImColor::HSV(0.583f, 0.323f, 0.122f, 0.940f);
    ImColor const BrowserLoginHintCardBorderColor = LineColor;
    ImColor const BrowserLoginHintIconColor = AccentColor;
    ImColor const BrowserPlaceholderTilePictureColor = ImColor::HSV(0.583f, 0.300f, 0.090f);
    ImColor const BrowserPlaceholderTileBarColor = LineSoftColor;
}

void StyleService::setup()
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

void StyleService::setupSizes(ImGuiStyle& style) const
{
    style.FrameRounding = scale(6.0f);
    style.ChildRounding = scale(8.0f);
    style.PopupRounding = scale(8.0f);
    style.GrabRounding = scale(6.0f);
    style.ScrollbarRounding = scale(6.0f);
    style.TabRounding = scale(6.0f);
}

void StyleService::setupColors(ImGuiStyle& style) const
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
    style.Colors[ImGuiCol_TabHovered] = Const::HeaderActiveColor.Value;
    style.Colors[ImGuiCol_TabSelected] = Const::AccentDeepColor.Value;
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

ImFont* StyleService::getIconFont() const
{
    return _iconFont;
}

ImFont* StyleService::getDefaultFont() const
{
    return ImGui::GetIO().Fonts->Fonts[0];
}

ImFont* StyleService::getTinyFont() const
{
    return _tinyFont;
}

ImFont* StyleService::getSmallBoldFont() const
{
    return _smallBoldFont;
}

ImFont* StyleService::getMediumBoldFont() const
{
    return _mediumBoldFont;
}

ImFont* StyleService::getMediumFont() const
{
    return _mediumFont;
}

ImFont* StyleService::getLargeFont() const
{
    return _largeFont;
}

ImFont* StyleService::getMonospaceMediumFont() const
{
    return _monospaceMediumFont;
}

ImFont* StyleService::getMonospaceLargeFont() const
{
    return _monospaceLargeFont;
}

ImFont* StyleService::getReefMediumFont() const
{
    return _reefMediumFont;
}

ImFont* StyleService::getReefLargeFont() const
{
    return _reefLargeFont;
}

float StyleService::scale(float value) const
{
    return WindowController::get().getContentScaleFactor() * value;
}

float StyleService::scaleInverse(float value) const
{
    return value / WindowController::get().getContentScaleFactor();
}
