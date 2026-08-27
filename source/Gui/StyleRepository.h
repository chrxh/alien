#pragma once

#include <cstdint>

#include <imgui.h>

#include <Base/Singleton.h>

#include "Definitions.h"

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

    ImColor const MenuButtonColor = ImColor::HSV(0.583f, 0.333f, 0.212f);
    ImColor const MenuButtonHoveredColor = AccentColor;
    ImColor const MenuButtonActiveColor = AccentLineColor;

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

    ImColor const CellTypeOverlayColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 0.5f);
    ImColor const CellTypeOverlayShadowColor = ImColor::HSV(0.0f, 0.0f, 0.0f, 0.7f);
    ImColor const ExecutionNumberOverlayColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 0.8f);
    ImColor const ExecutionNumberOverlayShadowColor = ImColor::HSV(0.0f, 0.0f, 0.0f, 0.7f);

    ImColor const SelectedObjectOverlayColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 0.5f);

    // Icon buttons keep a light blue tint so they stand out against the neutral panels
    ImColor const ToolbarButtonTextColor = ImColor::HSV(0.530f, 0.320f, 0.950f);
    ImColor const ToolbarButtonBackgroundColor = ImColor::HSV(0.0f, 0.0f, 0.0f, 0.0f);
    ImColor const ToolbarButtonHoveredColor = RaisedColor;

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
}

class StyleRepository
{
    MAKE_SINGLETON(StyleRepository);

public:
    void setup();

    ImFont* getIconFont() const;

    ImFont* getDefaultFont() const;

    ImFont* getTinyFont() const;

    ImFont* getSmallBoldFont() const;
    ImFont* getMediumBoldFont() const;

    ImFont* getMediumFont() const;
    ImFont* getLargeFont() const;

    ImFont* getMonospaceMediumFont() const;
    ImFont* getMonospaceLargeFont() const;

    ImFont* getReefMediumFont() const;
    ImFont* getReefLargeFont() const;

    float scale(float value) const;
    float scaleInverse(float value) const;

private:
    void setupSizes(ImGuiStyle& style) const;
    void setupColors(ImGuiStyle& style) const;

    ImFont* _iconFont = nullptr;
    ImFont* _tinyFont = nullptr;
    ImFont* _smallBoldFont = nullptr;
    ImFont* _mediumBoldFont = nullptr;
    ImFont* _mediumFont = nullptr;
    ImFont* _largeFont = nullptr;
    ImFont* _monospaceMediumFont = nullptr;
    ImFont* _monospaceLargeFont = nullptr;
    ImFont* _reefMediumFont = nullptr;
    ImFont* _reefLargeFont = nullptr;
};

inline float scale(float value)
{
    return StyleRepository::get().scale(value);
}

inline float scaleInverse(float value)
{
    return StyleRepository::get().scaleInverse(value);
}
