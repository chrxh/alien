#pragma once

#include <cstdint>

#include <imgui.h>

#include <Base/Singleton.h>

#include "Definitions.h"

namespace Const
{
    extern float const WindowAlpha;
    extern float const MaximizedWindowAlpha;
    extern float const SliderBarWidth;
    extern float const WindowsRounding;

    extern ImColor const BackgroundColor;
    extern ImColor const PanelColor;
    extern ImColor const RaisedColor;
    extern ImColor const InputColor;
    extern ImColor const LineColor;
    extern ImColor const LineSoftColor;
    extern ImColor const TextBaseColor;
    extern ImColor const TextDimColor;
    extern ImColor const TextFaintColor;
    extern ImColor const AccentColor;
    extern ImColor const AccentDeepColor;
    extern ImColor const AccentLineColor;
    extern ImColor const WarningColor;
    extern ImColor const DangerColor;

    extern ImColor const ProgramVersionTextColor;

    extern ImColor const RenderingDisabledTextColor;

    extern int64_t const SimulationSliderColor_Base;
    extern int64_t const SimulationSliderColor_Active;

    extern ImColor const TextTooltipColor;
    extern ImColor const TextInfoColor;
    extern ImColor const TextDecentColor;
    extern ImColor const TextConflictColor;

    extern ImColor const HeaderColor;
    extern ImColor const HeaderActiveColor;
    extern ImColor const HeaderHoveredColor;
    extern ImColor const HeaderSelectedHoveredColor;

    extern ImColor const MenuButtonColor;
    extern ImColor const MenuButtonHoveredColor;
    extern ImColor const MenuButtonActiveColor;

    extern ImColor const ImportantButtonColor;
    extern ImColor const ImportantButtonHoveredColor;
    extern ImColor const ImportantButtonActiveColor;

    extern ImColor const TreeNodeHighColor;
    extern ImColor const TreeNodeHighHoveredColor;
    extern ImColor const TreeNodeHighActiveColor;
    extern ImColor const TreeNodeDefaultColor;
    extern ImColor const TreeNodeDefaultHoveredColor;
    extern ImColor const TreeNodeDefaultActiveColor;
    extern ImColor const TreeNodeLowColor;
    extern ImColor const TreeNodeLowHoveredColor;
    extern ImColor const TreeNodeLowActiveColor;

    extern ImColor const DisabledOverlayColor1;
    extern ImColor const DisabledOverlayColor2;

    extern ImColor const GroupDefaultColor;
    extern ImColor const GroupHighColor;
    extern ImColor const GroupAccentBarColor;
    extern ImColor const GroupTextColor;
    extern ImColor const GroupHighTextColor;

    extern ImColor const MovableSeparatorColor;
    extern ImColor const MovableSeparatorHoveredColor;
    extern ImColor const MovableSeparatorActiveColor;

    extern ImColor const TableHeaderColor;

    extern ImColor const MonospaceColor;
    extern ImColor const StatusBarTextColor;

    extern ImColor const HeadlineColor;

    extern ImColor const UnsavedChangesColor;
    extern ImColor const UnsavedChangesBackgroundColor;

    extern ImColor const SelectionAreaFillColor;
    extern ImColor const SelectionAreaBorderColor;

    extern ImColor const ConstructionPreviewLineColor;
    extern ImColor const ConstructionPreviewHintLineColor;
    extern ImColor const ConstructionPreviewPointColor;
    extern ImColor const ConstructionPreviewBrushColor;

    extern ImColor const CellTypeOverlayColor;
    extern ImColor const CellTypeOverlayShadowColor;
    extern ImColor const ExecutionNumberOverlayColor;
    extern ImColor const ExecutionNumberOverlayShadowColor;

    extern ImColor const SelectedObjectOverlayColor;

    extern ImColor const ToolbarButtonTextColor;
    extern ImColor const ToolbarButtonBackgroundColor;
    extern ImColor const ToolbarButtonHoveredColor;
    extern ImColor const ToolbarButtonSelectedColor;
    extern ImColor const ToolbarButtonSelectedTextColor;
    extern ImColor const ToolbarButtonDisabledTextColor;
    extern ImColor const ToolbarSelectionBarColor;
    extern ImColor const ToolbarGroupColor;
    extern ImColor const ToolbarOverflowColor;
    extern ImColor const ToolbarOverflowHoveredColor;
    extern ImColor const ToolbarMenuHoveredColor;

    extern ImColor const EditToggleColor;
    extern ImColor const EditToggleSelectedColor;
    extern ImColor const EditToggleBorderColor;
    extern ImColor const EditToggleSelectedBorderColor;
    extern ImColor const EditToggleGlowColor;
    extern ImColor const EditToggleIconColor;
    extern ImColor const EditToggleHoveredIconColor;
    extern ImColor const EditToggleSelectedIconColor;
    extern ImColor const EditToggleLabelColor;
    extern ImColor const EditToggleShortcutColor;

    extern ImColor const ActionButtonTextColor;
    extern ImColor const ActionButtonHighlightedTextColor;
    extern ImColor const ActionButtonBackgroundColor;
    extern ImColor const ActionButtonHoveredColor;
    extern ImColor const ActionButtonActiveColor;

    extern ImColor const ButtonColor;
    extern ImColor const ToggleOnColor;
    extern ImColor const ToggleOnHoveredColor;
    extern ImColor const ToggleOffColor;
    extern ImColor const ToggleOffHoveredColor;
    extern ImColor const ToggleKnobColor;
    extern ImColor const ToggleKnobBorderColor;
    extern ImColor const DetailButtonColor;

    extern ImColor const InspectorLineColor;
    extern ImColor const InspectorRectColor;

    extern ImColor const CursorShadowColor;
    extern ImColor const CursorColor;

    extern ImColor const GenomePreviewBackgroundColor;
    extern ImColor const GenomePreviewSeparatorColor;
    extern ImColor const GenomePreviewConnectionColor;
    extern ImColor const GenomePreviewInactiveColor;
    extern ImColor const GenomePreviewDotSymbolColor;
    extern ImColor const GenomePreviewGeneRefBackgroundColor1;
    extern ImColor const GenomePreviewGeneRefBackgroundColor2;
    extern ImColor const GenomePreviewLinkToGeneTextColor;
    extern ImColor const GenomePreviewStartColor;
    extern ImColor const GenomePreviewEndColor;
    extern ImColor const GenomePreviewMultipleConstructorColor;
    extern ImColor const GenomePreviewSelfReplicatorColor;

    extern ImColor const FloatingCardBackgroundColor;
    extern ImColor const FloatingCardBorderColor;

    extern ImColor const NeuronEditorConnectionColor;
    extern ImColor const NeuronEditorGridColor;
    extern ImColor const NeuronEditorZeroLinePlotColor;
    extern ImColor const NeuronEditorPlotColor;

    extern ImColor const BrowserAddReactionButtonTextColor;
    extern ImColor const BrowserDownloadButtonTextColor;
    extern ImColor const BrowserDeleteButtonTextColor;
    extern ImColor const BrowserLeafTextColor;
    extern ImColor const BrowserResourceTextColor;
    extern ImColor const BrowserResourceLineColor;
    extern ImColor const BrowserResourceNewTextColor;
    extern ImColor const BrowserResourceSymbolColor;

    extern ImColor const BrowserLoginBannerColor;
    extern ImColor const BrowserLoginBannerBarColor;

    extern ImColor const BrowserLoginHintCardColor;
    extern ImColor const BrowserLoginHintCardBorderColor;
    extern ImColor const BrowserLoginHintIconColor;
    extern ImColor const BrowserPlaceholderTilePictureColor;
    extern ImColor const BrowserPlaceholderTileBarColor;
}

class StyleService
{
    MAKE_SINGLETON(StyleService);

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
    return StyleService::get().scale(value);
}

inline float scaleInverse(float value)
{
    return StyleService::get().scaleInverse(value);
}
