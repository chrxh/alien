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

    // Non-const colors depend on the color theme and are assigned by StyleService
    extern ImColor BackgroundColor;
    extern ImColor PanelColor;
    extern ImColor RaisedColor;
    extern ImColor InputColor;
    extern ImColor LineColor;
    extern ImColor LineSoftColor;
    extern ImColor TextStrongColor;
    extern ImColor TextBaseColor;
    extern ImColor TextDimColor;
    extern ImColor TextFaintColor;
    extern ImColor AccentColor;
    extern ImColor AccentDeepColor;
    extern ImColor AccentLineColor;
    extern ImColor WarningColor;
    extern ImColor DangerColor;

    extern ImColor const ProgramVersionTextColor;

    extern ImColor const RenderingDisabledTextColor;

    extern int64_t const SimulationSliderColor_Base;
    extern int64_t const SimulationSliderColor_Active;

    extern ImColor TextTooltipColor;
    extern ImColor TextInfoColor;
    extern ImColor TextDecentColor;
    extern ImColor TextConflictColor;

    extern ImColor HeaderColor;
    extern ImColor HeaderActiveColor;
    extern ImColor HeaderHoveredColor;
    extern ImColor HeaderSelectedHoveredColor;

    extern ImColor MenuButtonColor;
    extern ImColor MenuButtonHoveredColor;
    extern ImColor MenuButtonActiveColor;

    extern ImColor ImportantButtonColor;
    extern ImColor ImportantButtonHoveredColor;
    extern ImColor ImportantButtonActiveColor;

    extern ImColor TreeNodeHighColor;
    extern ImColor TreeNodeHighHoveredColor;
    extern ImColor TreeNodeHighActiveColor;
    extern ImColor TreeNodeDefaultColor;
    extern ImColor TreeNodeDefaultHoveredColor;
    extern ImColor TreeNodeDefaultActiveColor;
    extern ImColor TreeNodeLowColor;
    extern ImColor TreeNodeLowHoveredColor;
    extern ImColor TreeNodeLowActiveColor;

    extern ImColor DisabledOverlayColor1;
    extern ImColor DisabledOverlayColor2;

    extern ImColor GroupDefaultColor;
    extern ImColor GroupHighColor;
    extern ImColor GroupAccentBarColor;
    extern ImColor GroupTextColor;
    extern ImColor GroupHighTextColor;

    extern ImColor MovableSeparatorColor;
    extern ImColor MovableSeparatorHoveredColor;
    extern ImColor MovableSeparatorActiveColor;

    extern ImColor TableHeaderColor;
    extern ImColor TableRowAltColor;

    extern ImColor MonospaceColor;
    extern ImColor StatusBarTextColor;

    extern ImColor HeadlineColor;

    extern ImColor UnsavedChangesColor;
    extern ImColor UnsavedChangesBackgroundColor;

    extern ImColor const SelectionAreaFillColor;
    extern ImColor const SelectionAreaBorderColor;

    extern ImColor const ConstructionPreviewLineColor;
    extern ImColor const ConstructionPreviewHintLineColor;
    extern ImColor const ConstructionPreviewPointColor;
    extern ImColor const ConstructionPreviewBrushColor;

    extern ImColor McpSuccessColor;
    extern ImColor McpRunningBadgeColor;

    extern ImColor const CellTypeOverlayColor;
    extern ImColor const CellTypeOverlayShadowColor;
    extern ImColor const ExecutionNumberOverlayColor;
    extern ImColor const ExecutionNumberOverlayShadowColor;

    extern ImColor const SelectedObjectOverlayColor;

    extern ImColor ToolbarButtonTextColor;
    extern ImColor const ToolbarButtonBackgroundColor;
    extern ImColor ToolbarButtonHoveredColor;
    extern ImColor ToolbarButtonSelectedColor;
    extern ImColor ToolbarButtonSelectedTextColor;
    extern ImColor ToolbarButtonDisabledTextColor;
    extern ImColor ToolbarSelectionBarColor;
    extern ImColor ToolbarGroupColor;
    extern ImColor ToolbarOverflowColor;
    extern ImColor ToolbarOverflowHoveredColor;
    extern ImColor ToolbarMenuHoveredColor;

    extern ImColor EditToggleColor;
    extern ImColor EditToggleSelectedColor;
    extern ImColor EditToggleBorderColor;
    extern ImColor EditToggleSelectedBorderColor;
    extern ImColor EditToggleGlowColor;
    extern ImColor EditToggleIconColor;
    extern ImColor EditToggleHoveredIconColor;
    extern ImColor EditToggleSelectedIconColor;
    extern ImColor EditToggleLabelColor;
    extern ImColor EditToggleShortcutColor;

    extern ImColor ActionButtonTextColor;
    extern ImColor ActionButtonHighlightedTextColor;
    extern ImColor ActionButtonBackgroundColor;
    extern ImColor ActionButtonHoveredColor;
    extern ImColor ActionButtonActiveColor;

    extern ImColor const ButtonColor;
    extern ImColor ToggleOnColor;
    extern ImColor ToggleOnHoveredColor;
    extern ImColor ToggleOffColor;
    extern ImColor ToggleOffHoveredColor;
    extern ImColor ToggleKnobColor;
    extern ImColor ToggleKnobBorderColor;
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

    extern ImColor FloatingCardBackgroundColor;
    extern ImColor FloatingCardBorderColor;

    extern ImColor const SelectionFrameColor;
    extern ImColor SelectionHandleColor;
    extern ImColor SelectionHandleHoveredColor;
    extern ImColor SelectionHandleFillColor;
    extern ImColor SelectionChipTextColor;
    extern ImColor ScissorsTrailColor;
    extern ImColor const MultiplierPreviewColor;

    extern ImColor const NeuronEditorConnectionColor;
    extern ImColor const NeuronEditorGridColor;
    extern ImColor const NeuronEditorZeroLinePlotColor;
    extern ImColor const NeuronEditorPlotColor;
    extern ImColor NeuronEditorNodeFillColor;
    extern ImColor NeuronEditorLabelColor;

    extern ImColor DashboardCardBackgroundColor;
    extern ImColor DashboardCardBorderColor;
    extern ImColor DashboardSummaryRowColor;

    extern ImColor BrowserAddReactionButtonTextColor;
    extern ImColor BrowserDownloadButtonTextColor;
    extern ImColor BrowserDeleteButtonTextColor;
    extern ImColor BrowserLeafTextColor;
    extern ImColor BrowserResourceTextColor;
    extern ImColor const BrowserResourceLineColor;
    extern ImColor BrowserResourceNewTextColor;
    extern ImColor BrowserResourceSymbolColor;
    extern ImColor BrowserOwnReactionFrameColor;

    extern ImColor BrowserLoginBannerColor;
    extern ImColor BrowserLoginBannerBarColor;

    extern ImColor BrowserLoginHintCardColor;
    extern ImColor BrowserLoginHintCardBorderColor;
    extern ImColor BrowserLoginHintIconColor;
    extern ImColor BrowserPlaceholderTilePictureColor;
    extern ImColor BrowserPlaceholderTileBarColor;
}

class StyleService
{
    MAKE_SINGLETON(StyleService);

public:
    void setup();

    // Must be called before ImGui::NewFrame, since a theme change during a frame would be reverted by pending PopStyleColor calls
    void process();

    bool isLightMode() const;
    void setLightMode(bool value);

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
    void setupPalette() const;
    void setupColors(ImGuiStyle& style) const;

    bool _lightMode = false;
    bool _paletteOutdated = false;

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
