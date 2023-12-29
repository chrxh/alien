#pragma once

#include "Definitions.h"

#include <cstdint>

#include <imgui.h>

namespace Const
{
    ImColor const ProgramVersionTextColor = ImColor::HSV(0.5f, 0.1f, 1.0f, 1.0f);

    ImColor const RenderingDisabledTextColor = ImColor::HSV(0.5f, 0.1f, 1.0f, 0.2f);

    int64_t const SimulationSliderColor_Base = 0xff4c4c4c;
    int64_t const SimulationSliderColor_Active = 0xff6c6c6c;
    int64_t const TextDecentColor = 0xff909090;
    int64_t const TextInfoColor = 0xff308787;

    ImColor const MenuButtonColor = ImColor::HSV(0.6f, 0.6f, 0.5f);
    ImColor const MenuButtonHoveredColor = ImColor::HSV(0.6f, 1.0f, 1.0f);
    ImColor const MenuButtonActiveColor = ImColor::HSV(0.6f, 0.8f, 0.7f);

    ImColor const ShutdownButtonColor = ImColor::HSV(0.0f, 0.6f, 0.6f);
    ImColor const ShutdownButtonHoveredColor = ImColor::HSV(0.0f, 1.0f, 1.0f);
    ImColor const ShutdownButtonActiveColor = ImColor::HSV(0.0f, 1.0f, 1.0f);

    ImColor const ToggleButtonColor = ImColor::HSV(0.0f, 0.0f, 0.2f);
    ImColor const ToggleButtonHoveredColor = ImColor::HSV(0.14, 0.8, 0.5);
    ImColor const ToggleButtonActiveColor = ImColor::HSV(0.14, 0.8, 0.7);
    ImColor const ToggleButtonActiveHoveredColor = ImColor::HSV(0.14, 0.8, 0.8);

    ImColor const MonospaceColor = ImColor::HSV(0.3f, 1.0f, 1.0f);

    ImColor const HeadlineColor = ImColor::HSV(0.4f, 0.4f, 0.8f);

    ImColor const SelectionAreaFillColor = ImColor::HSV(0.33f, 0.0f, 1.0f, 0.6f);
    ImColor const SelectionAreaBorderColor = ImColor::HSV(0.33f, 0.0f, 1.0f, 1.0f);

    ImColor const CellFunctionOverlayColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 0.5f);
    ImColor const CellFunctionOverlayShadowColor = ImColor::HSV(0.0f, 0.0f, 0.0f, 0.7f);
    ImColor const ExecutionNumberOverlayColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 0.8f);
    ImColor const ExecutionNumberOverlayShadowColor = ImColor::HSV(0.0f, 0.0f, 0.0f, 0.7f);

    ImColor const SelectedCellOverlayColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 0.5f);

    ImColor const ToolbarButtonTextColor = ImColor::HSV(0.54f, 0.33f, 1.0f, 1.0f);
    ImColor const ToolbarButtonBackgroundColor = ImColor::HSV(0, 0, 0.06f, 0);
    ImColor const ToolbarButtonHoveredColor = ImColor::HSV(0, 0, 1, 0.35f);

    ImColor const ButtonColor = ImColor::HSV(0.54f, 0.33f, 1.0f, 1.0f);
    ImColor const ToggleColor = ImColor::HSV(0.58f, 0.83f, 1.0f, 1.0f);
    ImColor const DetailButtonColor = ImColor::HSV(0, 0, 1.0f);

    ImColor const InspectorLineColor = ImColor::HSV(0.54f, 0.0f, 1.0f, 1.0f);
    ImColor const InspectorRectColor = ImColor::HSV(0.54f, 0.0f, 0.5f, 1.0f);

    ImColor const NavigationCursorColor = ImColor::HSV(0, 0.0f, 1.0f, 0.4f);
    ImColor const EditCursorColor = ImColor::HSV(0.6, 0.6f, 1.0f, 0.7f);

    ImColor const GenomePreviewConnectionColor = ImColor::HSV(0, 0, 0.5f);
    ImColor const GenomePreviewDotSymbolColor = ImColor::HSV(0, 0, 0.7f);
    ImColor const GenomePreviewInfinitySymbolColor = ImColor::HSV(0, 0, 0.7f);
    ImColor const GenomePreviewStartColor = ImColor::HSV(0.58f, 0.8f, 1.0f, 1.0f);
    ImColor const GenomePreviewEndColor = ImColor::HSV(0.0f, 0.8f, 1.0f, 1.0f);
    ImColor const GenomePreviewMultipleConstructorColor = ImColor::HSV(0.375f, 0.8f, 1.0f, 1.0f);
    ImColor const GenomePreviewSelfReplicatorColor = ImColor::HSV(0.79f, 0.8f, 1.0f, 1.0f);

    ImColor const NeuronEditorConnectionColor = ImColor::HSV(0.0f, 0.0f, 0.1f);
    ImColor const NeuronEditorGridColor = ImColor::HSV(0.0f, 0.0f, 0.2f);
    ImColor const NeuronEditorZeroLinePlotColor = ImColor::HSV(0.6f, 1.0f, 0.7f);
    ImColor const NeuronEditorPlotColor = ImColor::HSV(0.0f, 0.0f, 1.0f);

    ImColor const BrowserAddReactionButtonTextColor = ImColor::HSV(0.375f, 0.6f, 0.7f, 1.0f);
    ImColor const BrowserDownloadButtonTextColor = ImColor::HSV(0.55f, 0.6f, 1.0f, 1.0f);
    ImColor const BrowserDeleteButtonTextColor = ImColor::HSV(0.0f, 0.6f, 0.8f, 1.0f);
    ImColor const BrowserFolderTextColor = ImColor::HSV(0.0f, 0.0f, 1.0f);
    ImColor const BrowserFolderLineColor = ImColor::HSV(0.0f, 0.0f, 0.5f);
    ImColor const BrowserFolderPropertiesTextColor = ImColor::HSV(0.0f, 0.0f, 0.5f, 1.0f);
    ImColor const BrowserFolderSymbolColor = ImColor::HSV(0.0f, 0.0f, 1.0f, 1.0f);

    ImColor const BrowserVersionOkTextColor = ImColor::HSV(0.58f, 0.0f, 1.0f);
    ImColor const BrowserVersionOutdatedTextColor = ImColor::HSV(0.0f, 0.0f, 0.6f);
    ImColor const BrowserVersionNewerTextColor = ImColor::HSV(0.0f, 0.2f, 1.0f);

    float const WindowAlpha = 0.9f;
    float const SliderBarWidth = 30.0f;
}

class StyleRepository
{
public:
    static StyleRepository& getInstance();
    StyleRepository(StyleRepository const&) = delete;

    void init();

    ImFont* getIconFont() const;

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
    StyleRepository() = default;

    ImFont* _iconFont = nullptr;
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
    return StyleRepository::getInstance().scale(value);
}
