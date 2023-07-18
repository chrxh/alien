#include "DisplaySettingsDialog.h"

#include <sstream>

#include <GLFW/glfw3.h>
#include <imgui.h>

#include "Base/LoggingService.h"

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "WindowController.h"
#include "StyleRepository.h"

namespace
{
    auto const RightColumnWidth = 185.0f;
}

_DisplaySettingsDialog::_DisplaySettingsDialog()
    : _AlienDialog("Display settings")
{
    auto primaryMonitor = glfwGetPrimaryMonitor();
    _videoModes = glfwGetVideoModes(primaryMonitor, &_videoModesCount);
    _videoModeStrings = createVideoModeStrings();
}

void _DisplaySettingsDialog::processIntern()
{
    auto isFullscreen = !WindowController::getInstance().isWindowedMode();

    if (AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Full screen"), isFullscreen)) {
        if (isFullscreen) {
            setFullscreen(_selectionIndex);
        } else {
            _origSelectionIndex = _selectionIndex;
            WindowController::getInstance().setWindowedMode();
        }
    }

    ImGui::BeginDisabled(!isFullscreen);

    if (AlienImGui::Combo(
            AlienImGui::ComboParameters().name("Resolution").textWidth(RightColumnWidth).defaultValue(_origSelectionIndex).values(_videoModeStrings),
            _selectionIndex)) {

        setFullscreen(_selectionIndex);
    }
    ImGui::EndDisabled();

    auto fps = WindowController::getInstance().getFps();
    if (AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Frames per second")
                .textWidth(RightColumnWidth)
                .defaultValue(&_origFps)
                .min(20)
                .max(100)
                .tooltip("A high frame rate leads to a greater GPU workload for rendering and thus lowers the simulation speed (time steps per second)."),
            &fps)) {
        WindowController::getInstance().setFps(fps);
    }

    AlienImGui::Separator();

    if (AlienImGui::Button("OK")) {
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
        WindowController::getInstance().setMode(_origMode);
        WindowController::getInstance().setFps(_origFps);
        _selectionIndex = _origSelectionIndex;
    }
}

void _DisplaySettingsDialog::openIntern()
{
    _selectionIndex = getSelectionIndex();
    _origSelectionIndex = _selectionIndex;
    _origMode = WindowController::getInstance().getMode();
    _origFps = WindowController::getInstance().getFps();
}

void _DisplaySettingsDialog::setFullscreen(int selectionIndex)
{
    if (0 == selectionIndex) {
        WindowController::getInstance().setDesktopMode();
    } else {
        WindowController::getInstance().setUserDefinedResolution(_videoModes[selectionIndex - 1]);
    }
}

namespace
{
    bool operator==(GLFWvidmode const& m1, GLFWvidmode const& m2)
    {
        return m1.width == m2.width && m1.height == m2.height && m1.redBits == m2.redBits
            && m1.greenBits == m2.greenBits && m1.blueBits == m2.blueBits && m1.refreshRate == m2.refreshRate;
    }
}

int _DisplaySettingsDialog::getSelectionIndex() const
{
    auto result = 0;
    auto& windowController = WindowController::getInstance();
    if (!windowController.isWindowedMode() && !windowController.isDesktopMode()) {
        auto userMode = windowController.getUserDefinedResolution();
        for (int i = 0; i < _videoModesCount; ++i) {
            if (_videoModes[i] == userMode) {
                return i + 1;
            }
        }
    }
    return result;
}

namespace
{
    std::string createVideoModeString(GLFWvidmode const& videoMode)
    {
        std::stringstream ss;
        ss << videoMode.width << " x " << videoMode.height << " @ " << videoMode.refreshRate << "Hz";
        return ss.str();
    }
}

std::vector<std::string> _DisplaySettingsDialog::createVideoModeStrings() const
{
    std::vector<std::string> result;
    result.emplace_back("Desktop");
    for (int i = 0; i < _videoModesCount; ++i) {
        result.emplace_back(createVideoModeString(_videoModes[i]));
    }

    return result;
}
