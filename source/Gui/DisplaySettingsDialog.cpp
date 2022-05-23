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
    auto const MaxContentTextWidth = 185.0f;
}

_DisplaySettingsDialog::_DisplaySettingsDialog(WindowController const& windowController)
    : _windowController(windowController)
{
    auto primaryMonitor = glfwGetPrimaryMonitor();
    _videoModes = glfwGetVideoModes(primaryMonitor, &_videoModesCount);
    _videoModeStrings = createVideoModeStrings();
}

_DisplaySettingsDialog::~_DisplaySettingsDialog()
{
}

void _DisplaySettingsDialog::process()
{
    if (!_show) {
        return;
    }

    ImGui::OpenPopup("Display settings");
    if (ImGui::BeginPopupModal("Display settings", NULL, ImGuiWindowFlags_None)) {
        auto isFullscreen = !_windowController->isWindowedMode();

        if(AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Full screen"), isFullscreen)) {
            if (isFullscreen) {
                setFullscreen(_selectionIndex);
            } else {
                _origSelectionIndex = _selectionIndex;
                _windowController->setWindowedMode();
            }
        }

        ImGui::BeginDisabled(!isFullscreen);

        if (AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Resolution")
                    .textWidth(MaxContentTextWidth)
                    .defaultValue(_origSelectionIndex)
                    .values(_videoModeStrings),
                _selectionIndex)) {

            setFullscreen(_selectionIndex);
        }
        ImGui::EndDisabled();

        auto fps = _windowController->getFps();
        if (AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name("Frames per second")
                    .textWidth(MaxContentTextWidth)
                    .defaultValue(_origFps)
                    .min(20)
                    .max(100)
                    .tooltip("A high frame rate leads to a greater GPU workload for rendering and thus lowers the simulation speed (time steps per second)."),
                fps)) {
            _windowController->setFps(fps);
        }

        AlienImGui::Separator();

        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }
        ImGui::SetItemDefaultFocus();

        ImGui::SameLine();
        if (AlienImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            _show = false;
            _windowController->setMode(_origMode);
            _windowController->setFps(_origFps);
            _selectionIndex = _origSelectionIndex;
        }

        ImGui::EndPopup();
    }
}

void _DisplaySettingsDialog::show()
{
    _show = true;
    _selectionIndex = getSelectionIndex();
    _origSelectionIndex = _selectionIndex;
    _origMode = _windowController->getMode();
    _origFps = _windowController->getFps();
}

void _DisplaySettingsDialog::setFullscreen(int selectionIndex)
{
    if (0 == selectionIndex) {
        _windowController->setDesktopMode();
    } else {
        _windowController->setUserDefinedResolution(_videoModes[selectionIndex - 1]);
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
    if (!_windowController->isWindowedMode() && !_windowController->isDesktopMode()) {
        auto userMode = _windowController->getUserDefinedResolution();
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
