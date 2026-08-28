#include "DisplaySettingsDialog.h"

#include <sstream>

#include <imgui.h>

#include <Base/LoggingService.h>

#include "AlienGui.h"
#include "GenericMessageDialog.h"
#include "MainLoopController.h"
#include "StyleRepository.h"
#include "WindowController.h"

#include <GLFW/glfw3.h>

namespace
{
    auto const RightColumnWidth = 270.0f;
}

void DisplaySettingsDialog::initIntern()
{
    auto primaryMonitor = glfwGetPrimaryMonitor();
    _videoModes = glfwGetVideoModes(primaryMonitor, &_videoModesCount);
    _videoModeStrings = createVideoModeStrings();
}

DisplaySettingsDialog::DisplaySettingsDialog()
    : AlienDialog("Display settings", {700.0f, 300.0f})
{}

void DisplaySettingsDialog::processIntern()
{
    auto isFullscreen = _pendingIsFullscreen;

    if (AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Full screen"), isFullscreen)) {
        _pendingIsFullscreen = isFullscreen;
    }

    ImGui::BeginDisabled(!_pendingIsFullscreen);

    AlienGui::Combo(
        AlienGui::ComboParameters().name("Resolution").textWidth(RightColumnWidth).defaultValue(_origSelectionIndex).values(_videoModeStrings),
        _pendingSelectionIndex);

    ImGui::EndDisabled();

    AlienGui::SliderInt(
        AlienGui::SliderIntParameters()
            .name("Frames per second")
            .textWidth(RightColumnWidth)
            .defaultValue(&_origFps)
            .min(20)
            .max(100)
            .tooltip("A high frame rate leads to a greater GPU workload for rendering and thus lowers the simulation speed (time steps per second)."),
        &_pendingFps);

    AlienGui::Checkbox(
        AlienGui::CheckboxParameters()
            .name("Adopt scaling from OS")
            .textWidth(RightColumnWidth)
            .defaultValue(_origAutoContentScaleFactor)
            .tooltip("If enabled, the scaling below is taken from the display settings of the operating system."),
        _pendingAutoContentScaleFactor);

    ImGui::BeginDisabled(_pendingAutoContentScaleFactor);
    AlienGui::SliderFloat(
        AlienGui::SliderFloatParameters()
            .name("Content scaling")
            .textWidth(RightColumnWidth)
            .defaultValue(&_origContentScaleFactor)
            .min(WindowController::MinContentScaleFactor)
            .max(WindowController::MaxContentScaleFactor)
            .format("%.2f"),
        &_pendingContentScaleFactor);
    ImGui::EndDisabled();

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button("Adopt")) {
        close();
        if (_pendingIsFullscreen) {
            setFullscreen(_pendingSelectionIndex);
        } else {
            WindowController::get().setWindowedMode();
        }
        WindowController::get().setFps(_pendingFps);
        _selectionIndex = _pendingSelectionIndex;

        if (isContentScalingChanged()) {
            auto automatic = _pendingAutoContentScaleFactor;
            auto scaleFactor = _pendingContentScaleFactor;
            delayedExecution([this, automatic, scaleFactor] { onChangeContentScaling(automatic, scaleFactor); });
        }
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void DisplaySettingsDialog::openIntern()
{
    _selectionIndex = getSelectionIndex();
    _origSelectionIndex = _selectionIndex;
    _origFps = WindowController::get().getFps();
    _pendingIsFullscreen = !WindowController::get().isWindowedMode();
    _origAutoContentScaleFactor = WindowController::get().isAutoContentScaleFactor();
    _origContentScaleFactor = WindowController::get().getUserDefinedContentScaleFactor();
    _pendingSelectionIndex = _selectionIndex;
    _pendingFps = _origFps;
    _pendingAutoContentScaleFactor = _origAutoContentScaleFactor;
    _pendingContentScaleFactor = _origContentScaleFactor;
}

bool DisplaySettingsDialog::isContentScalingChanged() const
{
    if (_pendingAutoContentScaleFactor != _origAutoContentScaleFactor) {
        return true;
    }
    return !_pendingAutoContentScaleFactor && _pendingContentScaleFactor != _origContentScaleFactor;
}

void DisplaySettingsDialog::onChangeContentScaling(bool automatic, float scaleFactor) const
{
    GenericMessageDialog::get().yesNo(
        "Content scaling", "The content scaling only takes effect at program start. Do you want to save it and terminate ALIEN now?", [automatic, scaleFactor] {
            WindowController::get().setAutoContentScaleFactor(automatic);
            if (!automatic) {
                WindowController::get().setUserDefinedContentScaleFactor(scaleFactor);
            }
            MainLoopController::get().scheduleClosing();
        });
}

void DisplaySettingsDialog::setFullscreen(int selectionIndex)
{
    if (0 == selectionIndex) {
        WindowController::get().setDesktopMode();
    } else {
        WindowController::get().setUserDefinedResolution(_videoModes[selectionIndex - 1]);
    }
}

namespace
{
    bool operator==(GLFWvidmode const& m1, GLFWvidmode const& m2)
    {
        return m1.width == m2.width && m1.height == m2.height && m1.redBits == m2.redBits && m1.greenBits == m2.greenBits && m1.blueBits == m2.blueBits
            && m1.refreshRate == m2.refreshRate;
    }
}

int DisplaySettingsDialog::getSelectionIndex() const
{
    auto result = 0;
    if (!WindowController::get().isWindowedMode() && !WindowController::get().isDesktopMode()) {
        auto userMode = WindowController::get().getUserDefinedResolution();
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

std::vector<std::string> DisplaySettingsDialog::createVideoModeStrings() const
{
    std::vector<std::string> result;
    result.emplace_back("Desktop");
    for (int i = 0; i < _videoModesCount; ++i) {
        result.emplace_back(createVideoModeString(_videoModes[i]));
    }

    return result;
}
