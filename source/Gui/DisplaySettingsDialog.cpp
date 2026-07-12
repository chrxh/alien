#include "DisplaySettingsDialog.h"

#include <sstream>

#include <imgui.h>

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>

#include "AlienGui.h"
#include "StyleRepository.h"
#include "TranslationService.h"
#include "WindowController.h"

#include <GLFW/glfw3.h>

namespace
{
    auto const RightColumnWidth = 185.0f;
    std::vector<std::string> const LanguageNames = {"English", "中文"};
    std::vector<std::string> const LanguageCodes = {"en", "zh"};
}

void DisplaySettingsDialog::initIntern()
{
    auto primaryMonitor = glfwGetPrimaryMonitor();
    _videoModes = glfwGetVideoModes(primaryMonitor, &_videoModesCount);
    _videoModeStrings = createVideoModeStrings();
}

DisplaySettingsDialog::DisplaySettingsDialog()
    : AlienDialog(_("Display settings"))
{}

void DisplaySettingsDialog::processIntern()
{
    auto isFullscreen = _pendingIsFullscreen;

    if (AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name(_("Full screen")), isFullscreen)) {
        _pendingIsFullscreen = isFullscreen;
    }

    ImGui::BeginDisabled(!_pendingIsFullscreen);

    AlienGui::Combo(
        AlienGui::ComboParameters().name(_("Resolution")).textWidth(RightColumnWidth).defaultValue(_origSelectionIndex).values(_videoModeStrings),
        _pendingSelectionIndex);

    ImGui::EndDisabled();

    if (AlienGui::Combo(
            AlienGui::ComboParameters().name(_("Language")).textWidth(RightColumnWidth).values(LanguageNames),
            _languageIndex)) {
        GlobalSettings::get().setValue("settings.language", LanguageCodes[_languageIndex]);
        TranslationService::get().load(LanguageCodes[_languageIndex]);
    }

    AlienGui::SliderInt(
        AlienGui::SliderIntParameters()
            .name(_("Frames per second"))
            .textWidth(RightColumnWidth)
            .defaultValue(&_origFps)
            .min(20)
            .max(100)
            .tooltip(_("A high frame rate leads to a greater GPU workload for rendering and thus lowers the simulation speed (time steps per second).")),
        &_pendingFps);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button(_("Adopt"))) {
        close();
        if (_pendingIsFullscreen) {
            setFullscreen(_pendingSelectionIndex);
        } else {
            WindowController::get().setWindowedMode();
        }
        WindowController::get().setFps(_pendingFps);
        _selectionIndex = _pendingSelectionIndex;
        GlobalSettings::get().setValue("settings.language", LanguageCodes[_languageIndex]);
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button(_("Cancel"))) {
        close();
        TranslationService::get().load(LanguageCodes[_origLanguageIndex]);
    }
}

void DisplaySettingsDialog::openIntern()
{
    _selectionIndex = getSelectionIndex();
    _origSelectionIndex = _selectionIndex;
    _origFps = WindowController::get().getFps();
    _pendingIsFullscreen = !WindowController::get().isWindowedMode();
    _pendingSelectionIndex = _selectionIndex;
    _pendingFps = _origFps;

    auto language = GlobalSettings::get().getValue("settings.language", std::string("en"));
    _languageIndex = 0;
    for (size_t i = 0; i < LanguageCodes.size(); ++i) {
        if (LanguageCodes[i] == language) {
            _languageIndex = static_cast<int>(i);
            break;
        }
    }
    _origLanguageIndex = _languageIndex;
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
    result.emplace_back(_("Desktop"));
    for (int i = 0; i < _videoModesCount; ++i) {
        result.emplace_back(createVideoModeString(_videoModes[i]));
    }

    return result;
}
