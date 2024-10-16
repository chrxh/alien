#include "AlienWindow.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"

#include "StyleRepository.h"
#include "WindowController.h"

AlienWindow::AlienWindow(std::string const& title, std::string const& settingsNode, bool defaultOn)
    : _title(title)
    , _settingsNode(settingsNode)
{
    _on = GlobalSettings::get().getBool(settingsNode + ".active", defaultOn);
}

AlienWindow::~AlienWindow()
{
    GlobalSettings::get().setBool(_settingsNode + ".active", _on);
}

void AlienWindow::process()
{
    processBackground();

    if (!_on) {
        return;
    }
    ImGui::PushID(_title.c_str());

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin(_title.c_str(), &_on)) {
        if (!_sizeInitialized) {
            auto size = ImGui::GetWindowSize();
            auto factor = WindowController::getContentScaleFactor() / WindowController::getLastContentScaleFactor();
            ImGui::SetWindowSize({size.x * factor, size.y * factor});
            _sizeInitialized = true;
        }
        processIntern();
    }
    ImGui::End();

    ImGui::PopID();
}

bool AlienWindow::isOn() const
{
    return _on;
}

void AlienWindow::setOn(bool value)
{
    _on = value;
    if (value) {
        processActivated();
    }
}
