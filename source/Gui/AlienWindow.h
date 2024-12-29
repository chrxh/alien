#pragma once

#include <imgui.h>

#include "Base/GlobalSettings.h"

#include "Definitions.h"
#include "StyleRepository.h"
#include "MainLoopEntity.h"
#include "MainLoopEntityController.h"
#include "WindowController.h"

template<typename ...Dependencies>
class AlienWindow : public MainLoopEntity<Dependencies...>
{
public:
    AlienWindow(std::string const& title, std::string const& settingsNode, bool defaultOn);

    bool isOn() const;
    void setOn(bool value);

protected:
    virtual void initIntern(Dependencies... dependencies) {}
    virtual void shutdownIntern() {}
    virtual void processIntern() = 0;
    virtual void processBackground() {}
    virtual void processActivated() {}

    virtual bool isShown() { return _on; }

    bool _sizeInitialized = false;
    bool _on = false;
    bool _defaultOn = false;
    std::string _title; 
    std::string _settingsNode;

private:
    void init(Dependencies... dependencies) override;
    void process() override;
    void shutdown() override;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

template <typename ... Dependencies>
AlienWindow<Dependencies...>::AlienWindow(std::string const& title, std::string const& settingsNode, bool defaultOn)
    : _title(title)
    , _settingsNode(settingsNode)
    , _defaultOn(defaultOn)
{}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::init(Dependencies... dependencies)
{
    _on = GlobalSettings::get().getValue(_settingsNode + ".active", _defaultOn);
    initIntern(dependencies...);
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::process()
{
    processBackground();

    if (!isShown()) {
        return;
    }
    ImGui::PushID(_title.c_str());

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({scale(650.0f), scale(350.0f)}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin(_title.c_str(), &_on)) {
        if (!_sizeInitialized) {
            auto size = ImGui::GetWindowSize();
            auto factor = WindowController::get().getContentScaleFactor() / WindowController::get().getLastContentScaleFactor();
            ImGui::SetWindowSize({size.x * factor, size.y * factor});
            _sizeInitialized = true;
        }
        processIntern();
    }
    ImGui::End();

    ImGui::PopID();
}

template <typename ... Dependencies>
bool AlienWindow<Dependencies...>::isOn() const
{
    return _on;
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::setOn(bool value)
{
    _on = value;
    if (value) {
        processActivated();
    }
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::shutdown()
{
    shutdownIntern();
    GlobalSettings::get().setValue(_settingsNode + ".active", _on);
}
