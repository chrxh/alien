#pragma once

#include <imgui.h>

#include "Base/GlobalSettings.h"

#include "Definitions.h"
#include "StyleRepository.h"
#include "MainLoopEntity.h"
#include "MainLoopEntityController.h"
#include "WindowController.h"

template<typename ...Dependencies>
class AlienWindow : public MainLoopEntity
{
public:
    AlienWindow(std::string const& title, std::string const& settingsNode, bool defaultOn);

    void init(Dependencies... dependencies);

    bool isOn() const;
    void setOn(bool value);


protected:
    virtual void initIntern(Dependencies... dependencies) {}
    virtual void shutdownIntern() {}
    virtual void processIntern() = 0;
    virtual void processBackground() {}
    virtual void processActivated() {}

    bool _sizeInitialized = false;
    bool _on = false;
    bool _defaultOn = false;
    std::string _title; 
    std::string _settingsNode;

private:
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
    _on = GlobalSettings::get().getBool(_settingsNode + ".active", _defaultOn);
    MainLoopEntityController::get().registerObject(this);
    initIntern(dependencies...);
}

template <typename ... Dependencies>
void AlienWindow<Dependencies...>::process()
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
    GlobalSettings::get().setBool(_settingsNode + ".active", _on);
}
