#pragma once

#include <ImGui.h>

#include "Definitions.h"
#include "MainLoopEntity.h"
#include "MainLoopEntityController.h"
#include "WindowController.h"
#include "DelayedExecutionController.h"

template <typename... Dependencies>
class AlienDialog : public MainLoopEntity
{
public:
    AlienDialog(std::string const& title);

    void init(Dependencies... dependencies);

    virtual void open();

protected:
    virtual void processIntern() {}
    virtual void initIntern(Dependencies... dependencies) {}
    virtual void shutdownIntern() {}

    virtual void openIntern() {}

    void changeTitle(std::string const& title);
    virtual void close();

private:
    void process() override;
    void shutdown() override;

    bool _sizeInitialized = false;
    enum class DialogState
    {
        Closed,
        JustOpened,
        Open
    };
    DialogState _state = DialogState::Closed;
    std::string _title;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

template <typename ... Dependencies>
AlienDialog<Dependencies...>::AlienDialog(std::string const& title)
    : _title(title)
{}

template <typename ... Dependencies>
void AlienDialog<Dependencies...>::init(Dependencies... dependencies)
{
    MainLoopEntityController::get().registerObject(this);
    initIntern(dependencies...);
}

template <typename ... Dependencies>
void AlienDialog<Dependencies...>::open()
{
    _state = DialogState::JustOpened;
    openIntern();
}

template <typename ... Dependencies>
void AlienDialog<Dependencies...>::close()
{
    delayedExecution([this] {
        ImGui::CloseCurrentPopup();
        _state = DialogState::Closed;
    });
}

template <typename ... Dependencies>
void AlienDialog<Dependencies...>::changeTitle(std::string const& title)
{
    _title = title;
}

template <typename ... Dependencies>
void AlienDialog<Dependencies...>::process()
{
    if (_state == DialogState::Closed) {
        return;
    }
    if (_state == DialogState::JustOpened) {
        ImGui::OpenPopup(_title.c_str());
        _state = DialogState::Open;
    }

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal(_title.c_str(), NULL, 0)) {
        if (!_sizeInitialized) {
            auto size = ImGui::GetWindowSize();
            auto factor = WindowController::get().getContentScaleFactor() / WindowController::get().getLastContentScaleFactor();
            ImGui::SetWindowSize({size.x * factor, size.y * factor});
            _sizeInitialized = true;
        }

        ImGui::PushID(_title.c_str());
        processIntern();
        ImGui::PopID();

        ImGui::EndPopup();
    }
}

template <typename ... Dependencies>
void AlienDialog<Dependencies...>::shutdown()
{
    shutdownIntern();
}
