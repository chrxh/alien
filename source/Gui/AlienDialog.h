#pragma once

#include <imgui.h>

#include "Definitions.h"
#include "DelayedExecutionController.h"
#include "MainLoopEntity.h"
#include "MainLoopEntityController.h"
#include "StyleRepository.h"
#include "WindowController.h"

class AlienDialog : public MainLoopEntity
{
public:
    AlienDialog(std::string const& title, RealVector2D const& defaultSize = RealVector2D(450.0f, 150.0f));

    virtual void open();

protected:
    virtual void processIntern() {}
    virtual void initIntern() {}
    virtual void shutdownIntern() {}

    virtual void openIntern() {}

    void changeTitle(std::string const& title);
    virtual void close();

private:
    void init() override;
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
    RealVector2D _defaultSize;
};
