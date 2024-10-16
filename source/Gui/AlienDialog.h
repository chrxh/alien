#pragma once

#include "Definitions.h"
#include "EngineInterface/Definitions.h"

class AlienDialog
{
public:
    AlienDialog(std::string const& title);
    virtual ~AlienDialog();

    void process();
    virtual void open();
    void close();

protected:
    virtual void processIntern() {}
    virtual void openIntern() {}

    void changeTitle(std::string const& title);

private:
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