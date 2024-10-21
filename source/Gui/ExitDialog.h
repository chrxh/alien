#pragma once

#include "Base/Singleton.h"

#include "AlienDialog.h"
#include "Definitions.h"

class ExitDialog : public AlienDialog<bool&>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ExitDialog);

private:
    ExitDialog();

    void initIntern(bool& onExit) override;
    void processIntern() override;

    bool* _onExit = nullptr;
};
