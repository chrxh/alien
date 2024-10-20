#pragma once

#include "Base/Singleton.h"

#include "AlienDialog.h"
#include "Definitions.h"

class ExitDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ExitDialog);

public:
    void init(bool& onExit);

private:
    ExitDialog();

    void processIntern() override;

    bool* _onExit = nullptr;
};
