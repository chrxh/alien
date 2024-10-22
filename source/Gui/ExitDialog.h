#pragma once

#include "Base/Singleton.h"

#include "AlienDialog.h"
#include "Definitions.h"

class ExitDialog : public AlienDialog<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ExitDialog);

private:
    ExitDialog();

    void processIntern() override;
};
