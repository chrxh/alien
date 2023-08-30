#pragma once

#include "AlienDialog.h"
#include "Definitions.h"

class _ExitDialog : public _AlienDialog
{
public:
    _ExitDialog(bool& onExit);

private:
    void processIntern() override;

    bool& _onExit;
};
