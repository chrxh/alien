#pragma once

#include "AlienDialog.h"
#include "Definitions.h"

class _AboutDialog : public AlienDialog
{
public:
    _AboutDialog();

private:
    void processIntern() override;
};
