#pragma once

#include "AlienDialog.h"

class _ChangeColorDialog : public _AlienDialog
{
public:
    _ChangeColorDialog();

private:
    void processIntern() override;

    int _sourceColor = 0;
    int _targetColor = 0;
};
