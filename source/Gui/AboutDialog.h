#pragma once

#include "Definitions.h"

class _AboutDialog
{
public:
    _AboutDialog();

    void process();

    void show();

private:
    bool _show = false;
};
