#pragma once

#include "Base/Singleton.h"

#include "AlienDialog.h"
#include "Definitions.h"

class AboutDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(AboutDialog);

private:
    AboutDialog();
    void processIntern() override;
};
