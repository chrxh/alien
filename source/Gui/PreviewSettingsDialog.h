#pragma once

#include <Base/Singleton.h>

#include "AlienDialog.h"
#include "Definitions.h"

class PreviewSettingsDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(PreviewSettingsDialog);

public:
    void setEditData(GenomeTabEditData const& editData);

private:
    PreviewSettingsDialog();

    void processIntern() override;

    GenomeTabEditData _editData;
};
