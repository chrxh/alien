#pragma once

#include <Base/Singleton.h>

#include "AlienDialog.h"
#include "Definitions.h"

class PreviewSettingsDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(PreviewSettingsDialog);

public:
    void setEditData(GenomeWindowEditData const& editData);

private:
    PreviewSettingsDialog();

    void processIntern() override;
    void openIntern() override;

    GenomeWindowEditData _editData;
    bool _showNodeIndex = true;
};
