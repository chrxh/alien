#pragma once

#include "EngineInterface/CreatureDescription.h"
#include "Base/Singleton.h"

#include "AlienDialog.h"

class ChangeColorDialog : public AlienDialog<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ChangeColorDialog);

public:
    void open(CreatureTabEditData const& editData);

private:
    ChangeColorDialog();

    void processIntern() override;

    void onChangeColor();

    CreatureTabEditData _editData;
    int _sourceColor = 0;
    int _targetColor = 0;
    bool _restrictToSelectedGene = true;
};

