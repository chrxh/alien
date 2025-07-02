#pragma once

#include "EngineInterface/GenomeDescription.h"
#include "Base/Singleton.h"

#include "AlienDialog.h"

class ChangeColorDialog : public AlienDialog<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ChangeColorDialog);

public:
    void open(GenomeTabEditData const& editData);

private:
    ChangeColorDialog();

    void processIntern() override;

    void onChangeColor();

    GenomeTabEditData _editData;
    int _sourceColor = 0;
    int _targetColor = 0;
    bool _restrictToSelectedGene = true;
};

