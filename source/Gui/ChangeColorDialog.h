#pragma once

#include "EngineInterface/GenomeDescriptions.h"
#include "Base/Singleton.h"

#include "AlienDialog.h"

class ChangeColorDialog
    : public AlienDialog<std::function<GenomeDescription()>, std::function<void(GenomeDescription const&)>>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ChangeColorDialog);

private:
    ChangeColorDialog();

    void initIntern(std::function<GenomeDescription()> getGenomeFunc, std::function<void(GenomeDescription const&)> setGenomeFunc) override;
    void processIntern() override;

    void onChangeColor(GenomeDescription& genome);

    std::function<GenomeDescription()> _getGenomeFunc;
    std::function<void(GenomeDescription const&)> _setGenomeFunc;
    int _sourceColor = 0;
    int _targetColor = 0;
    bool _includeSubGenomes = true;
};
