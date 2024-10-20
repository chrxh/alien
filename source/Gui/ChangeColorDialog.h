#pragma once

#include "EngineInterface/GenomeDescriptions.h"
#include "Base/Singleton.h"

#include "AlienDialog.h"

class ChangeColorDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ChangeColorDialog);

public:
    void init(std::function<GenomeDescription()> const& getGenomeFunc, std::function<void(GenomeDescription const&)> const& setGenomeFunc);

private:
    ChangeColorDialog();

    void processIntern() override;

    void onChangeColor(GenomeDescription& genome);

    std::function<GenomeDescription()> _getGenomeFunc;
    std::function<void(GenomeDescription const&)> _setGenomeFunc;
    int _sourceColor = 0;
    int _targetColor = 0;
    bool _includeSubGenomes = true;
};
