#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "AlienDialog.h"

class _ChangeColorDialog : public _AlienDialog
{
public:
    _ChangeColorDialog(std::function<GenomeDescription()> const& getGenomeFunc, std::function<void(GenomeDescription const&)> const& setGenomeFunc);

private:
    void processIntern() override;

    void onChangeColor(GenomeDescription& genome);

    std::function<GenomeDescription()> _getGenomeFunc;
    std::function<void(GenomeDescription const&)> _setGenomeFunc;
    int _sourceColor = 0;
    int _targetColor = 0;
    bool _includeSubGenomes = true;
};
