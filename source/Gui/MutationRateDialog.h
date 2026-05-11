#pragma once

#include <functional>

#include <Base/Singleton.h>

#include <EngineInterface/GenomeDesc.h>

#include "AlienDialog.h"
#include "Definitions.h"

class MutationRateDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(MutationRateDialog);

public:
    void open(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback);

private:
    MutationRateDialog();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void openIntern() override;

    void onAdopt();

    MutationRatesDesc _mutation;
    std::function<void(MutationRatesDesc const&)> _onAdoptCallback;
};
