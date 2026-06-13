#pragma once

#include <functional>
#include <string>

#include <Base/Singleton.h>

#include <EngineInterface/GenomeDesc.h>

#include "AlienDialog.h"
#include "Definitions.h"

class MutationRatesDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(MutationRatesDialog);

public:
    void loadSettings(MutationRatesDesc& mutationRates, std::string const& settingsPrefix) const;
    void saveSettings(MutationRatesDesc const& mutationRates, std::string const& settingsPrefix) const;

    void open(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback);
    void openNested(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback);

private:
    MutationRatesDialog();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void openIntern() override;

    void onAdopt();

    MutationRatesDesc _mutation;
    std::function<void(MutationRatesDesc const&)> _onAdoptCallback;
};
