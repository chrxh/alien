#pragma once

#include <functional>
#include <string>

#include <EngineInterface/GenomeDesc.h>

#include "Definitions.h"
#include "ModalWindow.h"

// Modal dialog for editing mutation rates. It is processed by its owner, which allows it to be opened from within
// another dialog as well.
class MutationRatesDialog
{
public:
    MutationRatesDialog();

    static void loadSettings(MutationRatesDesc& mutationRates, std::string const& settingsPrefix);
    static void saveSettings(MutationRatesDesc const& mutationRates, std::string const& settingsPrefix);

    void open(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback);
    void process();

private:
    void processContent();
    void onAdopt();

    ModalWindow _modalWindow;
    MutationRatesDesc _mutation;
    std::function<void(MutationRatesDesc const&)> _onAdoptCallback;
};
