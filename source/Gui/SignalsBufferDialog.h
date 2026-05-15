#pragma once

#include <functional>
#include <variant>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/GenomeDesc.h>

#include "AlienDialog.h"
#include "Definitions.h"

class SignalsBufferDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SignalsBufferDialog);

public:
    void open(std::vector<SignalEntryDesc> const& entries, std::function<void(std::vector<SignalEntryDesc> const&)> const& onAdoptCallback);
    void open(std::vector<SignalEntryGenomeDesc> const& entries, std::function<void(std::vector<SignalEntryGenomeDesc> const&)> const& onAdoptCallback);

private:
    SignalsBufferDialog();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void openIntern() override;

    void onAdopt();

    std::vector<std::vector<float>> _channelsBuffer;
    int _selectedEntry = 0;

    using AdoptCallback = std::variant<
        std::function<void(std::vector<SignalEntryDesc> const&)>,
        std::function<void(std::vector<SignalEntryGenomeDesc> const&)>>;
    std::optional<AdoptCallback> _onAdoptCallback;
};
