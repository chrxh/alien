#pragma once

#include "MutationRatesDialog.h"

struct MutationRatesDesc;

class MutationRatesWidget
{
public:
    void process(MutationRatesDesc& mutationRates, float rightColumnWidth, bool disabled = false);

    // A single row with the edit button, the active mutation types are listed in its tooltip
    void processSummary(MutationRatesDesc& mutationRates, float rightColumnWidth);

private:
    MutationRatesDialog _dialog;
};
