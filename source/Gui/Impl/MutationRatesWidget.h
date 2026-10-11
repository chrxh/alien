#pragma once

#include "MutationRatesDialog.h"

struct MutationRatesDesc;

class MutationRatesWidget
{
public:
    void process(MutationRatesDesc& mutationRates, float rightColumnWidth, bool disabled = false);
    void processAsSingleRow(MutationRatesDesc& mutationRates, float rightColumnWidth);

private:
    MutationRatesDialog _dialog;
};
