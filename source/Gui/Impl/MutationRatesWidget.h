#pragma once

#include "MutationRatesDialog.h"

struct MutationRatesDesc;

class MutationRatesWidget
{
public:
    void process(MutationRatesDesc& mutationRates, float rightColumnWidth, bool disabled = false);

private:
    MutationRatesDialog _dialog;
};
