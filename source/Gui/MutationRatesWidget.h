#pragma once

struct MutationRatesDesc;

class MutationRatesWidget
{
public:
    static void process(MutationRatesDesc& mutationRates, float rightColumnWidth, bool nested = false);
};
