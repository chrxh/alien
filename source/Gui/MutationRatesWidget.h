#pragma once

struct MutationRatesDesc;

class MutationRatesWidget
{
public:
    static void process(MutationRatesDesc& mutationRates, bool nested = false);
};
