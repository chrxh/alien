#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/GenomeDesc.h>

#include "AlienDialog.h"
#include "Definitions.h"

class MutationRateDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(MutationRateDialog);

private:
    MutationRateDialog();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void openIntern() override;

    void onAdopt();

    MutationsDesc _mutation;
};
