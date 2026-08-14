#pragma once

#include <string>

#include "Base/Singleton.h"

#include "AlienDialog.h"
#include "Definitions.h"

class SavePictureDialog : public AlienDialog<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SavePictureDialog);

private:
    SavePictureDialog();

    void initIntern() override;
    void shutdownIntern() override;
    void openIntern() override;
    void processIntern() override;

    void onSavePicture();

    int _width = 0;
    int _height = 0;
    std::string _referencePath;
};
