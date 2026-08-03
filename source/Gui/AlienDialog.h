#pragma once

#include <imgui.h>

#include "Definitions.h"
#include "DelayedExecutionController.h"
#include "MainLoopEntity.h"
#include "MainLoopEntityController.h"
#include "ModalWindow.h"
#include "StyleRepository.h"
#include "WindowController.h"

class AlienDialog : public MainLoopEntity
{
public:
    AlienDialog(std::string const& title, RealVector2D const& defaultSize = RealVector2D(450.0f, 150.0f), bool maximizable = false);

    virtual void open();
    bool isOpen() const;

protected:
    virtual void processIntern() {}
    virtual void initIntern() {}
    virtual void shutdownIntern() {}

    virtual void openIntern() {}

    void changeTitle(std::string const& title);
    virtual void close();

private:
    void init() override;
    void process() override;
    void shutdown() override;

    ModalWindow _modalWindow;
};
