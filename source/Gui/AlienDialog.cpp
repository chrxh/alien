#include "AlienDialog.h"

AlienDialog::AlienDialog(std::string const& title, RealVector2D const& defaultSize, bool maximizable)
    : _modalWindow(title, defaultSize, maximizable)
{}

void AlienDialog::init()
{
    initIntern();
}

void AlienDialog::open()
{
    _modalWindow.open();
    openIntern();
}

bool AlienDialog::isOpen() const
{
    return _modalWindow.isOpen();
}

void AlienDialog::close()
{
    delayedExecution([this] { _modalWindow.close(); });
}

void AlienDialog::changeTitle(std::string const& title)
{
    _modalWindow.setTitle(title);
}

void AlienDialog::process()
{
    _modalWindow.process([this] { processIntern(); });
}

void AlienDialog::shutdown()
{
    shutdownIntern();
}
