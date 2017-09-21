#include "gui/GuiSettings.h"
#include "model/context/SimulationParameters.h"

#include "addrectstructuredialog.h"
#include "ui_addrectstructuredialog.h"


AddRectStructureDialog::AddRectStructureDialog(SimulationParameters* simulationParameters, QWidget *parent)
	: QDialog(parent), ui(new Ui::AddRectStructureDialog)
{
    ui->setupUi(this);
    setFont(GuiFunctions::getGlobalFont());
    ui->energyEdit->setText(QString("%1").arg(simulationParameters->cellCreationEnergy));
}

AddRectStructureDialog::~AddRectStructureDialog()
{
    delete ui;
}

int AddRectStructureDialog::getBlockSizeX ()
{
    bool ok(true);
    return ui->sizeXEdit->text().toInt(&ok);
}

int AddRectStructureDialog::getBlockSizeY ()
{
    bool ok(true);
    return ui->sizeYEdit->text().toInt(&ok);
}

qreal AddRectStructureDialog::getDistance ()
{
    bool ok(true);
    return ui->distEdit->text().toDouble(&ok);
}

qreal AddRectStructureDialog::getInternalEnergy ()
{
    bool ok(true);
    return ui->energyEdit->text().toDouble(&ok);
}


