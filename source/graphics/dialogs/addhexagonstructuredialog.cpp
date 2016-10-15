#include "addhexagonstructuredialog.h"
#include "ui_addhexagonstructuredialog.h"

#include "../../globaldata/globalfunctions.h"
#include "../../globaldata/simulationsettings.h"

AddHexagonStructureDialog::AddHexagonStructureDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::AddHexagonStructureDialog)
{
    ui->setupUi(this);
    setFont(GlobalFunctions::getGlobalFont());
    ui->energyEdit->setText(QString("%1").arg(simulationParameters.NEW_CELL_ENERGY));
}

AddHexagonStructureDialog::~AddHexagonStructureDialog()
{
    delete ui;
}

int AddHexagonStructureDialog::getLayers ()
{
    bool ok(true);
    return ui->layersEdit->text().toInt(&ok);
}

qreal AddHexagonStructureDialog::getDistance ()
{
    bool ok(true);
    return ui->distEdit->text().toDouble(&ok);
}

qreal AddHexagonStructureDialog::getInternalEnergy ()
{
    bool ok(true);
    return ui->energyEdit->text().toDouble(&ok);
}
