#include "MultiplyRandomDialog.h"
#include "ui_MultiplyRandomDialog.h"

#include "gui/Settings.h"

MultiplyRandomDialog::MultiplyRandomDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::MultiplyRandomDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
}

MultiplyRandomDialog::~MultiplyRandomDialog()
{
    delete ui;
}

int MultiplyRandomDialog::getNumber ()
{
    bool ok(true);
    return ui->numberEdit->text().toUInt(&ok);
}

bool MultiplyRandomDialog::randomizeAngle ()
{
    return ui->randomizeAngleCheckBox->isChecked();
}

qreal MultiplyRandomDialog::randomizeAngleMin ()
{
    bool ok(true);
    return ui->minAngleEdit->text().toDouble(&ok);
}

qreal MultiplyRandomDialog::randomizeAngleMax ()
{
    bool ok(true);
    return ui->maxAngleEdit->text().toDouble(&ok);
}

bool MultiplyRandomDialog::randomizeVelX ()
{
    return ui->randomizeVelXcheckBox->isChecked();
}

qreal MultiplyRandomDialog::randomizeVelXMin ()
{
    bool ok(true);
    return ui->minVelXEdit->text().toDouble(&ok);
}

qreal MultiplyRandomDialog::randomizeVelXMax ()
{
    bool ok(true);
    return ui->maxVelXEdit->text().toDouble(&ok);
}

bool MultiplyRandomDialog::randomizeVelY ()
{
    return ui->randomizeVelYcheckBox->isChecked();
}

qreal MultiplyRandomDialog::randomizeVelYMin ()
{
    bool ok(true);
    return ui->minVelYEdit->text().toDouble(&ok);
}

qreal MultiplyRandomDialog::randomizeVelYMax ()
{
    bool ok(true);
    return ui->maxVelYEdit->text().toDouble(&ok);
}

bool MultiplyRandomDialog::randomizeAngVel ()
{
    return ui->randomizeAngVelcheckBox->isChecked();
}

qreal MultiplyRandomDialog::randomizeAngVelMin ()
{
    bool ok(true);
    return ui->minAngVelEdit->text().toDouble(&ok);
}

qreal MultiplyRandomDialog::randomizeAngVelMax ()
{
    bool ok(true);
    return ui->maxAngVelEdit->text().toDouble(&ok);
}
