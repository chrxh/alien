#include "RandomMultiplierDialog.h"
#include "ui_RandomMultiplierDialog.h"

#include "gui/Settings.h"

RandomMultiplierDialog::RandomMultiplierDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RandomMultiplierDialog)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
}

RandomMultiplierDialog::~RandomMultiplierDialog()
{
    delete ui;
}

int RandomMultiplierDialog::getNumber ()
{
    bool ok(true);
    return ui->numberEdit->text().toUInt(&ok);
}

bool RandomMultiplierDialog::randomizeAngle ()
{
    return ui->randomizeAngleCheckBox->isChecked();
}

qreal RandomMultiplierDialog::randomizeAngleMin ()
{
    bool ok(true);
    return ui->minAngleEdit->text().toDouble(&ok);
}

qreal RandomMultiplierDialog::randomizeAngleMax ()
{
    bool ok(true);
    return ui->maxAngleEdit->text().toDouble(&ok);
}

bool RandomMultiplierDialog::randomizeVelX ()
{
    return ui->randomizeVelXcheckBox->isChecked();
}

qreal RandomMultiplierDialog::randomizeVelXMin ()
{
    bool ok(true);
    return ui->minVelXEdit->text().toDouble(&ok);
}

qreal RandomMultiplierDialog::randomizeVelXMax ()
{
    bool ok(true);
    return ui->maxVelXEdit->text().toDouble(&ok);
}

bool RandomMultiplierDialog::randomizeVelY ()
{
    return ui->randomizeVelYcheckBox->isChecked();
}

qreal RandomMultiplierDialog::randomizeVelYMin ()
{
    bool ok(true);
    return ui->minVelYEdit->text().toDouble(&ok);
}

qreal RandomMultiplierDialog::randomizeVelYMax ()
{
    bool ok(true);
    return ui->maxVelYEdit->text().toDouble(&ok);
}

bool RandomMultiplierDialog::randomizeAngVel ()
{
    return ui->randomizeAngVelcheckBox->isChecked();
}

qreal RandomMultiplierDialog::randomizeAngVelMin ()
{
    bool ok(true);
    return ui->minAngVelEdit->text().toDouble(&ok);
}

qreal RandomMultiplierDialog::randomizeAngVelMax ()
{
    bool ok(true);
    return ui->maxAngVelEdit->text().toDouble(&ok);
}
