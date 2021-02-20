#include "Gui/Settings.h"
#include "Gui/StringHelper.h"

#include "RandomMultiplierDialog.h"
#include "ui_RandomMultiplierDialog.h"


RandomMultiplierDialog::RandomMultiplierDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RandomMultiplierDialog)
{
    ui->setupUi(this);

	ui->randomizeAngleCheckBox->setChecked(
		GuiSettings::getSettingsValue(Const::RandomMulChangeAngleKey, Const::RandomMulChangeAngleDefault));
	ui->randomizeVelXcheckBox->setChecked(
		GuiSettings::getSettingsValue(Const::RandomMulChangeVelXKey, Const::RandomMulChangeVelXDefault));
	ui->randomizeVelYcheckBox->setChecked(
		GuiSettings::getSettingsValue(Const::RandomMulChangeVelYKey, Const::RandomMulChangeVelYDefault));
	ui->randomizeAngVelCheckBox->setChecked(
		GuiSettings::getSettingsValue(Const::RandomMulChangeAngVelKey, Const::RandomMulChangeAngVelDefault));

	ui->numberEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::RandomMulCopiesKey, Const::RandomMulCopiesDefault)));
	ui->minAngleEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::RandomMulMinAngleKey, Const::RandomMulMinAngleDefault)));
	ui->maxAngleEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::RandomMulMaxAngleKey, Const::RandomMulMaxAngleDefault)));
	ui->minVelXEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::RandomMulMinVelXKey, Const::RandomMulMinVelXDefault)));
	ui->maxVelXEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::RandomMulMaxVelXKey, Const::RandomMulMaxVelXDefault)));
	ui->minVelYEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::RandomMulMinVelYKey, Const::RandomMulMinVelYDefault)));
	ui->maxVelYEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::RandomMulMaxVelYKey, Const::RandomMulMaxVelYDefault)));
	ui->minAngVelEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::RandomMulMinAngVelKey, Const::RandomMulMinAngVelDefault)));
	ui->maxAngVelEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::RandomMulMaxAngVelKey, Const::RandomMulMaxAngVelDefault)));

	connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &RandomMultiplierDialog::okClicked);
}

RandomMultiplierDialog::~RandomMultiplierDialog()
{
    delete ui;
}

int RandomMultiplierDialog::getNumberOfCopies ()
{
    bool ok(true);
    return ui->numberEdit->text().toUInt(&ok);
}

bool RandomMultiplierDialog::isChangeAngle ()
{
    return ui->randomizeAngleCheckBox->isChecked();
}

double RandomMultiplierDialog::getAngleMin ()
{
    bool ok(true);
    return ui->minAngleEdit->text().toDouble(&ok);
}

double RandomMultiplierDialog::getAngleMax ()
{
    bool ok(true);
    return ui->maxAngleEdit->text().toDouble(&ok);
}

bool RandomMultiplierDialog::isChangeVelX ()
{
    return ui->randomizeVelXcheckBox->isChecked();
}

double RandomMultiplierDialog::getVelXMin ()
{
    bool ok(true);
    return ui->minVelXEdit->text().toDouble(&ok);
}

double RandomMultiplierDialog::getVelXMax ()
{
    bool ok(true);
    return ui->maxVelXEdit->text().toDouble(&ok);
}

bool RandomMultiplierDialog::isChangeVelY ()
{
    return ui->randomizeVelYcheckBox->isChecked();
}

double RandomMultiplierDialog::getVelYMin ()
{
    bool ok(true);
    return ui->minVelYEdit->text().toDouble(&ok);
}

double RandomMultiplierDialog::getVelYMax ()
{
    bool ok(true);
    return ui->maxVelYEdit->text().toDouble(&ok);
}

bool RandomMultiplierDialog::isChangeAngVel ()
{
    return ui->randomizeAngVelCheckBox->isChecked();
}

double RandomMultiplierDialog::getAngVelMin ()
{
    bool ok(true);
    return ui->minAngVelEdit->text().toDouble(&ok);
}

double RandomMultiplierDialog::getAngVelMax ()
{
    bool ok(true);
    return ui->maxAngVelEdit->text().toDouble(&ok);
}

void RandomMultiplierDialog::okClicked()
{
	GuiSettings::setSettingsValue(Const::RandomMulChangeAngleKey, isChangeAngle());
	GuiSettings::setSettingsValue(Const::RandomMulChangeVelXKey, isChangeVelX());
	GuiSettings::setSettingsValue(Const::RandomMulChangeVelYKey, isChangeVelY());
	GuiSettings::setSettingsValue(Const::RandomMulChangeAngVelKey, isChangeAngVel());

	GuiSettings::setSettingsValue(Const::RandomMulCopiesKey, getNumberOfCopies());
	GuiSettings::setSettingsValue(Const::RandomMulMinAngleKey, getAngleMin());
	GuiSettings::setSettingsValue(Const::RandomMulMaxAngleKey, getAngleMax());
	GuiSettings::setSettingsValue(Const::RandomMulMinVelXKey, getVelXMin());
	GuiSettings::setSettingsValue(Const::RandomMulMaxVelXKey, getVelXMax());
	GuiSettings::setSettingsValue(Const::RandomMulMinVelYKey, getVelYMin());
	GuiSettings::setSettingsValue(Const::RandomMulMaxVelYKey, getVelYMax());
	GuiSettings::setSettingsValue(Const::RandomMulMinAngVelKey, getAngVelMin());
	GuiSettings::setSettingsValue(Const::RandomMulMaxAngVelKey, getAngVelMax());

	accept();
}
